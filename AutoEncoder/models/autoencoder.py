import torch
import torch.nn as nn
from torch import Tensor
import pytorch_lightning as pl
import torch.nn.functional as F
from .shared_encoder import Shared_Encoder
from .shared_decoder import Shared_Decoder

"""对是否加入InfoNCE加入条件控制"""
"""Encoder需要多次复用 only_latent/return_latent"""

class PreTrained_AutoEncoder(pl.LightningModule):
    def __init__(self, lr, lambda_rec, lambda_nce, proj_dim, temperature, input_size = [1, 256, 256]):
        super().__init__()
        self.encoder = Shared_Encoder()
        self.decoder = Shared_Decoder()
        self.lr = lr
        self.temperature = temperature
        self.lambda_rec = lambda_rec
        self.lambda_nce = lambda_nce
        self.save_hyperparameters()  ## Optional

        feat_c = None
        for m in reversed(list(self.encoder.net)):
            if isinstance(m, nn.Conv2d):
                feat_c = m.out_channels
                break
        if feat_c is None:
            raise RuntimeError("Cannot infer encoder output channels; please set 'feat_c' manually.")
        
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, C_out, 1, 1)
            nn.Flatten(),                  # (B, C_out)
            nn.Linear(feat_c, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)  
        )
    def _check_input(self, x: Tensor):
        # dtype check
        if x.dtype != torch.float32:
            x = x.float()

        # channel check
        expected_in = self.encoder.net[0].in_channels
        if x.shape[1] != expected_in:
            raise ValueError(f"Expected {expected_in} input channels, got {x.shape[1]}")

        # divisibility check
        H, W = x.shape[2], x.shape[3]
        if (H % 16) != 0 or (W % 16) != 0:
            raise ValueError(f"Input H,W must be divisible by 16, got H={H}, W={W}")

        # device check
        if x.device != self.device:
            x = x.to(self.device)

        return x

    def forward(self, x, return_latent=False, only_latent=False):
        """
        Args:
            x: 输入张量
            return_latent: 如果 True，返回 (latent, recon)
            only_latent: 如果 True，只返回 latent
        """
        x = self._check_input(x)
        z = self.encoder(x)
        if only_latent:
            return z
        out = self.decoder(z)
        if return_latent:
            return z, out
        return out

    
    def _compute_projection(self, feat:torch.Tensor):
        """feat: (B, C, H, W) -> returns normalized projection (B, proj_dim)"""
        h = self.projector(feat)            # (B, proj_dim)
        
        h = F.normalize(h, dim=1)          # L2 normalize
        return h
    
    def training_step(self, batch, batch_idx):
        # batch = (x_anchor, x_positive)
        x_anchor, x_pos = batch
        assert x_anchor.dim() == 4 and x_pos.dim() == 4, \
            f"Expected 4D tensors (B,C,H,W), got {x_anchor.shape}, {x_pos.shape}"
        
        x_anchor = self._check_input(x_anchor)
        x_pos = self._check_input(x_pos)
        B = x_anchor.size(0)

        # reconstruction
        z_anchor = self.encoder(x_anchor)
        z_pos = self.encoder(x_pos)
        recon_anchor = self.decoder(z_anchor)
        recon_pos = self.decoder(z_pos)
        rec_loss = F.mse_loss(recon_anchor, x_anchor) + F.mse_loss(recon_pos, x_pos)

        loss = self.lambda_rec * rec_loss

        # InfoNCE (only if enabled)
        nce_loss = torch.tensor(0.0, device=x_anchor.device)
        if self.lambda_nce > 0 and B >= 2:
            h_anchor = self._compute_projection(z_anchor)
            h_pos = self._compute_projection(z_pos)
            logits = torch.matmul(h_anchor, h_pos.T) / self.temperature
            labels = torch.arange(B, device=logits.device, dtype=torch.long)
            loss_a2p = F.cross_entropy(logits, labels)
            loss_p2a = F.cross_entropy(logits.T, labels)
            nce_loss = 0.5 * (loss_a2p + loss_p2a)
            loss = loss + self.lambda_nce * nce_loss

        # logging
        self.log_dict({
            "train_loss": loss,
            "train_rec_loss": rec_loss,
            "train_nce_loss": nce_loss
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss




    def validation_step(self, batch, batch_idx):
        # batch = (x_anchor, x_positive)
        x_anchor, x_pos = batch
        assert x_anchor.dim() == 4 and x_pos.dim() == 4, \
            f"Expected 4D tensors (B,C,H,W), got {x_anchor.shape}, {x_pos.shape}"
        
        x_anchor = self._check_input(x_anchor)
        x_pos = self._check_input(x_pos)
        B = x_anchor.size(0)

        z_anchor = self.encoder(x_anchor)
        z_pos = self.encoder(x_pos)
        recon_anchor = self.decoder(z_anchor)
        recon_pos = self.decoder(z_pos)
        rec_loss = F.mse_loss(recon_anchor, x_anchor) + F.mse_loss(recon_pos, x_pos)
        loss = self.lambda_rec * rec_loss

        nce_loss = torch.tensor(0.0, device=x_anchor.device)
        if self.lambda_nce > 0 and B >= 2:
            h_anchor = self._compute_projection(z_anchor)
            h_pos = self._compute_projection(z_pos)
            logits = torch.matmul(h_anchor, h_pos.T) / self.temperature
            labels = torch.arange(B, device=logits.device, dtype=torch.long)
            loss_a2p = F.cross_entropy(logits, labels)
            loss_p2a = F.cross_entropy(logits.T, labels)
            nce_loss = 0.5 * (loss_a2p + loss_p2a)
            loss = loss + self.lambda_nce * nce_loss

        # logging
        self.log_dict({
            "val_loss": loss,
            "val_rec_loss": rec_loss,
            "val_nce_loss": nce_loss
        }, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)
        return {"optimizer": optim, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}