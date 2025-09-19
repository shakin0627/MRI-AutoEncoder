import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import lightning as pl
from shared_encoder import Shared_Encoder # (B'* ,1,256,256) -> (B',256,16,16)
from fusion_transformer import FusionTransformer
from flow_matching_unet import UNetFlow

class ManiflowNet(pl.LightningModule):
    """
    Lightning wrapper that glues Shared_Encoder, FusionTransformer and UNetFlow together
    and trains with flow-matching loss(+ Contrastive loss).

    Expected batch:
      - "imgs": Tensor (B, N, 1, H, W)  (dataset should produce full-modal set; random masking applied)
      - optional "cond_mask": Tensor (B, N) bool/int (1 = visible, 0 = masked). If None, all visible.
    """
    def __init__(self, 
                 lr:float=1e-4,
                 C_cond:int=128,
                 N_slots: int = 4, ## 模态数max_N
                 remove_source_from_cond: bool=True, ## 检查是否与CrossAttention Injector的注入机制冲突
                 fusion_cfg: dict = None,
                 fusion_enabled: bool = True,
                 eps_t: float = 1e-3,
        ):
        super().__init__()
        self.lr = lr
        self.C_cond = C_cond
        self.save_hyperparameters()

        self.encoder = Shared_Encoder()
        self.fusion = FusionTransformer(
            C_tok=256, C_cond=C_cond, max_N=N_slots, **(fusion_cfg or {})
        )
        self.unet = UNetFlow(
            in_channels=256,
            base_channels=32,
            time_emb_dim=64,
            depth=3,
            use_cross_attn=True,
            cond_dim=C_cond,
        )

        # configs
        self.remove_source_from_cond = bool(remove_source_from_cond)
        self.set_fusion_enabled(bool(fusion_enabled))

        self.eps_t = float(eps_t)
        self.loss_fn = nn.MSELoss()

    def _make_cond_mask(self, cond_mask, B, N, device):
        if cond_mask is None:
            return torch.ones(B, N, dtype=torch.bool, device=device)
        return cond_mask.to(device=device).to(dtype=torch.bool)
    
    def _encode_all_modalities(self, imgs):
        """
        imgs: (B, N, 1, H, W)
        returns enc: (B, N, C_enc, Hp, Wp)
        """
        B, N, C, H, W = imgs.shape
        imgs_flat = imgs.view(B * N, C, H, W)                  # (B*N,1,H,W)
        enc_flat = self.encoder(imgs_flat)                  # (B*N,256,Hp,Wp) Hp=Wp=16
        C_enc, Hp, Wp = enc_flat.shape[1], enc_flat.shape[2], enc_flat.shape[3]
        enc = enc_flat.view(B, N, C_enc, Hp, Wp)               # (B,N,256,Hp,Wp) Hp=Wp=16
        return enc
    def _make_tokens_from_enc(self, enc):
        # enc: (B,N,256,Hp,Wp) -> tokens: (B,N,256)
        return enc.mean(dim=[3,4])
    
    def _choose_source_target_indices(self, cond_mask):
        """
        cond_mask: (B, N) bool (1 visible, 0 masked)
        Returns:
          source_idx: (B,) long  (selected from visible indices)
          target_idx: (B,) long  (selected from masked indices)
        If a sample has no masked index, force-mask a random slot for that sample.
        """
        B, N = cond_mask.shape
        device = cond_mask.device
        source_idx = torch.empty(B, dtype=torch.long, device=device)
        target_idx = torch.empty(B, dtype=torch.long, device=device)

        for b in range(B):
            avail = torch.nonzero(cond_mask[b], as_tuple=True)[0]  
            miss  = torch.nonzero(~cond_mask[b], as_tuple=True)[0]

            # if no missing modal, force mask a random slot
            if miss.numel() == 0:
                forced = torch.randint(0, N, (1,), device=device).long()
                # make miss = [forced], avail = others
                miss = forced
                avail = torch.tensor([i for i in range(N) if i != int(forced)], device=device, dtype=torch.long)
                if avail.numel() == 0:
                    # corner case: N==1 -> both source and target are same slot (degenerate)
                    avail = forced

            # if no available (all masked) then force pick one available (should not happen if dataset full and mask created properly)
            if avail.numel() == 0:
                forced_av = torch.randint(0, N, (1,), device=device).long()
                avail = forced_av

            # pick random from avail / miss
            s = avail[torch.randint(0, avail.numel(), (1,), device=device)]
            t = miss[torch.randint(0, miss.numel(), (1,), device=device)]
            source_idx[b] = s
            target_idx[b] = t

        return source_idx, target_idx
    
    def forward(self, imgs, cond_mask=None, source_idx=None, t=None, target_idx=None):
        B, N, _, H, W = imgs.shape
        device = imgs.device
        # encode all modalities
        enc = self._encode_all_modalities(imgs)
        tokens = self._make_tokens_from_enc(enc)
        cond_mask = self._make_cond_mask(cond_mask, B, N, device) 
        cond_feats = self.fusion(tokens, cond_mask)

        if source_idx is None or target_idx is None:
            s_idx, t_idx = self._choose_source_target_indices(cond_mask)
            source_idx = source_idx if source_idx is not None else s_idx
            target_idx = target_idx if target_idx is not None else t_idx

        # build cond_mask2 & optionally remove source token from cond_feats
        cond_mask2 = cond_mask.clone()
        if cond_feats is not None and self.remove_source_from_cond:
            batch_idx = torch.arange(B, device=device)
            cond_mask2[batch_idx, source_idx] = False
            cond_feats = cond_feats.clone()
            cond_feats[batch_idx, source_idx, :] = 0.0 ## 将源模态设为不可见, 假定了 cond_feats 是 3D (B,N,C_cond)

        batch_idx = torch.arange(B, device=device)
        z_start = enc[batch_idx, source_idx]   # (B,256,Hp,Wp)
        z_end   = enc[batch_idx, target_idx]   # (B,256,Hp,Wp)

        # sample time t if not provided
        if t is None:
            t = torch.rand(B, device=device)
        # clamp to avoid 0 or 1
        t = t.clamp(self.eps_t, 1.0 - self.eps_t)   # (B,)

        # construct path point x_t and v_target
        t_view = t.view(B, 1, 1, 1)
        x_t = (1.0 - t_view) * z_start + t_view * z_end
        v_target = (z_end - z_start) / (1.0 - t_view)   # (B,256,Hp,Wp)

        # call UNet to predict v_pred
        v_pred = self.unet(x_t, cond_feats, t, cond_mask2)

        meta = dict(
            enc=enc,
            tokens=tokens,
            cond_feats=cond_feats,
            cond_mask=cond_mask2,
            z_start=z_start,
            z_end=z_end,
            x_t=x_t,
            v_target=v_target,
            source_idx=source_idx,
            target_idx=target_idx,
            t=t
        )
        return v_pred, meta
        
    def training_step(self, batch, batch_idx):
        imgs = batch['imgs']                       # (B,N,1,H,W)
        cond_mask = batch.get('cond_mask', None)  
        device = imgs.device
        B, N = imgs.shape[0], imgs.shape[1]

        v_pred, meta = self.forward(imgs, cond_mask=cond_mask)
        v_target = meta['v_target']

        assert v_pred.shape == v_target.shape, f"v_pred {v_pred.shape} vs v_target {v_target.shape}"

        loss = self.loss_fn(v_pred, v_target)
        self.log("train loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs = batch['imgs']
        cond_mask = batch.get('cond_mask', None)

        v_pred, meta = self.forward(imgs, cond_mask=cond_mask)
        v_target = meta['v_target']

        assert v_pred.shape == v_target.shape, f"v_pred {v_pred.shape} vs v_target {v_target.shape}"

        loss = self.loss_fn(v_pred, v_target)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss.detach()}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    # ----------------- Ablation toggles -----------------
    def set_fusion_enabled(self, flag: bool):
        self.fusion.set_enabled(flag)
        # if fusion disabled, also disable UNet cross-attn to avoid expecting cond_feats
        self.unet.set_cross_attn_enabled(flag)

    def set_remove_source_from_cond(self, flag: bool):
        self.remove_source_from_cond = bool(flag)



# 测试/验证/消融 时把模型设为 eval() 并用 torch.no_grad()
# 在 开发/CI smoke-test 使用缩小版 UNet（例如 base_channels=8, depth=1）以快速验证接口与行为；在真实训练时再换回完整版并在 GPU 上运行