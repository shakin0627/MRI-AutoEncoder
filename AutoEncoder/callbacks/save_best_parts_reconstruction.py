import os
import torch
from torchvision.utils import make_grid, save_image
from pytorch_lightning.callbacks import Callback

class SaveBestPartsAndReconstruction(Callback):
    def __init__(self, monitor="val_loss", save_dir="./checkpoints", n_samples=4):
        super().__init__()
        self.monitor = monitor
        self.save_dir = save_dir
        self.best_score = float("inf")
        self.n_samples = n_samples
        os.makedirs(self.save_dir, exist_ok=True)

    def _normalize(self, tensor):
        # normalization
        batch_size = tensor.size(0)
        tensor_flat = tensor.view(batch_size, -1)
        min_vals = tensor_flat.min(dim=1, keepdim=True)[0]
        max_vals = tensor_flat.max(dim=1, keepdim=True)[0]
        original_shape = tensor.shape
        for _ in range(len(original_shape) - 2):
            min_vals = min_vals.unsqueeze(-1)
            max_vals = max_vals.unsqueeze(-1)
        normalized = (tensor - min_vals) / (max_vals - min_vals + 1e-8)
        return torch.clamp(normalized, 0.0, 1.0)

    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            val_loader = trainer.datamodule.val_dataloader()
            if val_loader is None or len(val_loader) == 0:
                print("Warning: Empty validation loader, skipping reconstruction save")
                return

            batch = next(iter(val_loader))
            if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                print("Warning: Invalid batch format, expected (x_anchor, x_pos)")
                return

            x_anchor, x_pos = batch
            n_samples = min(self.n_samples, x_anchor.size(0))
            x_anchor = x_anchor[:n_samples].to(pl_module.device)
            x_pos    = x_pos[:n_samples].to(pl_module.device)

            if len(x_anchor.shape) == 3: x_anchor = x_anchor.unsqueeze(1)
            if len(x_pos.shape) == 3:    x_pos = x_pos.unsqueeze(1)

            pl_module.eval()
            with torch.no_grad():
                recon_anchor = pl_module(x_anchor)
                recon_pos    = pl_module(x_pos)
            pl_module.train()

            x_anchor_norm = self._normalize(x_anchor)
            recon_anchor_norm = self._normalize(recon_anchor)
            x_pos_norm = self._normalize(x_pos)
            recon_pos_norm = self._normalize(recon_pos)

            comparison = torch.cat([x_anchor_norm, recon_anchor_norm, x_pos_norm, recon_pos_norm], dim=0)
            grid = make_grid(comparison, nrow=n_samples, normalize=False, padding=2)

            save_path = os.path.join(self.save_dir, f"reconstruction_epoch{trainer.current_epoch:03d}.png")
            save_image(grid, save_path)
            print(f" Saved reconstruction comparison to {save_path}")

        except Exception as e:
            print(f" Error in reconstruction callback: {e}")
            return

        try:
            metrics = trainer.callback_metrics
            if self.monitor in metrics:
                current_score = metrics[self.monitor].item()
                if current_score < self.best_score:
                    self.best_score = current_score
                    if hasattr(pl_module, 'encoder') and hasattr(pl_module, 'decoder'):
                        torch.save(pl_module.encoder.state_dict(), os.path.join(self.save_dir, "encoder_best.pth"))
                        torch.save(pl_module.decoder.state_dict(), os.path.join(self.save_dir, "decoder_best.pth"))
                        print(f" Saved best encoder/decoder at {self.save_dir} (val_loss={current_score:.4f})")
                    else:
                        torch.save(pl_module.state_dict(), os.path.join(self.save_dir, "model_best.pth"))
                        print(f" Saved best model at {self.save_dir} (val_loss={current_score:.4f})")
        except Exception as e:
            print(f" Error saving best model: {e}")
