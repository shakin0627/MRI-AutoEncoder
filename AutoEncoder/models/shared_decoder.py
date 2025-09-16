import torch
import torch.nn as nn
from torch import Tensor

class Shared_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 256, 16, 16)
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # (B, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # (B, 64, 64, 64)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # (B, 32, 128, 128)
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),    # (B, 16, 256, 256)
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),   # (B, 1, 256, 256)
            nn.Sigmoid()  
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype != torch.float32:
            x = x.float()

        # divisibility check
        H, W = x.shape[2], x.shape[3]
        if (H % 16) != 0 or (W % 16) != 0:
            raise ValueError(f"Input H,W must be divisible by 16, got H={H}, W={W}")

        # device check
        if next(self.parameters()).device != x.device:
            raise RuntimeError(
                f"Model on {next(self.parameters()).device}, input on {x.device}")
        return self.net(x)