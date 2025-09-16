import torch
import torch.nn as nn
from torch import Tensor

class Shared_Encoder(nn.Module):
    def __init__(self):
        super(Shared_Encoder, self).__init__()
        self.net = nn.Sequential(
            # 输入: (B, 1, 256, 256)
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),   # (B, 32, 256, 256)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # (B, 32, 128, 128)

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (B, 64, 128, 128)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # (B, 64, 64, 64)

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # (B, 128, 64, 64)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # (B, 128, 32, 32)

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# (B, 256, 32, 32)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                     # (B, 256, 16, 16)
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dtype != torch.float32:
            x = x.float()  # + optionally scale: x = x/255.0 if raw images

        expected_in = self.net[0].in_channels
        if x.shape[1] != expected_in:
            raise ValueError(f"Expected input with {expected_in} channels, got {x.shape[1]}")

        H, W = x.shape[2], x.shape[3]
        if (H % 16) != 0 or (W % 16) != 0:
            raise ValueError(f"Input H and W must be divisible by 16. Got H={H}, W={W}. ""Either resize inputs or change architecture (e.g. handle output_padding).")

        # Ensure device alignment (if you do manual .to calls)
        if next(self.parameters()).device != x.device:
            # either move x or move model -- choose one policy; here we raise for clarity
            raise RuntimeError(f"Model is on {next(self.parameters()).device}, but input is on {x.device}")
        
        return self.net(x)