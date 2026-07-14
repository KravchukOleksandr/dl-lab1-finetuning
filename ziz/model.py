"""
MicroNeXt-based binary classifier.

Class mapping:
    0 = helmet     (каска есть)
    1 = no_helmet  (каски нет; positive class)

The architecture deliberately stays very close to the successful segmentation
network: same channel widths, block counts, depthwise 7x7 convolutions, GRN and
two stride-2 reductions. The former 1-channel mask map is averaged into one
classification logit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GRN(nn.Module):
    """Global Response Normalization for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MicroNeXtBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel: int = 7,
        expansion: int = 2,
    ) -> None:
        super().__init__()
        hidden = channels * expansion
        padding = kernel // 2

        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.GELU()
        self.grn = GRN(hidden)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dw(x)
        x = self.bn(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pw2(x)
        return x + shortcut


class HelmetMicroNeXt(nn.Module):
    """
    Input:
        [B, 3, 64, 64]

    Output:
        [B] logits, where a larger value means class 1 (no_helmet).
    """

    def __init__(self) -> None:
        super().__init__()

        # 64x64 -> 32x32
        self.stem = ConvBNAct(3, 16, kernel=3, stride=2)

        self.stage1 = nn.Sequential(
            MicroNeXtBlock(16, kernel=7, expansion=2),
        )

        # 32x32 -> 16x16
        self.down = ConvBNAct(16, 24, kernel=3, stride=2)

        self.stage2 = nn.Sequential(
            MicroNeXtBlock(24, kernel=7, expansion=2),
            MicroNeXtBlock(24, kernel=7, expansion=2),
            MicroNeXtBlock(24, kernel=7, expansion=2),
        )

        self.proj = nn.Sequential(
            nn.Conv2d(24, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_logit_map: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down(x)
        x = self.stage2(x)
        x = self.proj(x)

        logit_map = self.head(x)  # [B, 1, 16, 16]
        logits = logit_map.mean(dim=(2, 3)).squeeze(1)

        if return_logit_map:
            return logits, logit_map
        return logits


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
