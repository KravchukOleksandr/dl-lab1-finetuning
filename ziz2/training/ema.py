"""Exponential moving average used for stable validation and inference."""

from __future__ import annotations

import copy

import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.997) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(
                    model_value.to(dtype=ema_value.dtype),
                    alpha=1.0 - self.decay,
                )
            else:
                ema_value.copy_(model_value)
