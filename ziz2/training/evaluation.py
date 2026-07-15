"""Shared full-validation pass and checkpoint weight loading."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from training.metrics import validation_metrics


def load_model_weights(model: torch.nn.Module, checkpoint: Any) -> None:
    """Load either a full training checkpoint or a raw state_dict."""
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint contains neither a model field nor a state_dict")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    amp_enabled: bool,
    dataset_names: Sequence[str],
) -> dict[str, Any]:
    model.eval()
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    all_losses: list[np.ndarray] = []
    all_dataset_ids: list[np.ndarray] = []

    for images, targets, dataset_ids in loader:
        images = images.to(device, non_blocking=True)
        targets_device = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            logits = model(images)
            losses = functional.binary_cross_entropy_with_logits(
                logits,
                targets_device,
                reduction="none",
            )
        all_logits.append(logits.float().cpu().numpy())
        all_targets.append(targets.numpy())
        all_losses.append(losses.float().cpu().numpy())
        all_dataset_ids.append(dataset_ids.numpy())

    if not all_logits:
        raise ValueError("Validation loader is empty")
    return validation_metrics(
        logits=np.concatenate(all_logits),
        targets=np.concatenate(all_targets).astype(np.int64),
        losses=np.concatenate(all_losses),
        dataset_ids=np.concatenate(all_dataset_ids),
        dataset_names=dataset_names,
    )
