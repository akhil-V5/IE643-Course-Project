from __future__ import annotations
import torch


def disease_weighted_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lesion: torch.Tensor,
    valid: torch.Tensor,
    lesion_multiplier: float = 5.0,
) -> torch.Tensor:
    """Return a per-image loss; ignore pixels contribute zero to this auxiliary loss.

    RGB inputs are in [-1, 1]. Absolute error is divided by two to express it in
    [0, 1] image units (predictions are NOT clipped during training). Normalize
    each image separately so lesion-heavy images do not dominate the batch.
    The native latent ResShift MSE remains a separate, unmasked objective.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Prediction and target must be equal N,C,H,W tensors")
    shape = (prediction.shape[0], 1, *prediction.shape[-2:])
    if lesion.shape != shape or valid.shape != shape:
        raise ValueError(f"Expected masks shaped {shape}")
    if lesion_multiplier < 1:
        raise ValueError("lesion_multiplier must be >= 1")
    weight = valid.float() * (1.0 + (lesion_multiplier - 1.0) * lesion.float())
    error = (prediction.float() - target.float()).abs().mean(dim=1, keepdim=True) / 2.0
    denom = weight.sum(dim=(1, 2, 3))
    if torch.any(denom <= 0):
        raise ValueError("An entire patch is ignored: reject it in data preparation")
    return (error * weight).sum(dim=(1, 2, 3)) / denom
