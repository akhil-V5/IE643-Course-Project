"""Whole-image letterboxing. No image region is cropped away."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .classes import IGNORE_INDEX

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

@dataclass
class Geometry:
    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    left: int
    top: int
    canvas: int

    def asdict(self):
        return asdict(self)


def letterbox(image: Image.Image, mask: np.ndarray | None, size: int):
    if size < 32 or size % 32:
        raise ValueError('image_size must be a positive multiple of 32, at least 32')
    w, h = image.size
    if mask is not None and mask.shape != (h, w):
        raise ValueError(f'Image/mask mismatch: image width,height={w,h}; mask height,width={mask.shape}')
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas = np.broadcast_to(MEAN, (size, size, 3)).copy()
    resized = image.convert('RGB').resize((nw, nh), Image.Resampling.BILINEAR)
    canvas[top:top+nh, left:left+nw] = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(((canvas - MEAN) / STD).transpose(2, 0, 1).copy())
    target = None
    if mask is not None:
        target = np.full((size, size), IGNORE_INDEX, dtype=np.int64)
        # Nearest-neighbour interpolation preserves categorical integer IDs.
        target[top:top+nh, left:left+nw] = np.asarray(
            Image.fromarray(mask.astype(np.uint8)).resize((nw, nh), Image.Resampling.NEAREST))
        target = torch.from_numpy(target)
    return tensor, target, Geometry(w, h, nw, nh, left, top, size)


def native_prediction(logits: torch.Tensor, geometry: Geometry, chunk_size=8):
    """Restore scores to the original image grid BEFORE argmax/softmax.

    Chunking avoids materializing all 116 full-resolution probability maps.
    Returns class IDs, P(any disease)=1-P(background), and max-class probability.
    """
    if logits.ndim == 4:
        if logits.shape[0] != 1:
            raise ValueError('Native restoration expects one image at a time')
        logits = logits[0]
    if logits.ndim != 3 or chunk_size < 1:
        raise ValueError('Expected C,H,W logits and positive chunk_size')
    logits = logits.detach().float().cpu()
    if not torch.isfinite(logits).all():
        raise FloatingPointError('Non-finite model logits')
    g = geometry
    shape = (g.original_height, g.original_width)
    best = torch.full(shape, -float('inf'))
    labels = torch.zeros(shape, dtype=torch.long)
    lse = torch.full(shape, -float('inf'))
    bg = None
    for start in range(0, logits.shape[0], chunk_size):
        part = F.interpolate(logits[None, start:start+chunk_size], size=(g.canvas, g.canvas),
                             mode='bilinear', align_corners=False)
        # Remove only letterbox padding, never original-image content.
        part = part[:, :, g.top:g.top+g.resized_height, g.left:g.left+g.resized_width]
        part = F.interpolate(part, size=shape, mode='bilinear', align_corners=False)[0]
        value, index = part.max(dim=0)
        take = value > best
        labels[take] = (index + start)[take]
        best = torch.maximum(best, value)
        lse = torch.logaddexp(lse, torch.logsumexp(part, dim=0))
        if start == 0:
            bg = part[0].clone()
    prob = (-torch.expm1(bg - lse)).clamp(0, 1)
    confidence = torch.exp(best - lse).clamp(0, 1)
    return labels.numpy().astype(np.uint8), prob.numpy(), confidence.numpy()
