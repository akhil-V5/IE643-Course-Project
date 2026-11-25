"""ResShift v1: sequential 64-LR-pixel tiles, CPU overlap blending.

Restores the earlier suite's 64/16 geometry and per-tile seeds. No extra
image downsampling. Training and the low-level backend are not modified.
"""
from __future__ import annotations
import time
import torch
import torch.nn.functional as F
from PIL import Image
from leafdiff.common import to_tensor, to_pil


def positions(length: int, tile: int, stride: int) -> list[int]:
    if length < tile or not 0 < stride <= tile:
        raise ValueError("Invalid padded length or tile stride")
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


@torch.no_grad()
def tiled_reconstruct(backend, image: Image.Image, seed: int = 42,
                      overlap: int = 16, amp: bool = True):
    """Return (x4 PIL RGB, truthful sampling metadata). One tile on GPU at a time."""
    tile, scale = 64, 4
    if not isinstance(overlap, int) or not 0 <= overlap < tile:
        raise ValueError("overlap must be an integer from 0 to 63 LR pixels")
    image = image.convert("RGB")
    w, h = image.size
    if min(w, h) < 1:
        raise ValueError("Empty input image")
    # Keep the complete input, output and blending arrays in system RAM.
    x = to_tensor(image).cpu()
    right, bottom = max(0, tile-w), max(0, tile-h)
    if right or bottom:
        x = F.pad(x[None], (0, right, 0, bottom), mode="replicate")[0]
    hp, wp = x.shape[-2:]
    accum = torch.zeros((3, hp*scale, wp*scale), dtype=torch.float32, device="cpu")
    norm = torch.zeros((1, hp*scale, wp*scale), dtype=torch.float32, device="cpu")
    win = torch.hann_window(tile*scale, periodic=False, dtype=torch.float32,
                            device="cpu").clamp_min(0.001)
    weight = (win[:, None]*win[None, :])[None]
    cuda = torch.device(backend.device).type == "cuda"
    if cuda:
        torch.cuda.synchronize(backend.device)
    started = time.perf_counter()
    count = 0
    for y in positions(hp, tile, tile-overlap):
        for xx in positions(wp, tile, tile-overlap):
            torch.manual_seed(seed+count)
            if cuda:
                torch.cuda.manual_seed_all(seed+count)
            patch = x[:, y:y+tile, xx:xx+tile][None].to(backend.device)
            sr_gpu = backend.sample(patch, amp=amp)
            if tuple(sr_gpu.shape) != (1, 3, tile*scale, tile*scale):
                raise RuntimeError(f"Unexpected ResShift tile shape: {tuple(sr_gpu.shape)}")
            if not torch.isfinite(sr_gpu).all():
                raise RuntimeError("Non-finite ResShift reconstruction tile")
            restored = sr_gpu[0].detach().to(device="cpu", dtype=torch.float32)
            del sr_gpu, patch
            yy, xxx, ss = y*scale, xx*scale, tile*scale
            accum[:, yy:yy+ss, xxx:xxx+ss] += restored*weight
            norm[:, yy:yy+ss, xxx:xxx+ss] += weight
            del restored
            count += 1
    if cuda:
        torch.cuda.synchronize(backend.device)
    result = (accum/norm.clamp_min(1e-12))[:, :h*scale, :w*scale]
    if not torch.isfinite(result).all():
        raise RuntimeError("Non-finite blended reconstruction")
    metadata = {
        "sampling_mode": "tiled", "sample_calls": count, "tiles": count,
        "tile_lr": tile, "overlap_lr": overlap, "scale": scale,
        "input_size_wh": [w, h], "output_size_wh": [w*scale, h*scale],
        "padding_right_bottom": [right, bottom],
        "padding_mode": "replicate" if right or bottom else "none",
        "seed": seed, "sampling_seconds": time.perf_counter()-started,
        "blend_device": "cpu", "tile_batch_size": 1,
    }
    return to_pil(result), metadata
