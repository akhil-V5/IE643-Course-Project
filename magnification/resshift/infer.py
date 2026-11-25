#!/usr/bin/env python
"""Reconstruct x4 RGB leaf crops. No ground-truth mask or disease label is required.

Large inputs are tiled at native input resolution. Independent tile sampling can
cause seams; blending mitigates, but does not guarantee against them. This is
frame-based SR, not temporal/video diffusion and not lesion segmentation.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from leafdiff.backend import ResShiftBackend
from leafdiff.common import safe_load, seed_all, to_tensor, to_pil, panel, write_json
from leafdiff.data import IMAGE_EXTS


# def positions(length: int, tile: int, stride: int) -> list[int]:
#     xs = list(range(0, length - tile + 1, stride))
#     if xs[-1] != length - tile:
#         xs.append(length - tile)
#     return xs


# @torch.no_grad()
# def tiled_reconstruct(backend, image: Image.Image, seed: int = 42, overlap: int = 16, amp: bool = True):
#     tile, scale = 64, 4
#     if not 0 <= overlap < tile:
#         raise ValueError("overlap must be 0..63 in LR pixels")
#     x = to_tensor(image)
#     _, h, w = x.shape
#     # Replicate padding is defined even for very small images; never invent a HR target.
#     if h < tile or w < tile:
#         x = torch.nn.functional.pad(x[None], (0, max(0, tile-w), 0, max(0, tile-h)), mode="replicate")[0]
#     hp, wp = x.shape[-2:]
#     accum = torch.zeros(3, hp*scale, wp*scale)
#     norm = torch.zeros(1, hp*scale, wp*scale)
#     win = torch.hann_window(tile*scale, periodic=False).clamp_min(0.001)
#     weight = (win[:, None] * win[None, :])[None]
#     count = 0
#     for y in positions(hp, tile, tile-overlap):
#         for xx in positions(wp, tile, tile-overlap):
#             torch.manual_seed(seed + count)
#             if backend.device.type == "cuda":
#                 torch.cuda.manual_seed_all(seed + count)
#             patch = x[:, y:y+tile, xx:xx+tile][None].to(backend.device)
#             restored = backend.sample(patch, amp=amp)[0].cpu().float()
#             yy, xxx, ss = y*scale, xx*scale, tile*scale
#             accum[:, yy:yy+ss, xxx:xxx+ss] += restored * weight
#             norm[:, yy:yy+ss, xxx:xxx+ss] += weight
#             count += 1
#     result = (accum / norm.clamp_min(1e-12))[:, :h*scale, :w*scale]
#     return to_pil(result), count
@torch.no_grad()
def whole_image_reconstruct(
    backend,
    image: Image.Image,
    seed: int = 42,
    amp: bool = True,
):
    """Input is already LR. Do not downsample it again here."""
    image = image.convert("RGB")

    torch.manual_seed(seed)
    if backend.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    lr = to_tensor(image)[None].to(backend.device)

    # ONE call for the entire low-resolution image.
    restored = backend.sample(lr, amp=amp)

    expected = (1, 3, image.height * 4, image.width * 4)
    if tuple(restored.shape) != expected:
        raise RuntimeError(
            f"Output shape {tuple(restored.shape)} != {expected}"
        )

    if not torch.isfinite(restored).all():
        raise RuntimeError("ResShift produced non-finite RGB values")

    sampling = {
        "sampling_mode": "tiled",
        "sample_calls": 1,
    }

    return to_pil(restored[0].cpu()), sampling

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/plantseg.yaml")
    p.add_argument("--checkpoint", help="Omit for pretrained baseline; otherwise best.pth/last.pth")
    p.add_argument("--input", required=True, help="Single leaf image or image directory")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--roi", help="Optional input crop: x,y,width,height. Not an automatic disease detector.")
    a = p.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(a.config), resolve=True)
    if a.checkpoint:
        saved = safe_load(a.checkpoint)
        if "config" in saved:
            cfg = saved["config"]
        del saved
    inp, out = Path(a.input).resolve(), Path(a.output).resolve()
    files = [inp] if inp.is_file() else sorted(p for p in inp.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise ValueError("No input images found")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new output folder")
    out.mkdir(parents=True, exist_ok=True)
    backend = ResShiftBackend(cfg, checkpoint=a.checkpoint)
    seed_all(a.seed)
    records = []
    for path in files:
        with Image.open(path) as im:
            image = im.convert("RGB").copy()
        original_size = image.size
        if a.roi:
            x, y, w, h = map(int, a.roi.split(","))
            if min(x, y) < 0 or min(w, h) <= 0 or x+w > image.width or y+h > image.height:
                raise ValueError("ROI is outside input bounds")
            image = image.crop((x, y, x+w, y+h))
        label = path.stem + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:8]
        from leafdiff.tiling import tiled_reconstruct as restore_tiles
        sr, sampling = restore_tiles(backend, image, seed=a.seed, overlap=16, amp=not a.no_amp)
        bic = image.resize(sr.size, Image.Resampling.BICUBIC)
        sr.save(out / f"{label}_sr.png")
        image.save(out / f"{label}_input.png")
        bic.save(out / f"{label}_bicubic.png")
        # Panels are display-size previews; full resolution is saved in separate files.
        v1, v2 = bic.copy(), sr.copy()
        v1.thumbnail((600, 600)); v2.thumbnail((600, 600))
        panel([v1, v2], ["Bicubic enlargement", "Diffusion reconstruction"], out / f"{label}_comparison.png")
        records.append({"input": str(path), "original_size": original_size, "crop_size": image.size,
                        "output_size": sr.size, **sampling, "seed": a.seed, "roi": a.roi,
                        "output": str(out / f"{label}_sr.png")})
        print(
            f"{path.name}: {image.size} -> {sr.size}; "
            f"sampling_mode=tiled; sample_calls={sampling['sample_calls']}",
            flush=True,
        )
    write_json(out / "inference.json", {"checkpoint": a.checkpoint or cfg["backend"]["diffusion_weights"],
            "sampling_mode": "tiled", "results": records,
            "note": "RGB restoration only. No disease mask, diagnostic confidence, or severity is predicted."})


if __name__ == "__main__":
    main()
