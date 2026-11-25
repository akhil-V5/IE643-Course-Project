from __future__ import annotations
import math
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from skimage.metrics import structural_similarity
from .common import to_pil, panel, annotation_overlay


def image_metrics(pred: torch.Tensor, target: torch.Tensor, lesion: torch.Tensor, valid: torch.Tensor) -> dict:
    # One image at a time; float RGB [0,1]. Lesion errors use full-resolution masks.
    a = ((pred.detach().cpu().float().clamp(-1, 1) + 1) / 2).numpy().transpose(1, 2, 0)
    b = ((target.detach().cpu().float().clamp(-1, 1) + 1) / 2).numpy().transpose(1, 2, 0)
    v = valid.cpu().numpy()[0].astype(bool)
    m = lesion.cpu().numpy()[0].astype(bool) & v
    err = np.abs(a - b).mean(axis=-1)
    sqerr = ((a - b) ** 2).mean(axis=-1)
    mse = float(sqerr[v].mean())
    # 120 dB caps exact matches for JSON compatibility; marked in metric documentation.
    psnr = -10 * math.log10(max(mse, 1e-12))
    return {"psnr_rgb": psnr,
            "ssim_rgb": float(structural_similarity(a, b, data_range=1.0, channel_axis=-1)),
            "mae": float(err[v].mean()),
            "lesion_mae": float(err[m].mean()) if m.any() else None,
            "nonlesion_mae": float(err[v & ~m].mean()) if (v & ~m).any() else None,
            "lesion_pixels": int(m.sum()), "valid_pixels": int(v.sum())}


def average_metrics(rows: list[dict]) -> dict:
    """Macro average over evaluated patches (not disease segmentation accuracy)."""
    keys = ["psnr_rgb", "ssim_rgb", "mae", "lesion_mae", "nonlesion_mae"]
    out = {}
    for key in keys:
        values = [r[key] for r in rows if r.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    out["patches"] = len(rows)
    out["patches_with_lesions"] = sum(r.get("lesion_mae") is not None for r in rows)
    return out


@torch.no_grad()
def evaluate_pairs(backend, dataset, seed: int = 42, amp: bool = True,
                   preview_dir: str | Path | None = None, preview_count: int = 6,
                   include_codec: bool = False):
    device = backend.device
    per_patch = []
    totals = {"diffusion": [], "bicubic": []}
    if include_codec:
        totals["codec_roundtrip"] = []
    # Preserve training RNG state so validation frequency does not alter train noise.
    devices = [device.index or torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        for i in range(len(dataset)):
            b = dataset[i]
            torch.manual_seed(seed + i)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + i)
            hr, lr = b["hr"].to(device)[None], b["lr"].to(device)[None]
            pred = backend.sample(lr, amp=amp)[0].cpu()
            methods = {"diffusion": pred, "bicubic": b["bicubic"]}
            if include_codec:
                methods["codec_roundtrip"] = backend.codec_roundtrip(hr)[0].cpu()
            for name, output in methods.items():
                metrics = image_metrics(output, b["hr"], b["lesion"], b["valid"])
                totals[name].append(metrics)
                per_patch.append({"id": b["id"], "source_id": b["source_id"], "method": name, **metrics})
            if preview_dir:
                out=Path(preview_dir)
                for name,im in {"original_rgb":to_pil(b["hr"]),"lr":to_pil(b["lr"]),"bicubic":to_pil(b["bicubic"]),"reconstructed_rgb":to_pil(pred),
                                "gt_disease_binary":Image.fromarray((b["lesion"][0].numpy()*255).astype(np.uint8)),
                                "gt_valid":Image.fromarray((b["valid"][0].numpy()*255).astype(np.uint8))}.items():
                    dest=out/name/(b["id"]+".png");dest.parent.mkdir(parents=True,exist_ok=True);im.save(dest)
            if preview_dir and (preview_count==0 or i < preview_count):
                mask = b["lesion"][0].numpy().astype(bool)
                ims = [to_pil(b["bicubic"]), to_pil(pred), to_pil(b["hr"]),
                       annotation_overlay(to_pil(pred), mask),
                       Image.fromarray((mask*255).astype(np.uint8))]
                labels = ["Bicubic input", "Diffusion reconstruction", "Original HR target", "GT overlay (not prediction)", "GT disease mask (not prediction)"]
                if include_codec:
                    ims.append(to_pil(methods["codec_roundtrip"])); labels.append("Frozen codec roundtrip")
                panel(ims, labels, Path(preview_dir) / f"{b['id']}.png")
                if i == 0:
                    # Convenient files for an inference smoke test; LR itself is not enlarged.
                    to_pil(b["lr"]).save(Path(preview_dir) / "example_lr.png")
                    to_pil(b["hr"]).save(Path(preview_dir) / "example_hr.png")
    return {k: average_metrics(v) for k, v in totals.items()}, per_patch
