from __future__ import annotations
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any
import numpy as np
import torch
from PIL import Image, ImageDraw


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict], fields: list[str] | None = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def to_tensor(image: Image.Image) -> torch.Tensor:
    a = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(a.transpose(2, 0, 1)).contiguous() * 2.0 - 1.0


def to_pil(tensor: torch.Tensor) -> Image.Image:
    x = tensor.detach().cpu().float()
    if x.ndim == 4:
        if len(x) != 1:
            raise ValueError("to_pil expects one image")
        x = x[0]
    a = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(a)


def panel(images: list[Image.Image], labels: list[str], path: str | Path) -> None:
    if len(images) != len(labels) or not images:
        raise ValueError("Provide one label per image")
    w = max(i.width for i in images)
    h = max(i.height for i in images)
    out = Image.new("RGB", (w * len(images), h + 32), "white")
    draw = ImageDraw.Draw(out)
    for n, (im, label) in enumerate(zip(images, labels)):
        out.paste(im.convert("RGB"), (n * w, 32))
        draw.text((n * w + 5, 9), label, fill="black")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.save(path)


def annotation_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Ground-truth overlay only, never a predicted lesion map."""
    a = np.array(image.convert("RGB"), dtype=np.float32)
    selected = mask.astype(bool)
    a[selected] = a[selected] * 0.55 + np.array([255, 50, 50]) * 0.45
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def safe_load(path: str | Path) -> dict:
    """Tensor-only loading. Never silently fall back to arbitrary-pickle loading."""
    return torch.load(str(path), map_location="cpu", weights_only=True)


def atomic_save(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(p)
