from __future__ import annotations
import hashlib
import io
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from .common import read_csv, read_json, to_tensor

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_mask(path: str | Path, mode: str = "plantseg") -> tuple[np.ndarray, np.ndarray]:
    """Preserve palette INDEX values; converting indexed PNGs to L would corrupt IDs."""
    with Image.open(path) as im:
        arr = np.array(im)
    if arr.ndim == 3:
        if arr.shape[2] >= 3 and np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2]):
            arr = arr[..., 0]
        else:
            raise ValueError(f"{path}: RGB colour mask needs an explicit colour-to-ID conversion. Do not threshold it.")
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D indexed mask")
    values = set(np.unique(arr).tolist())
    if mode == "plantseg":
        if not values <= (set(range(116)) | {255}):
            raise ValueError(f"{path}: unexpected PlantSeg class IDs: {sorted(values)}")
        # Official PlantSeg115Dataset: index 0 is background; 1..115 are disease classes.
        # Its underlying mmseg default ignore index is 255. Verify local annotation overlays.
        valid = arr != 255
        lesion = (arr >= 1) & (arr <= 115)
        # Shared preparation has already verified INDEXED multiclass masks.
        # {0,255} is valid non-disease+ignored annotation; never reinterpret255 as disease.
    elif mode == "binary255":
        if not values <= {0, 255}:
            raise ValueError(f"{path}: binary255 mode requires only 0 and 255")
        valid, lesion = np.ones(arr.shape, dtype=bool), arr == 255
    elif mode == "binary01":
        if not values <= {0, 1}:
            raise ValueError(f"{path}: binary01 mode requires only 0 and 1")
        valid, lesion = np.ones(arr.shape, dtype=bool), arr == 1
    else:
        raise ValueError(f"Unknown mask mode {mode}")
    if not valid.any():
        raise ValueError(f"{path}: no valid annotated pixels")
    return lesion.astype(np.uint8), valid.astype(np.uint8)


def degrade(image: Image.Image, scale: int, mode: str, rng: np.random.Generator) -> Image.Image:
    """PIL bicubic, not Matlab bicubic. This difference is recorded, not hidden."""
    size = (image.width // scale, image.height // scale)
    if size[0] < 1 or size[1] < 1:
        raise ValueError("Image too small")
    if mode == "bicubic":
        return image.resize(size, Image.Resampling.BICUBIC)
    if mode != "mild":
        raise ValueError("degradation must be bicubic or mild")
    # A project-defined simple degradation, NOT an exact Real-ESRGAN simulation.
    radius = float(rng.uniform(0.0, 1.2))
    blurred = image.filter(ImageFilter.GaussianBlur(radius)) if radius > 0.05 else image
    a = np.array(blurred.resize(size, Image.Resampling.BICUBIC), dtype=np.float32)
    a += rng.normal(0, float(rng.uniform(0, 3.0)), a.shape)
    out = Image.fromarray(a.clip(0, 255).round().astype(np.uint8))
    stream = io.BytesIO()
    out.save(stream, format="JPEG", quality=int(rng.integers(75, 101)))
    stream.seek(0)
    with Image.open(stream) as im:
        return im.convert("RGB").copy()


def grid_positions(length: int, patch: int) -> list[int]:
    if length < patch:
        raise ValueError("Do not upscale undersized images to fabricate high-resolution targets")
    positions = list(range(0, length - patch + 1, patch))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


class PlantPairs(Dataset):
    def __init__(self, prepared: str | Path, split: str, hr_size: int = 256, scale: int = 4,
                 degradation: str = "bicubic", training: bool = False, augment: bool = True,
                 eval_grid: bool = False, seed: int = 42, limit: int = 0):
        self.root = Path(prepared).resolve()
        self.meta = read_json(self.root / "metadata.json")
        self.rows = read_csv(self.root / "manifest.csv")
        self.rows = [r for r in self.rows if r["split"] == split]
        self.hr_size, self.scale = hr_size, scale
        self.degradation, self.training, self.augment = degradation, training, augment
        self.seed = seed
        if hr_size != 256 or scale != 4:
            raise ValueError("This initial package is intentionally fixed to HR=256 and x4 ResShift v1")
        self.samples = []
        for r in self.rows:
            w, h = int(r["width"]), int(r["height"])
            # Retain small originals. Edge-pad RGB; padding is excluded from the
            # lesion-weighted RGB loss. Do NOT upscale to fabricate detail.
            w,h=max(w,hr_size),max(h,hr_size)
            if eval_grid and not training:
                for y in grid_positions(h, hr_size):
                    for x in grid_positions(w, hr_size):
                        self.samples.append((r, x, y))
            else:
                self.samples.append((r, None, None))
        if not training:
            # Stable hash ordering avoids selecting only the first alphabetical disease.
            self.samples.sort(key=lambda s: hashlib.sha256(f"{seed}:{s[0]['id']}:{s[1]}:{s[2]}".encode()).hexdigest())
        if limit:
            self.samples = self.samples[:limit]
        if not self.samples:
            raise ValueError(f"No samples for split {split}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        row, x, y = self.samples[index]
        with Image.open(row["image"]) as im:
            rgb = im.convert("RGB").copy()  # no EXIF transpose unless masks are transposed too
        lesion, valid = load_mask(row["mask"], self.meta["mask_mode"])
        p = self.hr_size
        pad_h,pad_w=max(0,p-rgb.height),max(0,p-rgb.width)
        if pad_h or pad_w:
            rgb=Image.fromarray(np.pad(np.asarray(rgb),((0,pad_h),(0,pad_w),(0,0)),mode="edge"))
            lesion=np.pad(lesion,((0,pad_h),(0,pad_w)),constant_values=0)
            valid=np.pad(valid,((0,pad_h),(0,pad_w)),constant_values=0)
        if self.training:
            rng = np.random.default_rng(int(np.random.randint(0, 2**31)))
            x = int(rng.integers(0, rgb.width - p + 1))
            y = int(rng.integers(0, rgb.height - p + 1))
        else:
            x = (rgb.width - p) // 2 if x is None else x
            y = (rgb.height - p) // 2 if y is None else y
            key = f"{self.seed}:{row['id']}:{x}:{y}"
            rng = np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:8], 16))
        # An all-ignore random/center crop is not useful. Retry without using
        # disease labels; fallback anchors on a VALID annotated pixel only.
        if not valid[y:y+p,x:x+p].any():
            for _ in range(16):
                xx=int(rng.integers(0,rgb.width-p+1)); yy=int(rng.integers(0,rgb.height-p+1))
                if valid[yy:yy+p,xx:xx+p].any():x,y=xx,yy;break
            else:
                coords=np.argwhere(valid)
                if not len(coords):raise ValueError("Source has no valid annotated pixels")
                yy,xx=coords[len(coords)//2]
                x=int(np.clip(xx-p//2,0,rgb.width-p));y=int(np.clip(yy-p//2,0,rgb.height-p))
        box = (x, y, x + p, y + p)
        rgb = rgb.crop(box)
        lesion = lesion[y:y+p, x:x+p]
        valid = valid[y:y+p, x:x+p]
        if not valid.any():
            raise ValueError(f"All-ignore crop in {row['image']} at {box}")
        if self.training and self.augment:
            k = int(rng.integers(4))
            a = np.rot90(np.array(rgb), k)
            lesion, valid = np.rot90(lesion, k), np.rot90(valid, k)
            if rng.random() < 0.5:
                a, lesion, valid = a[:, ::-1], lesion[:, ::-1], valid[:, ::-1]
            rgb = Image.fromarray(np.ascontiguousarray(a))
        lr = degrade(rgb, self.scale, self.degradation, rng)
        bic = lr.resize(rgb.size, Image.Resampling.BICUBIC)
        return {
            "hr": to_tensor(rgb), "lr": to_tensor(lr), "bicubic": to_tensor(bic),
            "lesion": torch.from_numpy(np.ascontiguousarray(lesion[None])).float(),
            "valid": torch.from_numpy(np.ascontiguousarray(valid[None])).float(),
            "id": f"{row['id']}_x{x}_y{y}", "source_id": row["id"],
        }


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
