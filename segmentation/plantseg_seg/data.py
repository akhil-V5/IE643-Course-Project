from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from .classes import NUM_CLASSES, IGNORE_INDEX
from .io import read_csv, read_json
from .geometry import letterbox


def load_rgb(path):
    with Image.open(path) as image:
        # Match the existing diffusion pipeline: no independent EXIF transpose.
        return image.convert('RGB').copy()


def load_labels(path):
    with Image.open(path) as image:
        labels = np.array(image)  # preserve palette IDs; do not .convert('L')
    if labels.ndim == 3:
        if labels.shape[2] == 3 and np.array_equal(labels[...,0], labels[...,1]) and np.array_equal(labels[...,0], labels[...,2]):
            labels = labels[...,0]
        else:
            raise ValueError(f'RGB colour annotation needs explicit colour-to-ID conversion: {path}')
    if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f'Expected integer class-index PNG: {path}')
    bad = np.unique(labels[(labels < 0) | ((labels >= NUM_CLASSES) & (labels != IGNORE_INDEX))])
    if bad.size:
        raise ValueError(f'Unexpected label IDs {bad.tolist()} in {path}; expected 0..115 or 255=ignore')
    if not np.any(labels != IGNORE_INDEX):
        raise ValueError(f'All pixels are ignored: {path}')
    return labels.astype(np.uint8)


class PlantSegDataset(Dataset):
    def __init__(self, prepared, split, image_size=512, training=False,
                 augment=True, seed=42, limit=0, degrade_scales=(1,)):
        self.prepared = Path(prepared).resolve()
        self.rows = [r for r in read_csv(self.prepared/'manifest.csv') if r['split'] == split]
        self.meta = read_json(self.prepared/'metadata.json')
        self.image_size, self.training = int(image_size), bool(training)
        self.limit = int(limit)
        if self.limit < 0: raise ValueError("limit must be non-negative")
        self.augment, self.seed, self.epoch = augment, int(seed), 0
        self.degrade_scales = tuple(int(x) for x in degrade_scales)
        if not self.degrade_scales or min(self.degrade_scales) < 1:
            raise ValueError('degrade_scales must contain positive integers')
        if not training:
            self.rows.sort(key=lambda r: hashlib.sha256(f'{seed}:{r["id"]}'.encode()).hexdigest())
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError(f'No images found for split={split}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        rgb, labels = load_rgb(row['image']), load_labels(row['mask'])
        if labels.shape != (rgb.height, rgb.width):
            raise ValueError(f'Image-mask alignment changed since audit: {row["image"]}')
        key = hashlib.sha256(f'{self.seed}:{self.epoch}:{row["id"]}'.encode()).hexdigest()
        rng = np.random.default_rng(int(key[:16], 16))
        if self.training and self.augment:
            if rng.random() < 0.5:
                rgb = rgb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                labels = np.ascontiguousarray(labels[:, ::-1])
            if rng.random() < 0.5:
                rgb = rgb.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                labels = np.ascontiguousarray(labels[::-1])
        if self.training:
            factor = self.degrade_scales[int(rng.integers(len(self.degrade_scales)))]
            if factor != 1:
                # Optional resolution augmentation; keeps the complete FOV and aligned labels.
                size = rgb.size
                rgb = rgb.resize((max(1, size[0]//factor), max(1, size[1]//factor)), Image.Resampling.BICUBIC)
                rgb = rgb.resize(size, Image.Resampling.BICUBIC)
        pixels, target, geom = letterbox(rgb, labels, self.image_size)
        if not (target != IGNORE_INDEX).any():
            raise ValueError(f'No valid pixels survive resizing: {row["image"]}')
        return {'pixel_values': pixels, 'labels': target, 'id': row['id'],
                'image': row['image'], 'mask': row['mask'], 'index': index}
