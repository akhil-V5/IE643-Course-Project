"""Validate the actual split manifest and decoded source content, not stale metadata."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from .classes import CLASS_HASH
from .data import load_rgb, load_labels
from .io import file_hash, read_csv, read_json


def decoded_hashes(image, labels):
    rgb = np.asarray(image.convert('RGB'), dtype=np.uint8)
    ih = hashlib.sha256(f'{image.width},{image.height}:'.encode() + rgb.tobytes()).hexdigest()
    mh = hashlib.sha256(np.asarray(labels, dtype=np.uint8).tobytes()).hexdigest()
    return ih, mh


def verify_prepared(prepared, expected_manifest=None, verify_content=True):
    root = Path(prepared).expanduser().resolve()
    meta = read_json(root / 'metadata.json')
    actual = file_hash(root / 'manifest.csv')
    if meta.get('class_hash') != CLASS_HASH:
        raise ValueError('Class mapping mismatch; do not reinterpret mask IDs.')
    if actual != meta.get('manifest_sha256'):
        raise ValueError('Actual manifest hash differs from audited metadata. Re-audit the split.')
    if expected_manifest is not None and actual != expected_manifest:
        raise ValueError('Evaluation/training split differs from the checkpoint manifest.')
    rows = read_csv(root / 'manifest.csv')
    if not rows:
        raise ValueError('Empty manifest')
    seen_ids, seen_hashes, seen_paths = set(), {}, set()
    for row in rows:
        if row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Invalid split name')
        if row['id'] in seen_ids:
            raise ValueError('Repeated sample ID: ' + row['id'])
        seen_ids.add(row['id'])
        image_path = str(Path(row['image']).resolve())
        if image_path in seen_paths:
            raise ValueError('Repeated source path in prepared manifest: ' + image_path)
        seen_paths.add(image_path)
        if verify_content:
            image, labels = load_rgb(row['image']), load_labels(row['mask'])
            if labels.shape != (image.height, image.width):
                raise ValueError('Image/mask dimensions changed: ' + row['image'])
            ih, mh = decoded_hashes(image, labels)
            if ih != row.get('image_sha256_decoded') or mh != row.get('mask_sha256_decoded'):
                raise ValueError('Source content hash changed: ' + row['image'])
        else:
            ih = row['image_sha256_decoded']
        if ih in seen_hashes and seen_hashes[ih] != row['split']:
            if not meta.get('allow_cross_split_duplicates', False):
                raise ValueError(
                    'Cross-split duplicate source image: ' + row['image']
                )
        seen_hashes[ih] = row['split']
    return {**meta, 'verified_manifest_sha256': actual,
            'source_content_verified': bool(verify_content)}, rows
