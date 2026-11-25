from pathlib import Path
import sys
import csv
import yaml
import numpy as np
from PIL import Image, ImageOps

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "segmentation"))

from common.prepare import rows_from_config
from common.runtime import load_yaml

config = load_yaml("configs/data.yaml")
rows = rows_from_config(config)

with open("dimension_mismatches.csv") as f:
    mismatches = {
        r["image"]: r
        for r in csv.DictReader(f)
    }

fixed_dir = ROOT / "data/exif_fixed_images"
preview_dir = ROOT / "data/exif_fix_previews"

fixed_dir.mkdir(parents=True, exist_ok=True)
preview_dir.mkdir(parents=True, exist_ok=True)

updated = []
fixed_count = 0

for row in rows:
    row = dict(row)

    if row["image"] in mismatches:
        with Image.open(row["image"]) as img:
            with Image.open(row["mask"]) as mask:
                assert img.getexif().get(274, 1) == 6

                corrected = ImageOps.exif_transpose(img).convert("RGB")

                assert corrected.size == mask.size, row["image"]

                name = Path(row["image"]).stem + ".png"

                destination = fixed_dir / row["split"] / name
                destination.parent.mkdir(parents=True, exist_ok=True)

                # Lossless PNG avoids an additional JPEG compression.
                corrected.save(destination)

                # Create a preview to verify image-mask alignment.
                rgb = np.array(corrected)
                labels = np.array(mask)
                disease = (labels >= 1) & (labels <= 115)

                overlay = rgb.copy()
                overlay[disease] = (
                    0.55 * rgb[disease] +
                    0.45 * np.array([255, 0, 0])
                ).astype(np.uint8)

                Image.fromarray(overlay).save(
                    preview_dir / (row["split"] + "_" + name)
                )

                row["image"] = str(destination.resolve())
                fixed_count += 1

    updated.append(row)

assert fixed_count == len(mismatches) == 8

manifest = ROOT / "data/plantseg_exif_manifest.csv"

with open(manifest, "w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=["image", "mask", "split"]
    )
    writer.writeheader()
    for row in updated:
        writer.writerow({
            "image": row["image"],
            "mask": row["mask"],
            "split": row["split"]
        })

config["source"] = {
    "mode": "csv",
    "manifest": str(manifest.resolve())
}

new_config = ROOT / "configs/data_exif.yaml"

with open(new_config, "w") as f:
    yaml.safe_dump(config, f, sort_keys=False)

print("Total samples:", len(updated))
print("Corrected images:", fixed_count)
print("Manifest:", manifest)
print("Configuration:", new_config)
print("Overlay previews:", preview_dir)