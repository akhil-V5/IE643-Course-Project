#!/usr/bin/env python
"""Download the official ResShift journal source and its matching v1 / VQ weights.

Uses Git and HTTPS. Does not install dependencies or execute downloaded code.
Logs upstream commit and SHA256 file hashes. Hashes are provenance records, NOT
an independently authenticated publisher checksum. Obtain dependencies only
from trusted locations; upstream licensing is separate from this wrapper.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FILES = ["configs/realsr_swinunet_realesrgan256.yaml", "models/gaussian_diffusion.py",
         "models/script_util.py", "models/respace.py", "models/unet.py", "models/swin_transformer.py",
         "ldm/models/autoencoder.py", "ldm/modules/vqvae/quantize.py"]
URLS = {
    "resshift_realsrx4_s15_v1.pth": "https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v1.pth",
    "autoencoder_vq_f4.pth": "https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth",
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ref", default="journal", help="Branch/tag or commit; recorded in lock file")
    p.add_argument("--skip-weights", action="store_true")
    a = p.parse_args()
    repo = ROOT / "external/ResShift"
    lock = ROOT / "external/resshift_lock.json"
    repo.parent.mkdir(exist_ok=True)
    if not repo.exists():
        subprocess.run(["git", "clone", "--branch", "journal", "https://github.com/zsyOAOA/ResShift.git", str(repo)], check=True)
        if a.ref != "journal":
            subprocess.run(["git", "-C", str(repo), "checkout", a.ref], check=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise SystemExit("Upstream working tree is modified. Use an unmodified checkout for this package.")
    old = None
    record = {"url": "https://github.com/zsyOAOA/ResShift", "commit": commit,
              "source_sha256": {f: digest(repo / f) for f in FILES}, "weights": {}}
    if lock.exists():
        old = json.loads(lock.read_text())
        if old["commit"] != commit or old["source_sha256"] != record["source_sha256"]:
            raise SystemExit("Existing source lock differs. Do not silently change the backend of an experiment; use a new project copy.")
    weights = ROOT / "weights"
    weights.mkdir(exist_ok=True)
    if not a.skip_weights:
        for name, url in URLS.items():
            dest = weights / name
            if not dest.exists():
                print(f"Downloading official weight file: {name}", flush=True)
                tmp = dest.with_suffix(dest.suffix + ".part")
                req = urllib.request.Request(url, headers={"User-Agent": "PlantSeg-ResShift-research/0.1"})
                with urllib.request.urlopen(req, timeout=90) as src, tmp.open("wb") as dst:
                    while chunk := src.read(1024 * 1024):
                        dst.write(chunk)
                if tmp.stat().st_size < 1024 * 1024:
                    raise RuntimeError(f"Unexpectedly small download: {tmp}")
                tmp.replace(dest)
            current_hash = digest(dest)
            previous = (old or {}).get("weights", {}).get(name)
            if previous and previous["sha256"] != current_hash:
                raise SystemExit(f"Existing recorded weight changed: {dest}. Do not overwrite its provenance lock.")
            record["weights"][name] = {"url": url, "sha256": current_hash, "bytes": dest.stat().st_size}
    elif old:
        record["weights"] = old.get("weights", {})
        for name, info in record["weights"].items():
            if not (weights / name).is_file() or digest(weights / name) != info["sha256"]:
                raise SystemExit(f"Existing recorded weight missing or changed: {name}")
    lock.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Backend commit: {commit}\nProvenance: {lock}\nRead external/ResShift/LICENSE before use or redistribution.")


if __name__ == "__main__":
    main()
