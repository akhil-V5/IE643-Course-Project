#!/usr/bin/env python3
"""One image: shared LR -> bicubic / tiled ResShift -> comparison panel.

Add this file beside run.py in PlantSeg_Experiment_Suite_v3. This is an image-
only visualization entry point, not a video decoder or a replacement trainer.
It calls the suite's existing reconstruction and optional SegFormer commands.

reference: original -> edge-pad to multiple of 4 -> bicubic downsample x4;
           reconstruct both from that same LR; unpad before segmentation.
native:    input is already an observed LR image/crop; reconstruct directly x4.

Optional --checkpoint adds disease masks/overlays from the SAME SegFormer.
Optional --leaf-mask supplies a binary leaf mask on the input-image grid;
only with that mask is a predicted leaf-area percentage calculated.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

SCALE = 4


def read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Supply ONE still image, not a video or animated image.")
        return ImageOps.exif_transpose(image).convert("RGB").copy()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def prepare_input(image: Image.Image, mode: str):
    """Match experiments/cases.py geometry. Never resize the reference image."""
    width, height = image.size
    if mode == "reference":
        right, bottom = (-width) % SCALE, (-height) % SCALE
        padded = Image.fromarray(np.pad(np.asarray(image),
            ((0, bottom), (0, right), (0, 0)), mode="edge"))
        lr = padded.resize((padded.width // SCALE, padded.height // SCALE),
                           Image.Resampling.BICUBIC)
        target = (width, height)
    elif mode == "native":
        lr, right, bottom = image.copy(), 0, 0
        target = (width * SCALE, height * SCALE)
    else:
        raise ValueError("input mode must be reference or native")
    bicubic = lr.resize((lr.width * SCALE, lr.height * SCALE), Image.Resampling.BICUBIC)
    bicubic = bicubic.crop((0, 0, *target))
    return lr, bicubic, {
        "input_mode": mode, "input_size_wh": list(image.size),
        "lr_size_wh": list(lr.size), "output_size_wh": list(target),
        "reference_padding_right_bottom": [right, bottom],
        "scale": SCALE,
    }


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def fit_text(draw, text: str, width: int, size: int):
    selected = font(size)
    while draw.textbbox((0, 0), text, font=selected)[2] > width and size > 11:
        size -= 1
        selected = font(size)
    return selected


def panel_row(items: Sequence[tuple[str, Image.Image, bool]], width: int,
              cell_height: int) -> Image.Image:
    """Uniform display cells; no aspect-ratio distortion. bool = nearest display."""
    if not items:
        raise ValueError("No images to display")
    row = Image.new("RGB", (width, cell_height + 70), "white")
    draw = ImageDraw.Draw(row)
    for index, (label, image, nearest) in enumerate(items):
        x1, x2 = index * width // len(items), (index + 1) * width // len(items)
        label_font = fit_text(draw, label, x2-x1-24, 22)
        draw.text((x1+12, 8), label, fill="black", font=label_font)
        draw.text((x1+12, 35), f"{image.width} x {image.height} pixels",
                  fill="#444444", font=font(16))
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
        view = ImageOps.contain(image.convert("RGB"), (x2-x1-24, cell_height),
                                method=resample)
        px, py = x1+(x2-x1-view.width)//2, 64+(cell_height-view.height)//2
        row.paste(view, (px, py))
        draw.rectangle((px, py, px+view.width-1, py+view.height-1),
                       outline="#999999", width=1)
    return row


def make_panel(sections: Sequence[Sequence[tuple[str, Image.Image, bool]]],
               output: Path, title: str, subtitle: str, cell_size: int) -> None:
    width = 4 * cell_size
    rendered = [panel_row(section, width, cell_size) for section in sections]
    height = 94 + sum(row.height for row in rendered) + 48
    page = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(page)
    draw.text((16, 12), title, fill="black",
              font=fit_text(draw, title, width-32, 30))
    draw.text((16, 52), subtitle, fill="#444444",
              font=fit_text(draw, subtitle, width-32, 18))
    y = 94
    for row in rendered:
        page.paste(row, (0, y))
        y += row.height
    note = "Display fitted only; full-resolution images saved separately. White mask pixels = predicted disease."
    draw.text((16, height-31), note, fill="#444444",
              font=fit_text(draw, note, width-32, 17))
    page.save(output, dpi=(300, 300))


def run_command(command: list[str], root: Path, log: Path) -> None:
    print("RUN:", shlex.join(command), flush=True)
    with log.open("w", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                handle.write(line)
                handle.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise RuntimeError(f"Command failed (exit {code}); see {log}")


def verify_sr_output(folder: Path, lr_path: Path) -> tuple[Path, dict]:
    records = rows(folder / "images.csv")
    if len(records) != 1:
        raise ValueError("Expected exactly ONE reconstructed image")
    record = records[0]
    if Path(record["input"]).resolve() != lr_path.resolve():
        raise ValueError("Reconstruction belongs to a different input")
    sampling_mode = record.get("sampling_mode")
    calls = int(record.get("sample_calls", 0))
    if sampling_mode not in {"tiled", "whole_image"} or calls < 1:
        raise ValueError("Missing or invalid ResShift sampling metadata")
    if sampling_mode == "whole_image" and calls != 1:
        raise ValueError("Whole-image output must have exactly one sampler call")
    if sampling_mode == "tiled" and int(record.get("tiles", 0)) != calls:
        raise ValueError("Tiled output has inconsistent tile/call counts")
    path = Path(record["rgb"])
    return (path if path.is_absolute() else folder / path), record


def predicted_overlay(image: Image.Image, mask: np.ndarray) -> Image.Image:
    arr = np.asarray(image).astype(np.float32).copy()
    if mask.shape != arr.shape[:2]:
        raise ValueError("Image and predicted mask must share one coordinate grid")
    # Visualization only. Never feed this overlay into either model.
    arr[mask] = 0.55 * arr[mask] + 0.45 * np.array([255, 50, 50])
    return Image.fromarray(arr.round().clip(0, 255).astype(np.uint8))


def load_leaf_mask(path: Path, original_size: tuple[int, int], mode: str) -> np.ndarray:
    with Image.open(path) as image:
        # Do not auto-rotate labels: they must already align to the normalized RGB.
        labels = np.array(image)
    if labels.ndim != 2 or labels.shape != (original_size[1], original_size[0]):
        raise ValueError("--leaf-mask must be a 2D mask aligned to the input RGB image")
    if not set(np.unique(labels).tolist()).issubset({0, 1, 255}):
        raise ValueError("--leaf-mask must be binary (0 background, 1 or 255 leaf), not a lesion annotation")
    mask = labels > 0
    if not mask.any():
        raise ValueError("The supplied leaf mask is empty")
    return mask if mode == "reference" else mask.repeat(SCALE, 0).repeat(SCALE, 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="One image path; no directory/video")
    p.add_argument("--input-mode", choices=["reference", "native"], default="reference")
    p.add_argument("--output", required=True, help="New output directory")
    p.add_argument("--project-root", default=str(Path(__file__).resolve().parent))
    p.add_argument("--sr-python", default=sys.executable)
    p.add_argument("--seg-python", default=sys.executable)
    p.add_argument("--sr-config", default="magnification/configs/resshift_pretrained.yaml")
    p.add_argument("--sr-checkpoint", help="Optional fine-tuned ResShift; omit for frozen pretrained")
    p.add_argument("--checkpoint", help="Optional trained SegFormer; adds masks and overlays")
    p.add_argument("--leaf-mask", help="Optional binary SAM/leaf mask on input grid; NOT a disease mask")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cell-size", type=int, default=480, help="Display cell size only; never model input size")
    return p


def main(argv: Sequence[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    image_path = source_path(args.image, root)
    out = source_path(args.output, root)
    if not (root / "run.py").is_file():
        raise FileNotFoundError("--project-root must contain your existing run.py")
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if args.cell_size < 240:
        raise ValueError("--cell-size must be at least 240 (display only)")
    if args.leaf_mask and not args.checkpoint:
        raise ValueError("--leaf-mask is used with --checkpoint for segmentation")
    for field in ("sr_config", "sr_checkpoint", "checkpoint", "leaf_mask"):
        value = getattr(args, field)
        if value:
            resolved = source_path(value, root)
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            setattr(args, field, str(resolved))
    for field in ("sr_python", "seg_python"):
        value = getattr(args, field)
        executable = shutil.which(value)
        if executable is None:
            raise FileNotFoundError(f"Python executable not found: {value}")
        setattr(args, field, str(Path(executable).absolute()))
    image = read_rgb(image_path)
    lr, bicubic, geometry = prepare_input(image, args.input_mode)
    leaf_mask = load_leaf_mask(Path(args.leaf_mask), image.size, args.input_mode) if args.leaf_mask else None
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Use a new/empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "arguments.json", vars(args))
    image.save(out / "00_input.png")
    lr_path = out / "01_lr.png"
    lr.save(lr_path)
    rgb_dir = out / "reconstructions"
    rgb_dir.mkdir()
    bicubic_path = rgb_dir / "bicubic.png"
    bicubic.save(bicubic_path)

    # run.py launches the models in separate environments, preserving compatibility.
    prefix = [sys.executable, str(root / "run.py"), "--sr-python", args.sr_python,
              "--seg-python", args.seg_python]
    mode = "resshift-finetuned" if args.sr_checkpoint else "resshift-pretrained"
    sr_dir = out / "_resshift"
    command = prefix + ["reconstruct", "--mode", mode, "--input", str(lr_path),
        "--config", args.sr_config, "--output", str(sr_dir), "--device", args.device,
        "--preview-count", "1"]
    if args.sr_checkpoint:
        command += ["--sr-checkpoint", args.sr_checkpoint]
    if args.no_amp:
        command.append("--no-amp")
    run_command(command, root, out / "resshift.log")
    raw_path, sr_record = verify_sr_output(sr_dir, lr_path)
    sr = read_rgb(raw_path)
    if sr.size != (SCALE * lr.width, SCALE * lr.height):
        raise ValueError("ResShift output is not exactly x4 the shared LR image")
    sr = sr.crop((0, 0, *geometry["output_size_wh"]))
    if sr.size != bicubic.size:
        raise ValueError("Bicubic and ResShift are not aligned")
    sr_path = rgb_dir / "resshift.png"
    sr.save(sr_path)

    condition = "fine-tuned" if args.sr_checkpoint else "frozen pretrained"
    sr_title = "ResShift (fine-tuned)" if args.sr_checkpoint else "ResShift (frozen)"
    reference_title = "Original reference" if args.input_mode == "reference" else "Original LR input"
    lr_title = "Shared LR input (x4 downsampled)" if args.input_mode == "reference" else "Shared LR input (unchanged)"
    reconstruction_row = [(reference_title, image, False), (lr_title, lr, True),
                          ("Bicubic reconstruction", bicubic, False), (sr_title, sr, False)]
    subtitle = (f"Same LR input for both methods | ResShift: {sr_record['sampling_mode']} | "
                + ("reconstruct to the original image grid" if args.input_mode == "reference"
                   else "native LR input magnified x4; no extra downsampling"))
    make_panel([reconstruction_row], out / "reconstruction_panel.png",
               "Single-image reconstruction comparison", subtitle, args.cell_size)
    sections = [reconstruction_row]
    summary = {**geometry, "input": str(image_path), "sr_condition": condition,
               "sr_checkpoint": args.sr_checkpoint, "segformer_checkpoint": args.checkpoint,
               "sr_record": sr_record, "methods": {},
               "note": "Qualitative single-image comparison; no GT accuracy metrics are calculated."}

    if args.checkpoint:
        seg_dir = out / "_segmentation"
        command = prefix + ["infer-seg", "--checkpoint", args.checkpoint,
            "--input", str(rgb_dir), "--output", str(seg_dir), "--device", args.device,
            "--preview-count", "2"]
        if args.no_amp:
            command.append("--no-amp")
        run_command(command, root, out / "segformer.log")
        records = rows(seg_dir / "images.csv")
        indexed = {str(Path(r["input"]).resolve()): r for r in records}
        expected = {str(bicubic_path.resolve()), str(sr_path.resolve())}
        if len(records) != 2 or set(indexed) != expected:
            raise ValueError("Expected SegFormer predictions for exactly these two reconstructions")
        pred_dir = out / "predictions"
        pred_dir.mkdir()
        if leaf_mask is not None:
            Image.fromarray(leaf_mask.astype(np.uint8)*255).save(out / "leaf_mask_on_output_grid.png")
        comparison_rows = []
        for name, title, rgb, path in [("bicubic", "Bicubic", bicubic, bicubic_path),
                                       ("resshift", sr_title, sr, sr_path)]:
            record = indexed[str(path.resolve())]
            mp = Path(record["disease_binary"])
            mp = mp if mp.is_absolute() else seg_dir / mp
            with Image.open(mp) as mask_image:
                gray = np.asarray(mask_image.convert("L"))
            if gray.shape != (rgb.height, rgb.width):
                raise ValueError("SegFormer mask dimensions do not match reconstruction")
            if not set(np.unique(gray).tolist()).issubset({0, 255}):
                raise ValueError("Expected a predicted binary mask with values 0/255")
            mask = gray == 255
            if leaf_mask is not None:
                mask &= leaf_mask
            binary = Image.fromarray(mask.astype(np.uint8)*255)
            overlay = predicted_overlay(rgb, mask)
            binary.save(pred_dir / f"{name}_mask.png")
            overlay.save(pred_dir / f"{name}_overlay.png")
            entry = {"disease_pixels": int(mask.sum())}
            overlay_label = f"{title}: predicted overlay"
            if leaf_mask is not None:
                percentage = 100.0 * float(mask.sum()) / float(leaf_mask.sum())
                entry.update(leaf_pixels=int(leaf_mask.sum()), predicted_symptomatic_area_percent=percentage)
                overlay_label = f"{title}: {percentage:.2f}% predicted leaf area"
            else:
                entry["predicted_symptomatic_area_percent"] = None
                entry["note"] = "No leaf mask supplied; disease pixels / image area is NOT leaf severity."
            summary["methods"][name] = entry
            comparison_rows.append([(f"{title}: RGB", rgb, False),
                                    (f"{title}: predicted disease mask", binary, True),
                                    (overlay_label, overlay, False)])
        make_panel(comparison_rows, out / "localization_panel.png",
                   "Bicubic vs ResShift: disease localization", "Same frozen SegFormer for both reconstructions", args.cell_size)
        sections.extend(comparison_rows)
    make_panel(sections, out / "panel.png", "Bicubic vs ResShift: single-image comparison",
               subtitle, args.cell_size)
    write_json(out / "summary.json", summary)
    print(f"\nFINAL PANEL: {out / 'panel.png'}", flush=True)
    return out / "panel.png"


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
