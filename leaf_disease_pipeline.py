#!/usr/bin/env python3
"""Image / offline video: DINO -> SAM2 -> x4 bicubic/ResShift -> SegFormer.

Add this file beside the existing PlantSeg suite's run.py. No old add-on is
required and no existing project file is changed. ResShift defaults to tiled
inference (64 LR pixels, overlap 16, CPU blending). Each stage is a separate
process; models are loaded once per stage and released before the next stage.
Video leaves are independently detected per analysed frame, NOT tracked.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

SCRIPT = Path(__file__).resolve()
SCALE = 4


@dataclass
class Options:
    project_root: str = str(SCRIPT.parent)
    checkpoint: str = "runs/segformer_b2_10/best.pth"
    method: str = "resshift"  # bicubic | resshift | both
    sr_config: str = "magnification/configs/resshift_pretrained.yaml"
    sr_checkpoint: str | None = None
    sr_mode: str = "tiled"
    overlap: int = 16
    det_python: str = sys.executable
    sr_python: str = sys.executable
    seg_python: str = sys.executable
    device: str = "cuda"
    dino_model: str = "IDEA-Research/grounding-dino-tiny"
    sam2_model: str = "facebook/sam2.1-hiera-small"
    sam2_config: str | None = None
    sam2_checkpoint: str | None = None
    prompt: str = "leaf."
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    box_nms_iou: float = 0.5
    mask_nms_iou: float = 0.85
    max_leaves: int = 0  # zero: no cap; per analysed frame
    min_leaf_pixels: int = 1
    crop_margin: float = 0.0  # tight bounding rectangle of the SAM mask
    disease_threshold: float = 0.0  # no additional probability filtering
    min_disease_pixels: int = 1
    seed: int = 42
    amp: bool = True
    overlay_alpha: float = 0.45
    cards_per_page: int = 4
    card_size: int = 400
    sample_fps: float = 2.0  # video only; 0: every decoded frame
    max_frames: int = 0  # video only; zero: all selected frames
    video_width: int = 1600  # display only; no inference image resizing


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_rgb(value: Any) -> Image.Image:
    if isinstance(value, (str, Path)):
        with Image.open(value) as im:
            if getattr(im, "n_frames", 1) != 1:
                raise ValueError("Expected one still image, not animated/multipage input")
            return ImageOps.exif_transpose(im).convert("RGB").copy()
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB").copy()
    if isinstance(value, np.ndarray) and value.dtype == np.uint8 and value.ndim == 3 and value.shape[2] == 3:
        return Image.fromarray(value.copy())
    raise TypeError("Expected image path, PIL image or HxWx3 RGB uint8 array (not BGR)")


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("L")) > 0


def font(size: int = 18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def fitted_font(text: str, width: int, size: int = 20):
    while size > 9:
        ft = font(size)
        if ft.getbbox(text)[2] <= width:
            return ft
        size -= 1
    return font(size)


def colour(i: int) -> tuple[int, int, int]:
    return [(20, 120, 180), (150, 90, 190), (40, 150, 100), (200, 130, 20), (175, 75, 145)][(i-1) % 5]


def severity_colour(score: float) -> tuple[int, int, int]:
    """Strictly >50 red; exactly 50 and below black. Display rule only."""
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("Severity must be finite and between 0 and 100")
    return (220, 0, 0) if score > 50.0 else (0, 0, 0)


def overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.45,
            rgb=(255, 45, 45)) -> Image.Image:
    a = np.array(image.convert("RGB"), dtype=np.float32)
    if mask.shape != a.shape[:2] or not 0 <= alpha <= 1:
        raise ValueError("Invalid overlay grid or alpha")
    a[mask] = a[mask] * (1-alpha) + np.asarray(rgb) * alpha
    return Image.fromarray(a.round().clip(0, 255).astype(np.uint8))


def cutout(image: Image.Image, mask: np.ndarray) -> Image.Image:
    if mask.shape != (image.height, image.width):
        raise ValueError("Mask and cutout image must share a grid")
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask.astype(np.uint8) * 255))
    return rgba


def bbox_iou(a, b) -> float:
    inter = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    aa = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    ab = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return inter / (aa+ab-inter) if aa+ab-inter else 0.


def select_boxes(boxes, scores, width, height, threshold):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) != len(scores):
        raise ValueError("Detection score/box mismatch")
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    selected = []
    for i in np.argsort(-scores, kind="stable"):
        b = boxes[i]
        if not np.isfinite(b).all() or not np.isfinite(scores[i]) or b[2]-b[0] < 1 or b[3]-b[1] < 1:
            continue
        if all(bbox_iou(b, boxes[j]) <= threshold for j in selected):
            selected.append(int(i))
    return boxes, selected


def mask_crop_box(mask: np.ndarray, margin: float = 0.0):
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError("Cannot crop empty SAM mask")
    x1, y1, x2, y2 = int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1
    dx, dy = math.ceil((x2-x1)*margin), math.ceil((y2-y1)*margin)
    h, w = mask.shape
    return max(0, x1-dx), max(0, y1-dy), min(w, x2+dx), min(h, y2+dy)


def grounded_postprocess(processor, outputs, ids, size, box_threshold, text_threshold):
    fn = processor.post_process_grounded_object_detection
    key = "box_threshold" if "box_threshold" in inspect.signature(fn).parameters else "threshold"
    return fn(outputs, input_ids=ids, target_sizes=[(size[1], size[0])],
              text_threshold=text_threshold, **{key: box_threshold})[0]


def methods(opt: dict) -> list[str]:
    return ["bicubic", "resshift"] if opt["method"] == "both" else [opt["method"]]


def install_suite_paths(root: Path):
    for p in (root, root / "segmentation", root / "magnification/resshift"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def runtime_versions():
    out = {"python": sys.version, "executable": sys.executable}
    for name in ("torch", "torchvision", "transformers", "SAM-2", "numpy", "Pillow", "omegaconf", "opencv-python", "opencv-python-headless"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def frame_dir(job: dict, frame: dict) -> Path:
    return Path(job["output"]) / frame["directory"]


def detection_worker(job: dict):
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    o = job["options"]
    device = torch.device(o["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in detection environment")
    prompt = o["prompt"].strip().lower()
    if not prompt.endswith("."):
        prompt += "."
    processor = AutoProcessor.from_pretrained(o["dino_model"])
    model = AutoModelForZeroShotObjectDetection.from_pretrained(o["dino_model"]).to(device).eval()
    with torch.inference_mode():
        for frame in job["frames"]:
            out = frame_dir(job, frame)
            image = load_rgb(out / "00_input.png")
            start = time.perf_counter()
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            outputs = model(**inputs)
            result = grounded_postprocess(processor, outputs, inputs.input_ids, image.size,
                                          o["box_threshold"], o["text_threshold"])
            raw_boxes = result["boxes"].detach().float().cpu().numpy()
            scores = result["scores"].detach().float().cpu().numpy()
            boxes, indices = select_boxes(raw_boxes, scores, image.width, image.height, o["box_nms_iou"])
            metadata = {"model": o["dino_model"], "prompt": prompt,
                        "boxes_xyxy": boxes.tolist(), "scores": scores.tolist(),
                        "nms_selected_indices": indices, "seconds": time.perf_counter()-start,
                        "leaves": [], "skipped": []}
            # Guard against pathological non-finite scores returned by a model.
            for j in range(len(metadata["scores"])):
                if not math.isfinite(metadata["scores"][j]):
                    metadata["scores"][j] = None
                metadata["boxes_xyxy"][j] = [float(x) if math.isfinite(x) else None for x in metadata["boxes_xyxy"][j]]
            write_json(out / "detections.json", metadata)
            view = image.copy()
            d = ImageDraw.Draw(view)
            for n, idx in enumerate(indices, 1):
                d.rectangle(tuple(boxes[idx]), outline=colour(n), width=3)
                x, y = map(int, boxes[idx][:2])
                d.text((x, max(0, y-21)), f"DINO {n}: {scores[idx]:.2f}", fill=colour(n), font=font())
            view.save(out / "01_dino_boxes.png")
            del inputs, outputs, result
            print(f'{frame["directory"]}: {len(indices)} leaf boxes after NMS', flush=True)
    write_json(Path(job["output"]) / "dino_model.json", {"model": o["dino_model"], "runtime": runtime_versions()})


def save_extracted_leaf(out: Path, image: Image.Image, mask: np.ndarray,
                        leaf_id: int, box, score: float, quality: float,
                        margin: float) -> dict:
    """Rectangular crop retains observed RGB. Separate alpha mask defines the leaf."""
    folder = out / "leaves" / f"leaf_{leaf_id:03d}"
    folder.mkdir(parents=True)
    x1, y1, x2, y2 = mask_crop_box(mask, margin)
    crop = image.crop((x1, y1, x2, y2))
    cmask = mask[y1:y2, x1:x2]
    save_mask(folder / "01_leaf_mask_frame.png", mask)
    crop.save(folder / "02_crop_rgb.png")
    save_mask(folder / "02_leaf_mask.png", cmask)
    cutout(crop, cmask).save(folder / "02_crop_cutout.png")
    return {"leaf_id": leaf_id, "directory": str(folder.relative_to(out)),
            "dino_box_xyxy": list(map(float, box)), "dino_score": float(score),
            "sam_mask_quality": float(quality), "crop_xyxy": [x1, y1, x2, y2],
            "native_crop_size_wh": list(crop.size), "native_leaf_pixels": int(cmask.sum()),
            "crop_rgb_sha256": sha256(folder / "02_crop_rgb.png")}


def sam_worker(job: dict):
    import torch
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    o = job["options"]
    # No SAM model is needed when there are no candidate boxes anywhere.
    nonempty = any(read_json(frame_dir(job, f)/"detections.json")["nms_selected_indices"] for f in job["frames"])
    sam = None
    if nonempty:
        if o["sam2_checkpoint"]:
            from sam2.build_sam import build_sam2
            sam = SAM2ImagePredictor(build_sam2(o["sam2_config"], o["sam2_checkpoint"], device=o["device"]))
        else:
            sam = SAM2ImagePredictor.from_pretrained(o["sam2_model"], device=o["device"])
        sam.model.eval()
    with torch.inference_mode():
        for frame in job["frames"]:
            out = frame_dir(job, frame)
            meta = read_json(out / "detections.json")
            image = load_rgb(out / "00_input.png")
            view = image.copy()
            accepted = []
            selected = meta["nms_selected_indices"]
            if selected:
                sam.set_image(np.array(image))
            started = time.perf_counter()
            for k, idx in enumerate(selected):
                if o["max_leaves"] and len(accepted) >= o["max_leaves"]:
                    meta["skipped"].extend({"detection_index": i, "reason": "explicit_max_leaves_cap"} for i in selected[k:])
                    break
                box = np.array(meta["boxes_xyxy"][idx], dtype=np.float32)
                masks, quality, _ = sam.predict(box=box, multimask_output=False)
                mask = np.asarray(masks).reshape(-1, image.height, image.width)[0] > 0
                if int(mask.sum()) < o["min_leaf_pixels"]:
                    meta["skipped"].append({"detection_index": idx, "reason": "empty_or_below_min_leaf_pixels"})
                    continue
                if any(np.logical_and(mask, old).sum()/max(1, np.logical_or(mask, old).sum()) > o["mask_nms_iou"] for old in accepted):
                    meta["skipped"].append({"detection_index": idx, "reason": "duplicate_sam_mask"})
                    continue
                q = float(np.asarray(quality).reshape(-1)[0])
                if not math.isfinite(q):
                    raise RuntimeError("SAM2 returned non-finite mask quality")
                accepted.append(mask)
                leaf = save_extracted_leaf(out, image, mask, len(accepted), box,
                                           meta["scores"][idx], q, o["crop_margin"])
                meta["leaves"].append(leaf)
                view = overlay(view, mask, alpha=0.28, rgb=colour(leaf["leaf_id"]))
            d = ImageDraw.Draw(view)
            for leaf in meta["leaves"]:
                x1, y1, x2, y2 = leaf["crop_xyxy"]
                d.rectangle((x1, y1, x2-1, y2-1), outline=colour(leaf["leaf_id"]), width=2)
                d.text((x1+2, y1+2), f'L{leaf["leaf_id"]:03d}', font=font(), fill="white", stroke_width=1, stroke_fill="black")
            view.save(out / "02_sam_leaf_masks.png")
            meta["sam_seconds"] = time.perf_counter()-started
            meta["leaf_ids_are_frame_local"] = True
            write_json(out / "detections.json", meta)
            print(f'{frame["directory"]}: {len(accepted)} distinct SAM leaf crops', flush=True)
    write_json(Path(job["output"])/"sam_model.json", {"model": o["sam2_model"],
        "checkpoint": o["sam2_checkpoint"], "config": o["sam2_config"], "runtime": runtime_versions()})


def positions(length: int, tile: int = 64, stride: int = 48) -> list[int]:
    if length < tile or not 0 < stride <= tile:
        raise ValueError("Invalid padded length or tile stride")
    p = list(range(0, length-tile+1, stride))
    if p[-1] != length-tile:
        p.append(length-tile)
    return p


def reconstruct_resshift(backend, image: Image.Image, mode="tiled", overlap=16,
                        seed=42, amp=True) -> tuple[Image.Image, dict]:
    """Single GPU tile at a time; full RGB/blending tensors stay on the CPU."""
    import torch
    import torch.nn.functional as F
    if mode not in {"tiled", "whole_image"} or not 0 <= overlap < 64:
        raise ValueError("Invalid ResShift sampling mode or overlap")
    w, h = image.size
    x = torch.from_numpy((np.array(image.convert("RGB"), dtype=np.float32)/127.5-1).transpose(2, 0, 1).copy())
    device = torch.device(backend.device)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    calls = 0
    with torch.inference_mode():
        if mode == "tiled":
            pw, ph = max(0, 64-w), max(0, 64-h)
            if pw or ph:
                x = F.pad(x[None], (0, pw, 0, ph), mode="replicate")[0]
            hp, wp = x.shape[-2:]
            accum = torch.zeros((3, hp*4, wp*4), dtype=torch.float32, device="cpu")
            normalizer = torch.zeros((1, hp*4, wp*4), dtype=torch.float32, device="cpu")
            window = torch.hann_window(256, periodic=False, device="cpu").clamp_min(0.001)
            weight = (window[:, None]*window[None, :])[None]
            for yy in positions(hp, stride=64-overlap):
                for xx in positions(wp, stride=64-overlap):
                    torch.manual_seed((seed+calls) % (2**31))
                    if cuda:
                        torch.cuda.manual_seed_all((seed+calls) % (2**31))
                    patch = x[:, yy:yy+64, xx:xx+64][None].to(device)
                    pred = backend.sample(patch, amp=amp)
                    if tuple(pred.shape) != (1, 3, 256, 256) or not torch.isfinite(pred).all():
                        raise RuntimeError("Invalid or non-finite ResShift tile output")
                    rgb = pred[0].detach().to("cpu", dtype=torch.float32)
                    del pred, patch
                    accum[:, yy*4:yy*4+256, xx*4:xx*4+256] += rgb*weight
                    normalizer[:, yy*4:yy*4+256, xx*4:xx*4+256] += weight
                    del rgb
                    calls += 1
            accum.div_(normalizer.clamp_min_(1e-12))
            restored = accum[:, :h*4, :w*4]
            padmode = "replicate" if pw or ph else "none"
        else:
            pw, ph = (-w) % 64, (-h) % 64
            padmode = "reflect" if pw < w and ph < h else "replicate"
            if pw or ph:
                x = F.pad(x[None], (0, pw, 0, ph), mode=padmode)[0]
            else:
                padmode = "none"
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed_all(seed)
            pred = backend.sample(x[None].to(device), amp=amp)
            if tuple(pred.shape) != (1, 3, 4*(h+ph), 4*(w+pw)) or not torch.isfinite(pred).all():
                raise RuntimeError("Invalid or non-finite whole-image ResShift output")
            restored = pred[0, :, :h*4, :w*4].detach().to("cpu", dtype=torch.float32)
            del pred
            calls = 1
        if not torch.isfinite(restored).all():
            raise RuntimeError("Non-finite reconstructed RGB")
        if cuda:
            torch.cuda.synchronize(device)
        a = ((restored.clamp(-1, 1)+1)*127.5).round().byte().permute(1, 2, 0).numpy()
    record = {"sampling_mode": mode, "sample_calls": calls, "scale": 4,
              "input_size_wh": [w, h], "output_size_wh": [4*w, 4*h],
              "padding_right_bottom": [pw, ph], "padding_mode": padmode,
              "seed": seed, "seconds": time.perf_counter()-started,
              "tile_lr": 64 if mode == "tiled" else None,
              "overlap_lr": overlap if mode == "tiled" else None,
              "blend_device": "cpu" if mode == "tiled" else None}
    return Image.fromarray(a), record


def all_leaves(job: dict):
    for frame in job["frames"]:
        out = frame_dir(job, frame)
        for leaf in read_json(out / "detections.json")["leaves"]:
            yield frame, out, leaf


def reconstruction_worker(job: dict):
    o = job["options"]
    backend = None
    model_info = {"methods": methods(o), "runtime": runtime_versions()}
    if "resshift" in methods(o) and any(True for _ in all_leaves(job)):
        from omegaconf import OmegaConf
        from leafdiff.backend import ResShiftBackend
        from leafdiff.common import safe_load
        cfg = OmegaConf.to_container(OmegaConf.load(o["sr_config"]), resolve=True)
        if int(cfg.get("inference", {}).get("scale", 4)) != 4:
            raise ValueError("Expected x4 ResShift configuration")
        if o["sr_checkpoint"]:
            ck = safe_load(o["sr_checkpoint"])
            if "state_dict" not in ck or int(ck.get("step", 0)) < 1:
                raise ValueError("--sr-checkpoint must be a trained suite ResShift checkpoint")
            del ck
        backend = ResShiftBackend(cfg, checkpoint=o["sr_checkpoint"], device=o["device"])
        backend.model.eval().requires_grad_(False)
        backend.ae.eval().requires_grad_(False)
        selected = Path(o["sr_checkpoint"] or cfg["backend"]["diffusion_weights"])
        model_info.update({"checkpoint": str(selected.resolve()), "sha256": sha256(selected),
            "condition": "finetuned" if o["sr_checkpoint"] else "pretrained",
            "sampling_mode": o["sr_mode"]})
    for frame, out, leaf in all_leaves(job):
        base = out / leaf["directory"]
        image = load_rgb(base / "02_crop_rgb.png")
        mask = load_mask(base / "02_leaf_mask.png")
        mask4 = mask.repeat(4, axis=0).repeat(4, axis=1)
        for method in methods(o):
            folder = base / method
            folder.mkdir()
            if method == "bicubic":
                start = time.perf_counter()
                restored = image.resize((image.width*4, image.height*4), Image.Resampling.BICUBIC)
                record = {"sampling_mode": "bicubic", "scale": 4, "sample_calls": 0,
                    "input_size_wh": list(image.size), "output_size_wh": list(restored.size),
                    "seconds": time.perf_counter()-start}
            else:
                # Fixed seed per crop, independent of processing order or output path.
                restored, record = reconstruct_resshift(backend, image, o["sr_mode"], o["overlap"], o["seed"], o["amp"])
            record.update({"method": method, "source_crop_sha256": leaf["crop_rgb_sha256"],
                           "input_downsampled_again": False})
            restored.save(folder / "03_reconstruction.png")
            save_mask(folder / "03_leaf_mask_x4.png", mask4)
            cutout(restored, mask4).save(folder / "03_leaf_cutout.png")
            write_json(folder / "03_reconstruction.json", record)
            print(f'{frame["directory"]}/L{leaf["leaf_id"]:03d}: {method} {image.size} -> {restored.size}; '
                  f'{record["sampling_mode"]}; sample_calls={record["sample_calls"]}', flush=True)
    write_json(Path(job["output"]) / "reconstruction_model.json", model_info)


def disease_maps(ids: np.ndarray, probability: np.ndarray, mask: np.ndarray,
                 num_classes: int, threshold=0., min_pixels=1):
    if ids.shape != mask.shape or ids.shape != probability.shape or not mask.any():
        raise ValueError("Prediction/leaf grids differ or leaf mask is empty")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Disease probabilities are invalid")
    if not 0 <= threshold <= 1 or min_pixels < 1:
        raise ValueError("Invalid disease display thresholds")
    raw = (ids > 0) & (ids < num_classes) & (ids != 255) & mask
    disease = raw & (probability >= threshold)
    count, total = int(disease.sum()), int(mask.sum())
    score = 100.*count/total
    return disease, {"show_magnified": count >= min_pixels, "disease_pixels_x4": count,
        "leaf_pixels_x4": total, "raw_disease_pixels_x4": int(raw.sum()),
        "predicted_severity_percent": score,
        "severity_text_colour": "red" if score > 50 else "black",
        "disease_threshold": threshold, "min_disease_pixels_x4": min_pixels,
        "note": "Predicted visible symptomatic-area percentage; not verified diagnosis. >50 red is a user display rule, not a validated severity grade."}


def native_coverage(disease: np.ndarray, native_wh) -> np.ndarray:
    w, h = native_wh
    if disease.shape != (h*4, w*4):
        raise ValueError("Expected exact x4 disease grid")
    return disease.reshape(h, 4, w, 4).mean(axis=(1, 3)).astype(np.float32)


def stage_panel(images: list[tuple[str, Image.Image]], title: str, output: Path):
    cw, ch = 330, 320
    panel = Image.new("RGB", (cw*len(images), ch+85), "white")
    d = ImageDraw.Draw(panel)
    d.text((12, 8), title, font=fitted_font(title, panel.width-24, 22), fill="black")
    for i, (label, image) in enumerate(images):
        d.text((cw*i+10, 43), label, font=fitted_font(label, cw-20, 16), fill="black")
        view = ImageOps.contain(image.convert("RGB"), (cw-20, ch-12))
        panel.paste(view, (cw*i+(cw-view.width)//2, 78+(ch-12-view.height)//2))
    panel.save(output)


def segment_worker(job: dict):
    from segmentation.api import PlantSegSegmenter
    from plantseg_seg.classes import NUM_CLASSES, CLASS_NAMES, palette
    o = job["options"]
    model = None
    if any(True for _ in all_leaves(job)):
        model = PlantSegSegmenter(o["checkpoint"], device=o["device"], amp=o["amp"])
    for frame in job["frames"]:
        out = frame_dir(job, frame)
        meta = read_json(out / "detections.json")
        for method in methods(o):
            result_dir = out / method
            result_dir.mkdir(exist_ok=True)
            summaries = []
            for leaf in meta["leaves"]:
                base = out / leaf["directory"]
                folder = base / method
                image = load_rgb(folder / "03_reconstruction.png")
                mask = load_mask(folder / "03_leaf_mask_x4.png")
                start = time.perf_counter()
                pred = model.predict(image, gate_mode=None)
                ids = np.asarray(pred.class_ids)
                probability = np.asarray(pred.disease_probability, dtype=np.float32)
                disease, score = disease_maps(ids, probability, mask, NUM_CLASSES,
                                             o["disease_threshold"], o["min_disease_pixels"])
                if ids.dtype.kind not in "ui" or ids.max(initial=0) > 255:
                    raise ValueError("Expected uint8-compatible PlantSeg class IDs")
                Image.fromarray(ids.astype(np.uint8)).save(folder / "04_class_ids_raw.png")
                clipped = np.where(mask, ids, 0).astype(np.uint8)
                Image.fromarray(clipped).save(folder / "04_class_ids_in_leaf.png")
                Image.fromarray(palette()[clipped]).save(folder / "04_class_colours.png")
                save_mask(folder / "04_disease_mask.png", disease)
                np.save(folder / "04_disease_probability.npy", probability)
                Image.fromarray((np.where(mask, probability, 0)*255).round().astype(np.uint8)).save(folder / "04_disease_probability.png")
                visual = overlay(image, disease, o["overlay_alpha"])
                visual.save(folder / "04_overlay.png")
                cutout(visual, mask).save(folder / "04_overlay_cutout.png")
                coverage = native_coverage(disease, leaf["native_crop_size_wh"])
                np.save(folder / "05_native_disease_coverage.npy", coverage)
                values, counts = np.unique(ids[disease], return_counts=True)
                score.update({"method": method, "leaf_id": leaf["leaf_id"],
                    "segmentation_seconds": time.perf_counter()-start,
                    "native_equivalent_disease_area": float(disease.sum())/16,
                    "predicted_classes": [{"id": int(i), "name": CLASS_NAMES[int(i)], "pixels_x4": int(n)} for i, n in zip(values, counts)]})
                write_json(folder / "05_severity.json", score)
                title = f'L{leaf["leaf_id"]:03d} | {method} | predicted symptomatic area {score["predicted_severity_percent"]:.4f}%'
                stage_panel([
                    ("Native RGB crop", load_rgb(base / "02_crop_rgb.png")),
                    (f"{method} x4", image),
                    ("Predicted disease mask in leaf", Image.fromarray(disease.astype(np.uint8)*255)),
                    ("Disease overlay", visual),
                ], title, folder / "06_stages.png")
                summaries.append({**leaf, **score, "frame_index": frame["frame_index"],
                    "timestamp_seconds": frame["timestamp_seconds"]})
                print(f'{frame["directory"]}/L{leaf["leaf_id"]:03d} {method}: '
                      f'{score["predicted_severity_percent"]:.4f}% area; positive={score["show_magnified"]}', flush=True)
                del pred, ids, probability, disease, image, visual, coverage
            write_json(result_dir / "leaf_results.json", summaries)
    write_json(Path(job["output"])/"segformer_model.json", {"checkpoint": o["checkpoint"],
        "sha256": sha256(Path(o["checkpoint"])), "num_classes": NUM_CLASSES,
        "input_canvas": model.image_size if model is not None else None,
        "runtime": runtime_versions()})


SEVERITY_FIELDS = ["frame_index", "timestamp_seconds", "leaf_id", "method", "show_magnified",
    "predicted_severity_percent", "severity_text_colour", "disease_pixels_x4", "leaf_pixels_x4",
    "native_leaf_pixels", "dino_score", "sam_mask_quality"]


def exact_replacement(image: Image.Image, leaves: list[dict], out: Path, method: str):
    """Resize reconstructed overlay back to native footprint; paste ONLY in SAM mask.

    In overlapping predictions, the highest-DINO-score positive leaf owns pixels.
    No claim that the resulting native-size image displays x4 magnification.
    """
    result = np.array(image).copy()
    owner = np.full((image.height, image.width), -1, dtype=np.int32)
    coverage = np.zeros((image.height, image.width), dtype=np.float32)
    positives = sorted((x for x in leaves if x["show_magnified"]), key=lambda x: (-x["dino_score"], x["leaf_id"]))
    for leaf in positives:
        x1, y1, x2, y2 = leaf["crop_xyxy"]
        base = out / leaf["directory"]
        mask = load_mask(base / "02_leaf_mask.png")
        if mask.shape != (y2-y1, x2-x1):
            raise ValueError("Stored crop geometry and SAM mask differ")
        sr = load_rgb(base / method / "04_overlay.png")
        if sr.size != ((x2-x1)*4, (y2-y1)*4):
            raise ValueError("Stored overlay is not x4 the original crop")
        # BOX averages exactly the aligned x4 cells; no category interpolation.
        patch = np.array(sr.resize((x2-x1, y2-y1), Image.Resampling.BOX))
        use = mask & (owner[y1:y2, x1:x2] < 0)
        result[y1:y2, x1:x2][use] = patch[use]
        owner[y1:y2, x1:x2][use] = leaf["leaf_id"]
        cov = np.load(base / method / "05_native_disease_coverage.npy", allow_pickle=False)
        if cov.shape != mask.shape:
            raise ValueError("Native disease coverage/crop mismatch")
        coverage[y1:y2, x1:x2][use] = cov[use]
    return Image.fromarray(result), owner, coverage


def annotate_sources(image: Image.Image, leaves: list[dict], full_score=True):
    view = image.copy()
    d = ImageDraw.Draw(view)
    for leaf in leaves:
        if not leaf["show_magnified"]:
            continue
        x1, y1, x2, y2 = leaf["crop_xyxy"]
        d.rectangle((x1, y1, x2-1, y2-1), outline=colour(leaf["leaf_id"]), width=2)
        text = f'L{leaf["leaf_id"]:03d}'
        if full_score:
            text += f' | {leaf["predicted_severity_percent"]:.2f}%'
        ft = fitted_font(text, max(20, image.width-6), min(24, max(12, image.width//65)))
        box = d.textbbox((0, 0), text, font=ft)
        tw, th = box[2]+8, box[3]+8
        xx = min(max(0, x1), max(0, image.width-tw))
        yy = min(max(0, y1-th), max(0, image.height-th))
        d.rectangle((xx, yy, xx+tw, yy+th), fill="white", outline=colour(leaf["leaf_id"]))
        d.text((xx+4, yy+2), text, font=ft, fill=severity_colour(leaf["predicted_severity_percent"]))
    return view


def external_panels(image: Image.Image, leaves: list[dict], out: Path,
                    method: str, o: dict) -> dict:
    """Original frame retained at native size on the left; enlarged cards outside."""
    positives = sorted((x for x in leaves if x["show_magnified"]), key=lambda x: x["leaf_id"])
    result_dir = out / method
    base = annotate_sources(image, leaves, full_score=False)
    count = o["cards_per_page"]
    columns = 1 if count == 1 else 2
    rows = math.ceil(count/columns)
    size, gap, gutter = o["card_size"], 14, 58
    card_h = size+86
    header_h = 76
    panel_w = columns*size+(columns+1)*gap
    canvas_w = image.width+gutter+panel_w
    canvas_h = max(image.height, rows*card_h+(rows+1)*gap)+header_h
    pages, mappings = [], []
    groups = [positives[i:i+count] for i in range(0, len(positives), count)] or [[]]
    for page_index, group in enumerate(groups, 1):
        canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
        canvas.paste(base, (0, header_h))
        d = ImageDraw.Draw(canvas)
        title = f'{method.upper()} x4 | {len(positives)} disease-positive leaves | page {page_index}/{len(groups)}'
        d.text((12, 9), title, fill="black", font=fitted_font(title, canvas_w-24, 25))
        note = "Original frame at left; reconstructed leaf overlays at right. Predicted area: >50% red, <=50% black."
        d.text((12, 43), note, fill="black", font=fitted_font(note, canvas_w-24, 17))
        if not group:
            msg = "No disease pixels predicted in the processed leaves."
            if not leaves:
                msg = "No leaf accepted by DINO/SAM2; disease not assessed."
            d.text((image.width+gutter+gap, header_h+35), msg,
                   font=fitted_font(msg, panel_w-2*gap, 20), fill="black")
        # Connection lines drawn first so they cannot obscure cards.
        slots = []
        for j, leaf in enumerate(group):
            cx = image.width+gutter+gap+(j%columns)*(size+gap)
            cy = header_h+gap+(j//columns)*(card_h+gap)
            x1, y1, x2, y2 = leaf["crop_xyxy"]
            source_x, source_y = (x1+x2)//2, (y1+y2)//2+header_h
            d.line([(source_x, source_y), (image.width+gutter//2, source_y),
                    (image.width+gutter//2, cy+card_h//2), (cx, cy+card_h//2)],
                   fill=colour(leaf["leaf_id"]), width=2)
            slots.append((leaf, cx, cy))
        for leaf, cx, cy in slots:
            card = Image.new("RGB", (size, card_h), (248, 248, 248))
            cd = ImageDraw.Draw(card)
            score = leaf["predicted_severity_percent"]
            label = f'L{leaf["leaf_id"]:03d} | Severity: {score:.2f}%'
            cd.text((10, 7), label, font=fitted_font(label, size-20, 23), fill=severity_colour(score))
            path = out / leaf["directory"] / method / "04_overlay_cutout.png"
            with Image.open(path) as handle:
                rgba = handle.convert("RGBA").copy()
            # Never upscale a card beyond the x4 reconstruction; fit only if needed.
            sf = min(1., (size-20)/rgba.width, (size-20)/rgba.height)
            disp_wh = (max(1, round(rgba.width*sf)), max(1, round(rgba.height*sf)))
            view = rgba.resize(disp_wh, Image.Resampling.LANCZOS)
            px, py = (size-view.width)//2, 40+(size-view.height)//2
            card.paste(view, (px, py), view)
            scale = view.width/leaf["native_crop_size_wh"][0]
            foot = f'Reconstructed x4 | shown {scale:.2f}x native'
            cd.text((10, card_h-27), foot, font=fitted_font(foot, size-20, 16), fill="black")
            cd.rectangle((0, 0, size-1, card_h-1), outline=colour(leaf["leaf_id"]), width=2)
            canvas.paste(card, (cx, cy))
            mappings.append({"leaf_id": leaf["leaf_id"], "page": page_index,
                "card_xyxy": [cx, cy, cx+size, cy+card_h],
                "native_crop_xyxy": leaf["crop_xyxy"], "display_scale_relative_to_native": scale,
                "severity_colour": leaf["severity_text_colour"]})
        name = "final_external.png" if page_index == 1 else f"final_external_page_{page_index:03d}.png"
        canvas.save(result_dir / name)
        pages.append(name)
    return {"pages": pages, "cards": mappings, "positive_leaves": len(positives),
            "external_canvas_wh": [canvas_w, canvas_h],
            "note": "Full x4 files are saved separately. Fitted cards may display less than x4; source geometry at left is unchanged. Overflow pages are saved as PNGs and as separate synchronized videos."}


def finish_frame(job: dict, frame: dict) -> dict:
    out = frame_dir(job, frame)
    o = job["options"]
    image = load_rgb(out / "00_input.png")
    summaries = {}
    for method in methods(o):
        result_dir = out / method
        leaves = read_json(result_dir / "leaf_results.json")
        exact, owners, cov = exact_replacement(image, leaves, out, method)
        exact.save(result_dir / "final_exact_clean.png")
        annotate_sources(exact, leaves).save(result_dir / "final_exact.png")
        np.save(result_dir / "pixel_owner.npy", owners)
        np.save(result_dir / "native_disease_coverage.npy", cov)
        display = external_panels(image, leaves, out, method, o)
        write_json(result_dir / "display.json", display)
        write_csv(result_dir / "severity.csv", leaves, SEVERITY_FIELDS)
        status = "no_leaf_detected" if not leaves else ("disease_predicted" if display["positive_leaves"] else "no_disease_predicted")
        summaries[method] = {"status": status, "leaves_processed": len(leaves),
            "disease_positive_leaves": display["positive_leaves"], "external_pages": display["pages"],
            "exact_overlay": str((result_dir / "final_exact.png").relative_to(out)),
            "external_overlay": str((result_dir / "final_external.png").relative_to(out))}
    write_json(out / "summary.json", {"frame": frame, "methods": summaries,
        "severity_colour_rule": ">50 red; <=50 black", "leaf_ids_are_frame_local": True})
    return summaries


def prepare_image(source: Any, output: Path) -> tuple[list[dict], dict]:
    folder = output / "image"
    folder.mkdir()
    image = load_rgb(source)
    image.save(folder / "00_input.png")
    return [{"directory": "image", "frame_index": 0, "timestamp_seconds": None,
             "input_size_wh": list(image.size)}], {"kind": "image", "source": str(source) if isinstance(source, (str, Path)) else "in_memory_RGB"}


def sample_index(k: int, source_fps: float, target_fps: float) -> int:
    """Nearest source frame on a regular sampling timeline; no duplicate frames."""
    if source_fps <= 0 or target_fps <= 0 or target_fps > source_fps:
        raise ValueError("Require 0 < target_fps <= source_fps")
    return int(math.floor(k*source_fps/target_fps + 0.5))


def prepare_video(source: Path, output: Path, o: dict) -> tuple[list[dict], dict]:
    import cv2
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")
    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("Video FPS unavailable; convert to a constant-frame-rate video before use")
        requested = o["sample_fps"]
        target = source_fps if requested == 0 else min(source_fps, requested)
        frames, index, k = [], 0, 0
        next_index = sample_index(k, source_fps, target)
        observed_hw = None
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if observed_hw is None:
                observed_hw = bgr.shape[:2]
            elif bgr.shape[:2] != observed_hw:
                raise ValueError("Video frame dimensions changed; normalize video first")
            if index == next_index:
                name = f"frames/frame_{index:06d}"
                folder = output / name
                folder.mkdir(parents=True)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(folder / "00_input.png")
                reported_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                frames.append({"directory": name, "frame_index": index,
                    "timestamp_seconds": index/source_fps,
                    "decoder_position_ms": reported_ms if math.isfinite(reported_ms) else None,
                    "output_timeline_seconds": k/target,
                    "input_size_wh": [bgr.shape[1], bgr.shape[0]]})
                k += 1
                if o["max_frames"] and len(frames) >= o["max_frames"]:
                    break
                next_index = sample_index(k, source_fps, target)
            index += 1
        if not frames:
            raise ValueError("No video frames could be decoded")
    finally:
        cap.release()
    info = {"kind": "video", "source": str(source.resolve()), "source_fps": source_fps,
        "requested_sample_fps": requested, "output_fps": target, "analysed_frames": len(frames),
        "max_frames": o["max_frames"], "tracking": False, "audio_preserved": False,
        "timing_assumption": "Constant-frame-rate input; timestamps=source frame index/FPS. Variable-frame-rate input must be converted to CFR for accurate timing.",
        "output_duration_seconds": len(frames)/target,
        "sampling_note": "One newly analysed source frame per output frame; no copied/proxy masks for skipped frames."}
    write_csv(output / "frames.csv", frames, ["directory", "frame_index", "timestamp_seconds", "output_timeline_seconds", "decoder_position_ms"])
    return frames, info


def video_canvas(image: Image.Image, width: int) -> Image.Image:
    """Presentation resize only, even dimensions for MP4. No crop/warp."""
    scale = min(1., width/image.width)
    w, h = max(2, int(image.width*scale)), max(2, int(image.height*scale))
    fitted = image.resize((w, h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (w+w%2, h+h%2+26), "white")
    canvas.paste(fitted, (0, 0))
    return canvas


def render_videos(job: dict) -> list[dict]:
    """Native-footprint video plus synchronized external-panel pages; no leaf dropped."""
    import cv2
    out = Path(job["output"]) / "videos"
    out.mkdir()
    reports = []
    fps = job["input"]["output_fps"]
    for method in methods(job["options"]):
        page_count = max(len(read_json(frame_dir(job, f)/method/"display.json")["pages"])
                         for f in job["frames"])
        streams = [("exact", 0, "final_exact.png")]
        streams += [("external", page, "final_external.png" if page == 1 else f"final_external_page_{page:03d}.png")
                    for page in range(1, page_count+1)]
        for display, page, source_name in streams:
            suffix = f"_page_{page:03d}" if page > 1 else ""
            path = out / f"{method}_{display}{suffix}.mp4"
            # All frame pages share dimensions, including those with no positives.
            reference_name = "final_exact.png" if display == "exact" else "final_external.png"
            first = video_canvas(load_rgb(frame_dir(job, job["frames"][0])/method/reference_name),
                                 job["options"]["video_width"])
            size = first.size
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            if not writer.isOpened():
                raise RuntimeError("MP4 writer unavailable. Frame PNGs are intact; use an OpenCV build with FFmpeg.")
            try:
                for f in job["frames"]:
                    frame_out = frame_dir(job, f)
                    source_path = frame_out/method/source_name
                    if source_path.is_file():
                        source = load_rgb(source_path)
                    else:
                        # No page-N leaves in THIS frame: never repeat old predictions.
                        original = load_rgb(frame_out/"00_input.png")
                        shape = read_json(frame_out/method/"display.json")["external_canvas_wh"]
                        source = Image.new("RGB", tuple(shape), "white")
                        source.paste(original, (0, 76))
                        d = ImageDraw.Draw(source)
                        title = f"{method.upper()} x4 | external page {page}/{page_count}"
                        d.text((12, 9), title, font=fitted_font(title, shape[0]-24, 25), fill="black")
                        msg = "No additional disease-positive leaves on this page in this frame."
                        d.text((12, 43), msg, font=fitted_font(msg, shape[0]-24, 18), fill="black")
                    view = video_canvas(source, job["options"]["video_width"])
                    if view.size != size:
                        raise ValueError("Video visualization dimensions changed unexpectedly")
                    # Timestamp strip is in ADDED bottom padding, not over source pixels.
                    d = ImageDraw.Draw(view)
                    label = f'source frame {f["frame_index"]} | t={f["timestamp_seconds"]:.3f}s | analysed {fps:.3g} fps | IDs frame-local'
                    d.text((5, view.height-23), label, fill="black", font=fitted_font(label, view.width-10, 15))
                    writer.write(cv2.cvtColor(np.array(view), cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Empty output video: {path}")
            reports.append({"file": str(path.relative_to(Path(job["output"]))), "method": method,
                "display": display, "external_page": page if display == "external" else None,
                "frames": len(job["frames"]), "fps": fps, "size_wh": list(size),
                "note": "Overflow leaf cards are in synchronized additional external-page videos; no previous-frame masks are reused."})
    return reports


def resolve_options(options: Options) -> dict:
    o = asdict(options)
    root = Path(o["project_root"]).expanduser().resolve()
    for name in ("run.py", "segmentation/api.py"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"--project-root must contain {name}")
    o["project_root"] = str(root)
    if o["method"] not in {"bicubic", "resshift", "both"} or o["sr_mode"] not in {"tiled", "whole_image"}:
        raise ValueError("Unknown method or sr_mode")
    if o["method"] == "bicubic" and o["sr_checkpoint"]:
        raise ValueError("Bicubic has no ResShift checkpoint; remove --sr-checkpoint")
    required = ["checkpoint", "sam2_checkpoint"]
    if "resshift" in methods(o):
        required += ["sr_config", "sr_checkpoint"]
        if not (root/"magnification/resshift/leafdiff/backend.py").is_file():
            raise FileNotFoundError("The suite's ResShift backend is missing")
    for key in required:
        if o[key]:
            p = Path(o[key]).expanduser()
            p = p if p.is_absolute() else root/p
            if not p.is_file():
                raise FileNotFoundError(f"{key}: {p}")
            o[key] = str(p.resolve())
    if bool(o["sam2_config"]) != bool(o["sam2_checkpoint"]):
        raise ValueError("Local SAM2 requires BOTH --sam2-config and --sam2-checkpoint")
    for key in ("det_python", "sr_python", "seg_python"):
        if key == "sr_python" and o["method"] == "bicubic":
            o[key] = sys.executable
        exe = shutil.which(o[key])
        if not exe:
            raise FileNotFoundError(f"Python interpreter not found: {o[key]}")
        o[key] = str(Path(exe).absolute())
    for key in ("box_threshold", "text_threshold", "box_nms_iou", "mask_nms_iou", "disease_threshold", "overlay_alpha"):
        if not math.isfinite(o[key]) or not 0 <= o[key] <= 1:
            raise ValueError(f"{key} must be within [0,1]")
    if not 0 <= o["crop_margin"] <= 0.5 or not 0 <= o["overlap"] < 64:
        raise ValueError("Require crop_margin 0..0.5 and overlap 0..63")
    if not 0 <= o["seed"] < 2**31 or not o["prompt"].strip():
        raise ValueError("Invalid seed or empty detector prompt")
    if min(o["max_leaves"], o["max_frames"]) < 0 or min(o["min_leaf_pixels"], o["min_disease_pixels"]) < 1:
        raise ValueError("Invalid count limit")
    if not 1 <= o["cards_per_page"] <= 12 or o["card_size"] < 160 or o["video_width"] < 320:
        raise ValueError("Require 1..12 cards, card_size>=160, video_width>=320")
    if not math.isfinite(o["sample_fps"]) or o["sample_fps"] < 0:
        raise ValueError("sample_fps must be >=0; zero means every source frame")
    return o


def run_worker(stage: str, job_path: Path, python: str, root: str, log: Path):
    cmd = [python, "-u", str(SCRIPT), "_worker", stage, "--job", str(job_path)]
    print("RUN:", shlex.join(cmd), flush=True)
    start = time.perf_counter()
    with log.open("w", encoding="utf-8") as f:
        f.write(shlex.join(cmd)+"\n")
        proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            code = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    if code:
        raise RuntimeError(f"Stage {stage} failed (exit {code}); see {log}. No silent fallback or image skipping.")
    return time.perf_counter()-start


def run_pipeline(*, image=None, video=None, output: str | Path, options: Options | None = None) -> dict:
    """Image supports path/PIL/RGB array. Video accepts a file path. Fresh output required."""
    if (image is None) == (video is None):
        raise ValueError("Supply exactly one image or video")
    o = resolve_options(options or Options())
    out = Path(output).expanduser().resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError("Use a new/empty output directory; stale results are never silently reused")
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        if video is None:
            frames, info = prepare_image(image, out)
        else:
            src = Path(video).expanduser().resolve()
            if not src.is_file():
                raise FileNotFoundError(src)
            frames, info = prepare_video(src, out, o)
        job = {"schema": "leaf_disease_pipeline_v1", "output": str(out), "options": o,
               "input": info, "frames": frames}
        path = out / "job.json"
        write_json(path, job)
        (out / "logs").mkdir()
        timing = {}
        for stage, interpreter in (("dino", o["det_python"]), ("sam", o["det_python"]),
                                   ("reconstruct", o["sr_python"]), ("segment", o["seg_python"])):
            timing[stage] = run_worker(stage, path, interpreter, o["project_root"], out/"logs"/f"{stage}.log")
        frames_result = [finish_frame(job, f) for f in frames]
        records = []
        for f in frames:
            for method in methods(o):
                records.extend(read_json(frame_dir(job, f)/method/"leaf_results.json"))
        write_csv(out/"severity.csv", records, SEVERITY_FIELDS)
        videos = render_videos(job) if video is not None else []
        summary = {"status": "complete", "input": info, "methods": methods(o),
            "frames": frames, "frame_results": frames_result, "videos": videos,
            "stage_seconds": timing, "total_seconds": time.perf_counter()-start,
            "severity_colour_rule": ">50 red; <=50 black", "leaf_ids_are_frame_local": True,
            "inference_metrics_not_accuracy": "No reference masks were supplied; severity values and timings are predictions/runtime, not Dice/IoU accuracy.",
            "memory": "Separate GPU processes per model stage; ResShift tiled by default. Source/crop pixels never silently reduced."}
        from leaf_display_fix import render_saved_results

        summary["magnified_display"] = render_saved_results(
            out,
            display_scale=4.0,
            write_videos=video is not None,
        )

        write_json(out/"summary.json", summary)
        print(f"DONE: {out}", flush=True)
        return summary
    except BaseException as exc:
        write_json(out/"error.json", {"status": "failed", "error": str(exc), "type": type(exc).__name__,
            "traceback": traceback.format_exc(), "note": "Prior stage outputs retained. Use a fresh output directory on rerun."})
        raise


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        p = argparse.ArgumentParser()
        p.add_argument("_worker")
        p.add_argument("stage", choices=["dino", "sam", "reconstruct", "segment"])
        p.add_argument("--job", required=True)
        a = p.parse_args()
        job = read_json(Path(a.job))
        install_suite_paths(Path(job["options"]["project_root"]))
        {"dino": detection_worker, "sam": sam_worker,
         "reconstruct": reconstruction_worker, "segment": segment_worker}[a.stage](job)
        return
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--image", help="One still image; no pre-extracted crop required")
    inp.add_argument("--video", help="Offline constant-frame-rate video")
    p.add_argument("--output", required=True, help="New/empty output directory")
    defaults = asdict(Options())
    for key, default in defaults.items():
        if key == "amp":
            p.add_argument("--no-amp", action="store_true")
            continue
        choices = {"method": ["bicubic", "resshift", "both"], "sr_mode": ["tiled", "whole_image"]}.get(key)
        p.add_argument("--"+key.replace("_", "-"), dest=key,
                       type=type(default) if default is not None else str,
                       default=default, choices=choices)
    a = vars(p.parse_args())
    image, video, output = a.pop("image"), a.pop("video"), a.pop("output")
    a["amp"] = not a.pop("no_amp")
    run_pipeline(image=image, video=video, output=output, options=Options(**a))


if __name__ == "__main__":
    main()
