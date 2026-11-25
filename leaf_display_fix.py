#!/usr/bin/env python3
"""Re-render saved leaf_disease_pipeline.py results; no models or CUDA required.

Produces two NEW views without changing inference or severity:
* final_inplace_magnified.png: enlarged SAM-masked reconstruction centred on
  its original crop location, with severity. Keeps source-frame dimensions.
* final_external_magnified.png: source frame plus full-size reconstructed
  cutouts outside it, with severity. No automatic shrinking into small cards.

Default display scale is 4 native pixels -> 4 display pixels, matching x4 SR.
The in-place composite is a visualization, NOT a spatially faithful mask for
accuracy scoring. Enlarged cutouts can overlap and are clipped at frame edges.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class DisplayOptions:
    display_scale: float = 4.0
    cards_per_page: int = 4
    individual_views: bool = True
    max_canvas_pixels: int = 60_000_000
    video_width: int = 1920  # video presentation only; PNGs are not resized


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def child(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root.resolve() and root.resolve() not in path.parents:
        raise ValueError(f"Path leaves its result directory: {relative}")
    return path


def rgb(path: Path) -> Image.Image:
    # These are already normalized PNGs saved by the pipeline, not raw EXIF inputs.
    with Image.open(path) as im:
        return im.convert("RGB").copy()


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L")) > 0


def font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def fit_font(text: str, width: int, size: int):
    while size > 9 and font(size).getbbox(text)[2] > width:
        size -= 1
    return font(size)


def severity_colour(score: float) -> tuple[int, int, int]:
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("Severity must be within [0,100]")
    return (220, 0, 0) if score > 50 else (0, 0, 0)


def id_colour(i: int):
    return [(25, 105, 155), (140, 85, 170), (45, 140, 90),
            (165, 115, 30), (155, 75, 115)][(i - 1) % 5]


def rounded(x: float) -> int:
    return int(math.floor(x + 0.5))


def scaled_size(leaf: dict, scale: float) -> tuple[int, int]:
    x1, y1, x2, y2 = leaf["crop_xyxy"]
    return max(1, rounded((x2-x1)*scale)), max(1, rounded((y2-y1)*scale))


def label(leaf: dict) -> str:
    # Extra decimal precision avoids labelling a >50% red score as exactly 50.00%.
    s = float(leaf["predicted_severity_percent"])
    stext = f"{s:.2f}"
    if s > 50 and float(stext) <= 50:
        stext = ">50.00"
    return f'L{int(leaf["leaf_id"]):03d} | Severity {stext}%'


def validate_options(o: DisplayOptions) -> None:
    if not math.isfinite(o.display_scale) or not 1 <= o.display_scale <= 4:
        raise ValueError("display_scale must be 1..4; the saved reconstruction is x4")
    if not 1 <= o.cards_per_page <= 12:
        raise ValueError("cards_per_page must be 1..12")
    if o.max_canvas_pixels < 1 or o.video_width < 0:
        raise ValueError("Invalid canvas limit/video width")


def load_leaves(frame: Path, method: str, frame_size: tuple[int, int]) -> list[dict]:
    leaves = read_json(frame / method / "leaf_results.json")
    if not isinstance(leaves, list):
        raise ValueError("leaf_results.json must contain a list")
    seen = set()
    for leaf in leaves:
        i = int(leaf["leaf_id"])
        if i in seen:
            raise ValueError(f"Duplicate leaf ID: {i}")
        seen.add(i)
        if not isinstance(leaf["show_magnified"], bool):
            raise ValueError("show_magnified must be a JSON boolean")
        severity_colour(float(leaf["predicted_severity_percent"]))
        box = leaf["crop_xyxy"]
        if len(box) != 4 or any(not isinstance(v, int) for v in box):
            raise ValueError("Expected integer crop_xyxy coordinates")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= frame_size[0] and 0 <= y1 < y2 <= frame_size[1]):
            raise ValueError("Crop coordinates lie outside the saved input frame")
        child(frame, leaf["directory"])
        score = float(leaf.get("dino_score", 0))
        if not math.isfinite(score):
            raise ValueError("Non-finite detection score")
    return leaves


def positive_leaves(leaves: list[dict]) -> list[dict]:
    return sorted((x for x in leaves if x["show_magnified"]), key=lambda x: int(x["leaf_id"]))


def overlay_cutout(frame: Path, leaf: dict, method: str, scale: float) -> Image.Image:
    """Use the ACTUAL saved SR/bicubic overlay and SAM mask, not a resized input."""
    base = child(frame, leaf["directory"]) / method
    im = rgb(base / "04_overlay.png")
    mask = load_mask(base / "03_leaf_mask_x4.png")
    x1, y1, x2, y2 = leaf["crop_xyxy"]
    expected = ((x2-x1)*4, (y2-y1)*4)
    if im.size != expected or mask.shape != (expected[1], expected[0]):
        raise ValueError(f'L{leaf["leaf_id"]}: x4 overlay/mask dimensions do not match crop coordinates')
    if not mask.any():
        raise ValueError("Positive leaf has an empty SAM mask")
    alpha = Image.fromarray(mask.astype(np.uint8)*255)
    target = scaled_size(leaf, scale)
    if im.size != target:
        im = im.resize(target, Image.Resampling.LANCZOS)
        alpha = alpha.resize(target, Image.Resampling.NEAREST)
    result = im.convert("RGBA")
    result.putalpha(alpha)
    return result


def anchored_box(leaf: dict, patch_size: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = leaf["crop_xyxy"]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = patch_size
    left, top = rounded(cx-w/2), rounded(cy-h/2)
    return left, top, left+w, top+h


def paste_clipped(canvas: Image.Image, patch: Image.Image, box: tuple[int, int, int, int],
                  owners: np.ndarray | None = None, leaf_id: int = 0) -> dict:
    l, t, r, b = box
    if (r-l, b-t) != patch.size:
        raise ValueError("Paste geometry mismatch")
    vl, vt, vr, vb = max(l, 0), max(t, 0), min(r, canvas.width), min(b, canvas.height)
    visible = [vl, vt, vr, vb]
    count = 0
    if vl < vr and vt < vb:
        part = patch.crop((vl-l, vt-t, vr-l, vb-t))
        mask = np.asarray(part.getchannel("A")) > 0
        count = int(mask.sum())
        canvas.paste(part, (vl, vt), part.getchannel("A"))
        if owners is not None:
            owners[vt:vb, vl:vr][mask] = leaf_id
    return {"requested_box_xyxy": list(box), "visible_box_xyxy": visible,
            "edge_clipped": [l, t, r, b] != visible, "visible_leaf_pixels_before_overlap": count}


def rectangles_overlap(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def add_labels(image: Image.Image, records: list[dict]) -> Image.Image:
    view = image.copy()
    d = ImageDraw.Draw(view)
    used = []
    for rec in sorted(records, key=lambda x: x["leaf_id"]):
        leaf = rec["leaf"]
        txt = label(leaf)
        ft = fit_font(txt, max(8, image.width-12), min(24, max(12, image.width//60)))
        bb = d.textbbox((0, 0), txt, font=ft)
        w, h = bb[2]-bb[0]+12, bb[3]-bb[1]+10
        l, t, r, b = rec["visible_box_xyxy"]
        cx, cy = rec["anchor_xy"]
        candidates = [(l, t-h-3), (l, b+3), (r-w, t+3), (l, t+3), (cx-w/2, cy)]
        selected = None
        for x, y in candidates:
            x = int(min(max(0, x), max(0, image.width-w)))
            y = int(min(max(0, y), max(0, image.height-h)))
            candidate = (x, y, x+w, y+h)
            if selected is None:
                selected = candidate
            if not any(rectangles_overlap(candidate, prev) for prev in used):
                selected = candidate
                break
        x, y, _, _ = selected
        used.append(selected)
        c = id_colour(int(leaf["leaf_id"]))
        d.line([(rounded(cx), rounded(cy)), (x+w//2, y+h//2)], fill=c, width=1)
        d.rectangle(selected, fill="white", outline=c, width=1)
        d.text((x+6-bb[0], y+5-bb[1]), txt, font=ft,
               fill=severity_colour(float(leaf["predicted_severity_percent"])))
        rec["label_box_xyxy"] = list(selected)
    return view


def render_inplace(frame: Path, method: str, leaves: list[dict], o: DisplayOptions) -> dict:
    original = rgb(frame / "00_input.png")
    canvas = original.copy()
    owners = np.zeros((canvas.height, canvas.width), dtype=np.int32)
    result_dir = frame / method
    individual = result_dir / "inplace_magnified_by_leaf"
    if individual.exists():
        for path in individual.glob("leaf_*.png"):
            path.unlink()
    if o.individual_views:
        individual.mkdir(exist_ok=True)
    records = []
    # Higher detection score is rendered LAST, hence visible on top. Does not
    # affect model predictions, the disease mask, or any severity calculation.
    ordered = sorted(positive_leaves(leaves), key=lambda x: (float(x.get("dino_score", 0)), -int(x["leaf_id"])))
    for order, leaf in enumerate(ordered):
        patch = overlay_cutout(frame, leaf, method, o.display_scale)
        box = anchored_box(leaf, patch.size)
        details = paste_clipped(canvas, patch, box, owners, int(leaf["leaf_id"]))
        x1, y1, x2, y2 = leaf["crop_xyxy"]
        rec = {"leaf": leaf, "leaf_id": int(leaf["leaf_id"]), "paint_order": order,
               "source_crop_xyxy": [x1, y1, x2, y2], "anchor_xy": [(x1+x2)/2, (y1+y2)/2],
               "display_size_wh": list(patch.size), "display_scale_requested": o.display_scale,
               "display_scale_xy": [patch.width/(x2-x1), patch.height/(y2-y1)],
               "predicted_severity_percent": float(leaf["predicted_severity_percent"]),
               "severity_colour": "red" if float(leaf["predicted_severity_percent"]) > 50 else "black",
               **details}
        if o.individual_views:
            single = original.copy()
            paste_clipped(single, patch, box)
            add_labels(single, [dict(rec)]).save(individual / f'leaf_{int(leaf["leaf_id"]):03d}.png')
        records.append(rec)
    for rec in records:
        remaining = int((owners == rec["leaf_id"]).sum())
        rec["visible_leaf_pixels_after_overlap"] = remaining
        rec["hidden_by_other_leaves_pixels"] = rec["visible_leaf_pixels_before_overlap"]-remaining
    clean_name, final_name = "final_inplace_magnified_clean.png", "final_inplace_magnified.png"
    canvas.save(result_dir / clean_name)
    add_labels(canvas, records).save(result_dir / final_name)
    np.save(result_dir / "inplace_magnified_owner.npy", owners)
    for rec in records:
        rec.pop("leaf")
    return {"file": final_name, "clean_file": clean_name, "frame_size_wh": list(original.size),
            "leaves": records, "positive_leaves": len(records),
            "note": "Magnified display only: alpha-masked leaves centred at source crop centres. Frame edges clip the cutouts, not resize or shift them. Larger leaves may occlude other content. No background inpainting. Do not score localization using this distorted display."}


def external_layout(frame_size: tuple[int, int], leaves: list[dict], o: DisplayOptions) -> dict:
    sizes = [scaled_size(x, o.display_scale) for x in positive_leaves(leaves)]
    maxw = max([x[0] for x in sizes] or [0])
    maxh = max([x[1] for x in sizes] or [0])
    card_w, image_h = max(300, maxw+24), max(140, maxh)
    card_h = image_h+90
    cols = 1 if o.cards_per_page == 1 else 2
    rows = math.ceil(o.cards_per_page/cols)
    gap, gutter, header = 16, 64, 80
    width = frame_size[0]+gutter+cols*card_w+(cols+1)*gap
    height = max(frame_size[1], rows*card_h+(rows+1)*gap)+header
    if width*height > o.max_canvas_pixels:
        raise ValueError(f"External canvas {width}x{height} exceeds --max-canvas-pixels. "
                         "Use --cards-per-page 1, --display-scale 2, or explicitly raise the memory limit. No silent shrinking.")
    return {"width": width, "height": height, "card_w": card_w, "card_h": card_h,
            "image_h": image_h, "cols": cols, "gap": gap, "gutter": gutter, "header": header}


def render_external(frame: Path, method: str, leaves: list[dict], o: DisplayOptions,
                    layout: dict | None = None, minimum_pages: int = 1) -> dict:
    original = rgb(frame / "00_input.png")
    layout = layout or external_layout(original.size, leaves, o)
    positives = positive_leaves(leaves)
    pages = max(1, minimum_pages, math.ceil(len(positives)/o.cards_per_page))
    result_dir = frame / method
    # Remove only old extra pages belonging to THIS renderer, not inference files.
    for stale in result_dir.glob("final_external_magnified_page_*.png"):
        stale.unlink()
    out_files, cards = [], []
    W, H, hh = layout["width"], layout["height"], layout["header"]
    gap, gutter = layout["gap"], layout["gutter"]
    cw, ch = layout["card_w"], layout["card_h"]
    for p in range(pages):
        canvas = Image.new("RGB", (W, H), "white")
        canvas.paste(original, (0, hh))
        d = ImageDraw.Draw(canvas)
        title = f'{method.upper()} | {o.display_scale:g}x display at native-frame scale | page {p+1}/{pages}'
        d.text((12, 9), title, fill="black", font=fit_font(title, W-24, 24))
        note = "Predicted symptomatic area: >50% red; <=50% black. Original frame at left; magnified disease overlays at right."
        d.text((12, 45), note, fill="black", font=fit_font(note, W-24, 16))
        group = positives[p*o.cards_per_page:(p+1)*o.cards_per_page]
        if not group:
            msg = "No additional disease-positive leaves in this frame." if p else ("No leaf detected; disease not assessed." if not leaves else "No disease predicted in the processed leaves.")
            d.text((original.width+gutter+gap, hh+24), msg, fill="black",
                   font=fit_font(msg, W-original.width-gutter-2*gap, 20))
        slots = []
        for j, leaf in enumerate(group):
            x1, y1, x2, y2 = leaf["crop_xyxy"]
            cx = original.width+gutter+gap+(j % layout["cols"])*(cw+gap)
            cy = hh+gap+(j // layout["cols"])*(ch+gap)
            sx, sy = (x1+x2)//2, hh+(y1+y2)//2
            col = id_colour(int(leaf["leaf_id"]))
            d.rectangle((x1, hh+y1, x2-1, hh+y2-1), outline=col, width=2)
            d.line([(sx, sy), (original.width+gutter//2, sy),
                    (original.width+gutter//2, cy+ch//2), (cx, cy+ch//2)], fill=col, width=2)
            source_label = f'L{int(leaf["leaf_id"]):03d}'
            sf = fit_font(source_label, max(12, original.width-4), 15)
            bb = d.textbbox((0, 0), source_label, font=sf)
            lx = min(x1, max(0, original.width-bb[2]-8))
            ly = max(hh, hh+y1-(bb[3]-bb[1]+8))
            d.rectangle((lx, ly, lx+bb[2]+8, ly+bb[3]-bb[1]+8), fill="white")
            d.text((lx+4, ly+4-bb[1]), source_label, font=sf, fill=col)
            slots.append((leaf, cx, cy, col))
        # Draw all cutout cards after connectors, so connectors cannot cross cards.
        for leaf, cx, cy, col in slots:
            x1, y1, x2, y2 = leaf["crop_xyxy"]
            card = Image.new("RGB", (cw, ch), (248, 248, 248))
            cd = ImageDraw.Draw(card)
            txt = label(leaf)
            cd.text((10, 8), txt, font=fit_font(txt, cw-20, 23),
                    fill=severity_colour(float(leaf["predicted_severity_percent"])))
            patch = overlay_cutout(frame, leaf, method, o.display_scale)
            px, py = (cw-patch.width)//2, 47+(layout["image_h"]-patch.height)//2
            # NO fit/contain call: retain the requested magnified pixel dimensions.
            card.paste(patch, (px, py), patch.getchannel("A"))
            foot = f'{patch.width} x {patch.height} px | shown {o.display_scale:g}x native'
            cd.text((10, ch-29), foot, font=fit_font(foot, cw-20, 16), fill="black")
            cd.rectangle((0, 0, cw-1, ch-1), outline=col, width=2)
            canvas.paste(card, (cx, cy))
            cards.append({"leaf_id": int(leaf["leaf_id"]), "page": p+1,
                          "display_size_wh": list(patch.size),
                          "patch_xyxy": [cx+px, cy+py, cx+px+patch.width, cy+py+patch.height],
                          "source_crop_xyxy": [x1, y1, x2, y2]})
        filename = "final_external_magnified.png" if p == 0 else f"final_external_magnified_page_{p+1:03d}.png"
        canvas.save(result_dir / filename)
        out_files.append(filename)
    return {"pages": out_files, "canvas_size_wh": [W, H], "cards": cards,
            "display_scale_requested": o.display_scale, "positive_leaves": len(positives),
            "note": "Saved PNG uses the requested magnified dimensions without fitting cards. Image viewers may visually fit the complete PNG to a screen; inspect at 100% for pixel scale."}


def save_magnified_views(frame_dir: str | Path, method: str, *, display_scale: float = 4.0,
                         cards_per_page: int = 4, individual_views: bool = True,
                         max_canvas_pixels: int = 60_000_000) -> dict:
    """Optional per-frame hook for finish_frame(); no models or GPU allocation."""
    frame = Path(frame_dir).expanduser().resolve()
    o = DisplayOptions(display_scale, cards_per_page, individual_views, max_canvas_pixels)
    validate_options(o)
    original = rgb(frame / "00_input.png")
    leaves = load_leaves(frame, method, original.size)
    # Check layout before saving output, to catch unexpectedly huge canvases.
    layout = external_layout(original.size, leaves, o)
    report = {"method": method,
              "inplace": render_inplace(frame, method, leaves, o),
              "external": render_external(frame, method, leaves, o, layout)}
    write_json(frame / method / "magnified_display.json", report)
    return report


def video_view(image: Image.Image, width: int) -> Image.Image:
    factor = 1 if width == 0 else min(1, width/image.width)
    shape = max(1, rounded(image.width*factor)), max(1, rounded(image.height*factor))
    scaled = image if image.size == shape else image.resize(shape, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (shape[0]+shape[0]%2, shape[1]+shape[1]%2), "white")
    canvas.paste(scaled, (0, 0))
    return canvas


def encode_videos(root: Path, frames: list[dict], outputs: dict, job: dict, o: DisplayOptions) -> list[dict]:
    import cv2
    if job.get("input", {}).get("kind") != "video":
        raise ValueError("--write-videos requires a completed video run")
    fps = float(job["input"]["output_fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid saved video output FPS")
    destination = root / "videos"
    destination.mkdir(exist_ok=True)
    reports = []
    for method, info in outputs.items():
        streams = [("inplace_magnified", "final_inplace_magnified.png")]
        for p, name in enumerate(info["frames"][0]["external"]["pages"], 1):
            suffix = "" if p == 1 else f"_page_{p:03d}"
            streams.append(("external_magnified"+suffix, name))
        for stream, filename in streams:
            initial_path = child(root, frames[0]["directory"]) / method / filename
            full_size = rgb(initial_path).size
            first = video_view(rgb(initial_path), o.video_width)
            path = destination / f"{method}_{stream}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, first.size)
            if not writer.isOpened():
                raise RuntimeError("OpenCV MP4 encoder unavailable; saved PNGs remain available")
            try:
                for frame in frames:
                    view = video_view(rgb(child(root, frame["directory"]) / method / filename), o.video_width)
                    if view.size != first.size:
                        raise ValueError("Video display dimensions changed between frames")
                    writer.write(cv2.cvtColor(np.asarray(view), cv2.COLOR_RGB2BGR))
            finally:
                writer.release()
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"No video written: {path}")
            reports.append({"file": str(path.relative_to(root)), "fps": fps, "frames": len(frames),
                            "video_size_wh": list(first.size), "source_png_size_wh": list(full_size),
                            "note": "Video may be downscaled for presentation; PNGs retain requested magnification. Audio not copied; IDs are frame-local."})
    return reports


def render_saved_results(results: str | Path, *, display_scale: float = 4.0,
                         cards_per_page: int = 4, individual_views: bool = True,
                         max_canvas_pixels: int = 60_000_000, write_videos: bool = False,
                         video_width: int = 1920) -> dict:
    """Point to a completed run root (job.json), or to one saved frame directory."""
    root = Path(results).expanduser().resolve()
    o = DisplayOptions(display_scale, cards_per_page, individual_views, max_canvas_pixels, video_width)
    validate_options(o)
    if (root / "job.json").is_file():
        job = read_json(root / "job.json")
        frames = job["frames"]
        selected = job["options"]["method"]
        methods = ["bicubic", "resshift"] if selected == "both" else [selected]
    elif (root / "00_input.png").is_file():
        job = {"input": {"kind": "image"}}
        frames = [{"directory": ".", "frame_index": 0, "timestamp_seconds": None}]
        methods = [name for name in ("bicubic", "resshift") if (root / name / "leaf_results.json").is_file()]
    else:
        raise FileNotFoundError("Supply the completed result folder containing job.json, or a saved frame folder containing 00_input.png")
    if not frames or not methods:
        raise ValueError("No completed frames/methods to render")
    if write_videos and job.get("input", {}).get("kind") != "video":
        raise ValueError("--write-videos is for video-result folders only")
    outputs = {}
    for method in methods:
        data = []
        all_leaves = []
        frame_size = None
        for f in frames:
            frame = child(root, f["directory"])
            with Image.open(frame / "00_input.png") as im:
                size = im.size
            if frame_size is not None and frame_size != size:
                raise ValueError("Input frame dimensions change; render these frame folders individually")
            frame_size = size
            leaves = load_leaves(frame, method, size)
            data.append((frame, leaves))
            all_leaves.extend(leaves)
        layout = external_layout(frame_size, all_leaves, o)
        page_count = max(1, max(math.ceil(len(positive_leaves(x))/o.cards_per_page) for _, x in data))
        reports = []
        for frame, leaves in data:
            report = {"frame_directory": str(frame.relative_to(root)), "method": method,
                      "inplace": render_inplace(frame, method, leaves, o),
                      "external": render_external(frame, method, leaves, o, layout, page_count)}
            write_json(frame / method / "magnified_display.json", report)
            reports.append(report)
            print(f'{frame.name}/{method}: saved enlarged in-place and external views ({o.display_scale:g}x)', flush=True)
        outputs[method] = {"frames": reports}
    videos = encode_videos(root, frames, outputs, job, o) if write_videos else []
    summary = {"status": "complete", "results": str(root), "display_scale": display_scale,
               "severity_colour_rule": ">50 red; <=50 black", "methods": outputs,
               "videos": videos, "inference_repeated": False,
               "warning": "In-place magnified views intentionally distort scene geometry; evaluate localization and severity using original grids and saved masks, not these composites."}
    write_json(root / "display_fix_summary.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--display-scale", type=float, default=4.0)
    p.add_argument("--cards-per-page", type=int, default=4)
    p.add_argument("--no-individual", action="store_true", help="Skip per-leaf in-place PNGs")
    p.add_argument("--max-canvas-pixels", type=int, default=60_000_000)
    p.add_argument("--write-videos", action="store_true")
    p.add_argument("--video-width", type=int, default=1920, help="Video output width cap; 0 preserves PNG width. Does not resize PNGs.")
    a = p.parse_args()
    render_saved_results(a.results, display_scale=a.display_scale,
                         cards_per_page=a.cards_per_page, individual_views=not a.no_individual,
                         max_canvas_pixels=a.max_canvas_pixels, write_videos=a.write_videos,
                         video_width=a.video_width)


if __name__ == "__main__":
    main()
