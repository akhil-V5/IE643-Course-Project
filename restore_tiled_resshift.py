#!/usr/bin/env python3
"""Restore memory-bounded tiled ResShift inference in the supplied PlantSeg suite.

Run beside run.py: python restore_tiled_resshift.py --dry-run
Then:             python restore_tiled_resshift.py --apply

Standard library only. Backs up every changed existing file before writing.
Does not alter backend.py, trainers, datasets, checkpoints, or run.py.
"""
from __future__ import annotations
import argparse
import ast
from datetime import datetime
import difflib
import json
from pathlib import Path
import re
import shutil
import sys
import uuid

TILING_SOURCE = '"""ResShift v1: sequential 64-LR-pixel tiles, CPU overlap blending.\n\nRestores the earlier suite\'s 64/16 geometry and per-tile seeds. No extra\nimage downsampling. Training and the low-level backend are not modified.\n"""\nfrom __future__ import annotations\nimport time\nimport torch\nimport torch.nn.functional as F\nfrom PIL import Image\nfrom leafdiff.common import to_tensor, to_pil\n\n\ndef positions(length: int, tile: int, stride: int) -> list[int]:\n    if length < tile or not 0 < stride <= tile:\n        raise ValueError("Invalid padded length or tile stride")\n    starts = list(range(0, length - tile + 1, stride))\n    if starts[-1] != length - tile:\n        starts.append(length - tile)\n    return starts\n\n\n@torch.no_grad()\ndef tiled_reconstruct(backend, image: Image.Image, seed: int = 42,\n                      overlap: int = 16, amp: bool = True):\n    """Return (x4 PIL RGB, truthful sampling metadata). One tile on GPU at a time."""\n    tile, scale = 64, 4\n    if not isinstance(overlap, int) or not 0 <= overlap < tile:\n        raise ValueError("overlap must be an integer from 0 to 63 LR pixels")\n    image = image.convert("RGB")\n    w, h = image.size\n    if min(w, h) < 1:\n        raise ValueError("Empty input image")\n    # Keep the complete input, output and blending arrays in system RAM.\n    x = to_tensor(image).cpu()\n    right, bottom = max(0, tile-w), max(0, tile-h)\n    if right or bottom:\n        x = F.pad(x[None], (0, right, 0, bottom), mode="replicate")[0]\n    hp, wp = x.shape[-2:]\n    accum = torch.zeros((3, hp*scale, wp*scale), dtype=torch.float32, device="cpu")\n    norm = torch.zeros((1, hp*scale, wp*scale), dtype=torch.float32, device="cpu")\n    win = torch.hann_window(tile*scale, periodic=False, dtype=torch.float32,\n                            device="cpu").clamp_min(0.001)\n    weight = (win[:, None]*win[None, :])[None]\n    cuda = torch.device(backend.device).type == "cuda"\n    if cuda:\n        torch.cuda.synchronize(backend.device)\n    started = time.perf_counter()\n    count = 0\n    for y in positions(hp, tile, tile-overlap):\n        for xx in positions(wp, tile, tile-overlap):\n            torch.manual_seed(seed+count)\n            if cuda:\n                torch.cuda.manual_seed_all(seed+count)\n            patch = x[:, y:y+tile, xx:xx+tile][None].to(backend.device)\n            sr_gpu = backend.sample(patch, amp=amp)\n            if tuple(sr_gpu.shape) != (1, 3, tile*scale, tile*scale):\n                raise RuntimeError(f"Unexpected ResShift tile shape: {tuple(sr_gpu.shape)}")\n            if not torch.isfinite(sr_gpu).all():\n                raise RuntimeError("Non-finite ResShift reconstruction tile")\n            restored = sr_gpu[0].detach().to(device="cpu", dtype=torch.float32)\n            del sr_gpu, patch\n            yy, xxx, ss = y*scale, xx*scale, tile*scale\n            accum[:, yy:yy+ss, xxx:xxx+ss] += restored*weight\n            norm[:, yy:yy+ss, xxx:xxx+ss] += weight\n            del restored\n            count += 1\n    if cuda:\n        torch.cuda.synchronize(backend.device)\n    result = (accum/norm.clamp_min(1e-12))[:, :h*scale, :w*scale]\n    if not torch.isfinite(result).all():\n        raise RuntimeError("Non-finite blended reconstruction")\n    metadata = {\n        "sampling_mode": "tiled", "sample_calls": count, "tiles": count,\n        "tile_lr": tile, "overlap_lr": overlap, "scale": scale,\n        "input_size_wh": [w, h], "output_size_wh": [w*scale, h*scale],\n        "padding_right_bottom": [right, bottom],\n        "padding_mode": "replicate" if right or bottom else "none",\n        "seed": seed, "sampling_seconds": time.perf_counter()-started,\n        "blend_device": "cpu", "tile_batch_size": 1,\n    }\n    return to_pil(result), metadata\n'

RECONSTRUCT_FUNCTION = 'def reconstruct(config,output,comparison=None,input_path=None,checkpoint=None,backend_override=None):\n    import copy\n    if bool(comparison)==bool(input_path):raise ValueError(\'Choose comparison OR input, not both\')\n    cfg=copy.deepcopy(load_yaml(config) if isinstance(config,(str,Path)) else config)\n    mode=cfg[\'inference\'][\'method\'];scale=int(cfg[\'inference\'][\'scale\'])\n    if mode==\'resshift\':\n        if int(cfg[\'inference\'].get(\'tile_lr\',64)) != 64:\n            raise ValueError(\'The restored v1 tiled wrapper uses tile_lr=64\')\n        overlap = int(cfg[\'inference\'].get(\'overlap\',16))\n        if not 0 <= overlap < 64:\n            raise ValueError(\'overlap must be 0..63 LR pixels\')\n        cfg[\'inference\'].update(sampling_mode=\'tiled\', tile_lr=64, overlap=overlap)\n    if mode not in (\'resshift\',\'bicubic\') or scale!=4:raise ValueError(\'Supported methods: x4 ResShift or bicubic\')\n    if mode==\'bicubic\' and checkpoint:raise ValueError(\'Bicubic has no checkpoint\')\n    meta=None;root=Path(comparison).resolve() if comparison else None\n    if root:\n        meta,rows=verify_cases(root)\n    else:\n        files=collect_inputs(input_path)\n        if not files:raise ValueError(\'No images found\')\n        dest=Path(output).resolve();ip=Path(input_path).resolve()\n        if ip.is_dir() and (dest==ip or dest.is_relative_to(ip)):raise ValueError(\'Output must be outside input directory; prevent reprocessing prior previews\')\n        rows=[{\'id\':hashlib.sha256(str(p).encode()).hexdigest()[:20],\'input\':str(p)} for p in files]\n    backend=None;selected=None;checkpoint_sha=None\n    if mode==\'resshift\':\n        from leafdiff.backend import ResShiftBackend\n        from leafdiff.common import safe_load\n        import torch\n        selected=checkpoint or cfg[\'backend\'][\'diffusion_weights\'];checkpoint_sha=digest(selected)\n        if checkpoint and backend_override is None:\n            saved=safe_load(checkpoint)\n            if \'config\' not in saved or \'manifest_sha256\' not in saved or int(saved.get(\'step\',0))<1:\n                raise ValueError(\'Finetuned condition requires a trained checkpoint with shared-split provenance; do not silently label pretrained weights finetuned\')\n            if meta and saved[\'manifest_sha256\']!=meta[\'source_manifest_sha256\']:\n                raise ValueError(\'ResShift was trained with a DIFFERENT source split manifest\')\n            del saved\n        backend=backend_override or ResShiftBackend(cfg,checkpoint=selected,device=cfg[\'inference\'][\'device\'])\n        if hasattr(backend,\'model\'):backend.model.eval().requires_grad_(False)\n        if hasattr(backend,\'ae\'):backend.ae.eval().requires_grad_(False)\n    out=fresh_dir(output);record=[];scores=[];preview_count=int(cfg[\'inference\'].get(\'preview_count\',12))\n    write_json(out/\'resolved_config.json\',cfg)\n    print(f\'Reconstruction: {len(rows)} images; optional visual examples={preview_count} (0=all). RGB outputs for ALL selected images are required for the next segmentation stage.\',flush=True)\n    for i,r in enumerate(rows):\n        ip=(root/r[\'lr\']) if root else Path(r[\'input\']);image=load_rgb(ip)\n        seed=(int(cfg[\'inference\'][\'seed\'])+int(hashlib.sha256(r[\'id\'].encode()).hexdigest()[:8],16))%(2**31)\n        start=time.perf_counter()\n        if mode==\'bicubic\':\n            restored=image.resize((image.width*4,image.height*4),Image.Resampling.BICUBIC)\n            sampling={\'sampling_mode\':\'bicubic\',\'sample_calls\':0}\n        else:\n            from leafdiff.tiling import tiled_reconstruct\n            restored,sampling=tiled_reconstruct(backend,image,seed=seed,\n                       overlap=int(cfg[\'inference\'][\'overlap\']),amp=bool(cfg[\'inference\'][\'amp\']))\n            if backend.device.type==\'cuda\':\n                import torch;torch.cuda.synchronize(backend.device)\n        seconds=time.perf_counter()-start\n        if restored.size!=(image.width*4,image.height*4):raise ValueError(f\'Unexpected reconstruction dimensions for {r["id"]}\')\n        if root:\n            if restored.size!=(int(r[\'padded_width\']),int(r[\'padded_height\'])):raise ValueError(\'Reconstruction is not on declared padded x4 grid\')\n            restored=restored.crop((0,0,int(r[\'width\']),int(r[\'height\'])))\n        dest=out/\'rgb\'/(r[\'id\']+\'.png\');dest.parent.mkdir(parents=True,exist_ok=True);restored.save(dest)\n        entry={\'id\':r[\'id\'],\'rgb\':str(dest.relative_to(out)),\'rgb_sha256\':digest(dest),\'width\':restored.width,\n             \'height\':restored.height,\'input\':str(ip),\'input_sha256\':digest(ip),**sampling,\'seed\':seed,\'seconds\':seconds}\n        if root:\n            original=load_rgb(root/r[\'reference\']);gt=load_labels(root/r[\'mask\']);bic=load_rgb(root/r[\'bicubic\'])\n            values=rgb_metrics(restored,original);entry.update(values);scores.append(values)\n            if preview_count==0 or i<preview_count:\n                save_ground_truth(original,gt,out,r[\'id\'])\n                panel_grid([original,image,bic,restored,disease_gray(gt)],[\'Original reference RGB\',\'Actual low-resolution RGB (preview)\',\n                           \'Bicubic RGB\',\'Reconstructed RGB: \'+mode,\'GROUND TRUTH disease mask (not prediction)\'],out/\'panels\'/(r[\'id\']+\'.png\'))\n        elif preview_count==0 or i<preview_count:\n            panel_grid([image,restored],[\'Input RGB\',\'Reconstructed RGB: \'+mode],out/\'panels\'/(r[\'id\']+\'.png\'))\n        record.append(entry);append_json(out/\'progress.jsonl\',entry)\n        print(f\'Reconstructed {i+1}/{len(rows)} {r["id"]}: {image.size} -> {restored.size}; sampling_mode={sampling["sampling_mode"]}; sample_calls={sampling["sample_calls"]}; {seconds:.3f}s\',flush=True)\n    write_csv(out/\'images.csv\',record)\n    info={\'schema\':\'plantseg_reconstruction_v3\',\'method\':mode,\'condition\':\'finetuned\' if checkpoint else (\'pretrained\' if mode==\'resshift\' else \'bicubic\'),\n       \'sampling_mode\':\'tiled\' if mode==\'resshift\' else \'bicubic\',\n       \'actual_backend\':backend_override is None,\'checkpoint\':str(Path(selected).resolve()) if selected else None,\n       \'checkpoint_sha256\':checkpoint_sha,\'images_csv_sha256\':digest(out/\'images.csv\'),\n       \'cases_sha256\':meta[\'cases_sha256\'] if meta else None,\'source_manifest_sha256\':meta[\'source_manifest_sha256\'] if meta else None,\n       \'split\':meta[\'split\'] if meta else None,\'images\':len(record),\'inference_config\':cfg[\'inference\'],\'versions\':versions(),\n       \'label_inputs\':\'None. Annotations are read for integrity checks and output scoring/visualization, never passed to the reconstruction model.\',\n       \'metrics\':aggregate_rgb(scores) if meta else None,\n       \'metric_note\':\'RGB PSNR/SSIM on original unpadded image; no GT-dependent pixel selection. Without references these metrics are unavailable.\'}\n    write_json(out/\'manifest.json\',info)\n    return out\n'

VERIFY_SR_FUNCTION = 'def verify_sr_output(folder: Path, lr_path: Path) -> tuple[Path, dict]:\n    records = rows(folder / "images.csv")\n    if len(records) != 1:\n        raise ValueError("Expected exactly ONE reconstructed image")\n    record = records[0]\n    if Path(record["input"]).resolve() != lr_path.resolve():\n        raise ValueError("Reconstruction belongs to a different input")\n    sampling_mode = record.get("sampling_mode")\n    calls = int(record.get("sample_calls", 0))\n    if sampling_mode not in {"tiled", "whole_image"} or calls < 1:\n        raise ValueError("Missing or invalid ResShift sampling metadata")\n    if sampling_mode == "whole_image" and calls != 1:\n        raise ValueError("Whole-image output must have exactly one sampler call")\n    if sampling_mode == "tiled" and int(record.get("tiles", 0)) != calls:\n        raise ValueError("Tiled output has inconsistent tile/call counts")\n    path = Path(record["rgb"])\n    return (path if path.is_absolute() else folder / path), record\n'

def replace_function(source: str, name: str, replacement: str) -> str:
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1:
        raise ValueError(f"Expected one {name} function; refusing unknown file layout")
    node = nodes[0]
    first = min([node.lineno] + [d.lineno for d in node.decorator_list])
    lines = source.splitlines(keepends=True)
    return "".join(lines[:first-1]) + replacement.rstrip() + "\n" + "".join(lines[node.end_lineno:])


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    if source.count(old) != 1:
        raise ValueError(f"Cannot locate {label}; no files have been changed")
    return source.replace(old, new, 1)


def set_tiled_yaml(source: str) -> str:
    lines = source.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if re.match(r"^inference:\s*(?:#.*)?$", line.rstrip())]
    if len(starts) != 1:
        raise ValueError("Cannot identify inference section in config")
    start = starts[0]
    end = next((i for i in range(start+1, len(lines))
                if lines[i].strip() and not lines[i].startswith((" ", "\t", "#"))), len(lines))
    section = [line for line in lines[start+1:end]
               if not re.match(r"^\s+(sampling_mode|tile_lr|overlap):", line)]
    section = ["  sampling_mode: tiled  # One 64x64 LR tile on GPU at a time.\n",
               "  tile_lr: 64\n", "  overlap: 16\n"] + section
    result = "".join(lines[:start+1]+section+lines[end:])
    result = result.replace("Whole-image inference; no tiling, extra resize, or segmentation gate.",
                            "Tiled inference; 64-pixel LR tiles with 16-pixel overlap.")
    return result


def planned_changes(root: Path) -> dict[str, str]:
    if not (root/"run.py").is_file():
        raise FileNotFoundError("Run from the project folder containing run.py")
    changes = {}
    def read(name):
        return (root/name).read_text(encoding="utf-8")
    changes["magnification/resshift/leafdiff/tiling.py"] = TILING_SOURCE
    changes["magnification/reconstruct.py"] = replace_function(
        read("magnification/reconstruct.py"), "reconstruct", RECONSTRUCT_FUNCTION)

    # Keep source/grid/hash checks, but permit truthfully recorded tiled output.
    name = "experiments/evaluate.py"
    source = read(name)
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name == "load_variant"]
    if len(nodes) != 1:
        raise ValueError("Cannot identify evaluation manifest loader")
    guards = [node for node in nodes[0].body if isinstance(node, ast.If)
              and "sampling_mode" in ast.get_source_segment(source, node.test)]
    if len(guards) > 1:
        raise ValueError("Ambiguous sampling-mode guards")
    guard = ("    if info.get('method') == 'resshift' and info.get('sampling_mode') not in {'tiled', 'whole_image'}:\n"
             "        raise ValueError('ResShift output lacks valid sampling metadata; regenerate it.')\n")
    lines = source.splitlines(keepends=True)
    if guards:
        node = guards[0]
        source = "".join(lines[:node.lineno-1]) + guard + "".join(lines[node.end_lineno:])
    else:
        schema = [node for node in nodes[0].body if isinstance(node, ast.If)
                  and "schema" in ast.get_source_segment(source, node.test)]
        if len(schema) != 1:
            raise ValueError("Cannot locate reconstruction schema check")
        pos = schema[0].end_lineno
        source = "".join(lines[:pos]) + guard + "".join(lines[pos:])
    changes[name] = source

    # The panel used to reject any tiled result; update both guard and displayed label.
    name = "visualize_one_image.py"
    if (root/name).is_file():
        source = replace_function(read(name), "verify_sr_output", VERIFY_SR_FUNCTION)
        source = replace_once(source,
            'subtitle = ("Same LR input for both methods | whole-image ResShift | no tiles | "',
            'subtitle = (f"Same LR input for both methods | ResShift: {sr_record[\'sampling_mode\']} | "',
            "panel sampling label")
        source = source.replace("shared LR -> bicubic / whole-image ResShift", "shared LR -> bicubic / tiled ResShift")
        changes[name] = source

    # Optional DINO/SAM image entry point has its own sampler caller.
    name = "drone_leaf_image.py"
    if (root/name).is_file():
        source = read(name)
        if "def restore_whole_crop(" in source:
            old_name = "restore_whole_crop"
        elif "def restore_tiled_crop(" in source:
            old_name = "restore_tiled_crop"
        else:
            raise ValueError("Cannot identify drone-image reconstruction function")
        body = ("def restore_tiled_crop(backend, image: Image.Image, seed: int, amp: bool):\n"
                '    """Reconstruct one leaf using memory-bounded x4 tiles; no extra downsampling."""\n'
                "    from leafdiff.tiling import tiled_reconstruct\n"
                "    return tiled_reconstruct(backend, image, seed=seed, overlap=16, amp=amp)\n")
        source = replace_function(source, old_name, body)
        source = source.replace("sr, record = restore_whole_crop(", "sr, record = restore_tiled_crop(")
        source = source.replace("; whole_image; sample_calls=1', flush=True)",
                                '; sampling_mode={record["sampling_mode"]}; sample_calls={record["sample_calls"]}\', flush=True)')
        source = source.replace("whole-crop ResShift x4", "tiled ResShift x4")
        source = source.replace("Whole-crop ResShift x4", "Tiled ResShift x4")
        changes[name] = source

    # Also restore the stand-alone inference CLI. Keep legacy callable functions intact.
    name = "magnification/resshift/infer.py"
    source = read(name)
    old = "sr, sampling = whole_image_reconstruct(backend, image, seed=a.seed, amp=not a.no_amp)"
    new = ("from leafdiff.tiling import tiled_reconstruct as restore_tiles\n"
           "        sr, sampling = restore_tiles(backend, image, seed=a.seed, overlap=16, amp=not a.no_amp)")
    if old in source:
        source = source.replace(old,new,1)
        source = source.replace("Whole-image diffusion reconstruction", "Tiled diffusion reconstruction")
        source = source.replace('sampling_mode=whole_image; sample_calls=1", flush=True)',
                                'sampling_mode=tiled; sample_calls={sampling[\"sample_calls\"]}", flush=True)')
        # Quotes inside f-string need to differ from surrounding double quotes.
        source = source.replace('sample_calls={sampling["sample_calls"]}', "sample_calls={sampling['sample_calls']}")
        source = source.replace('"sampling_mode": "whole_image", "preview_count": a.preview_count',
                                '"sampling_mode": "tiled", "preview_count": a.preview_count')
        source = source.replace('"sampling_mode": "whole_image",\n', '"sampling_mode": "tiled",\n') if '"sampling_mode": "whole_image", "sample_calls": 1' not in source else source
        # Manual-edit version has no preview_count in the manifest.
        source = source.replace('"sampling_mode": "whole_image", "results": records',
                                '"sampling_mode": "tiled", "results": records')
        source = source.replace("The entire low-resolution image is reconstructed in one diffusion sequence.\nThere is no tiling, overlap blending, or extra image downsampling. Temporary\nright/bottom padding is removed after decoding.",
                                "The CLI reconstructs overlapping 64x64 LR tiles and blends on CPU.\nNo extra image downsampling is performed.")
        # Cover the multiline print/manifest used in the earlier manual-edit instructions.
        main_node = next(node for node in ast.parse(source).body
                         if isinstance(node, ast.FunctionDef) and node.name == "main")
        main_text = "".join(source.splitlines(keepends=True)[main_node.lineno-1:main_node.end_lineno])
        main_text = main_text.replace('"sampling_mode": "whole_image"', '"sampling_mode": "tiled"')
        main_text = main_text.replace('"whole_image; sample_calls=1"',
                                      'f"sampling_mode=tiled; sample_calls={sampling[\'sample_calls\']}"')
        source = replace_function(source, "main", main_text)
        changes[name] = source

    for name in ("magnification/configs/resshift_pretrained.yaml", "magnification/configs/resshift_finetune.yaml"):
        changes[name] = set_tiled_yaml(read(name))
    # Validate every Python edit before any filesystem write, including optional addons.
    for name, text in changes.items():
        if name.endswith(".py"):
            ast.parse(text, filename=name, feature_version=(3, 11))
            compile(text, name, "exec")
    return {name: text for name,text in changes.items()
            if not (root/name).exists() or read(name) != text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true")
    group.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    changes = planned_changes(root)
    if not changes:
        print("Already configured for tiled inference; no files changed.")
        return
    print("Changes:")
    for name in changes:
        print("  "+name)
    if not args.apply:
        print("Dry run only. Use --apply to write these changes with a backup.")
        return
    backup = root/("backup_before_restore_tiling_"+datetime.now().strftime("%Y%m%d_%H%M%S")+"_"+uuid.uuid4().hex[:6])
    backup.mkdir()
    diff = []
    new_files = []
    for name,text in changes.items():
        dest = root/name
        before = dest.read_text(encoding="utf-8") if dest.exists() else ""
        if dest.exists():
            target = backup/name
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(dest,target)
        else:
            new_files.append(name)
        diff += list(difflib.unified_diff(before.splitlines(keepends=True),text.splitlines(keepends=True),
                                        fromfile="before/"+name,tofile="after/"+name))
    (backup/"changes.diff").write_text("".join(diff),encoding="utf-8")
    (backup/"new_files.json").write_text(json.dumps(new_files,indent=2)+"\n",encoding="utf-8")
    for name,text in changes.items():
        path = root/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text,encoding="utf-8")
    print("Backup: "+str(backup))
    print("Tiled ResShift enabled: LR tiles 64x64; overlap 16; one tile per sampler call; CPU blending.")
    print("Training, backend.py, run.py, data and checkpoints were NOT changed.")
    print("Restart your process and use a NEW output folder.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, SyntaxError) as exc:
        raise SystemExit("ERROR: "+str(exc)) from exc
