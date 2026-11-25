#!/usr/bin/env python
"""Single-GPU ResShift v1 fine-tuning with optional lesion-weighted RGB supervision."""
from __future__ import annotations
import argparse
import copy
import json
import platform
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from leafdiff.backend import ResShiftBackend
from leafdiff.common import seed_all, atomic_save, safe_load, write_json, file_hash
from leafdiff.data import PlantPairs, seed_worker
from leafdiff.evaluation import evaluate_pairs
from leafdiff.losses import disease_weighted_l1


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/plantseg.yaml")
    p.add_argument("--mode", choices=["baseline", "disease"], default="disease")
    p.add_argument("--run", required=True)
    p.add_argument("--init", help="Model weights only; starts a NEW optimization phase")
    p.add_argument("--resume", help="Resume optimizer/model/scaler from this package's last.pth")
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--accumulate", type=int)
    p.add_argument("--workers", type=int)
    p.add_argument("--val-limit", type=int)
    p.add_argument("--save-every", type=int)
    p.add_argument("--validate-every", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--degradation", choices=["bicubic", "mild"])
    p.add_argument("--trainable", choices=["all", "out"])
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--aux-weight", type=float)
    p.add_argument("--lesion-multiplier", type=float)
    return p


def main():
    args = args_parser().parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    for key in ("steps", "batch_size", "accumulate", "val_limit", "save_every", "validate_every", "lr", "trainable", "aux_weight", "lesion_multiplier"):
        value = getattr(args, key)
        if value is not None:
            cfg["train"][key] = value
    if args.workers is not None:
        cfg["data"]["workers"] = args.workers
    if args.degradation:
        cfg["data"]["degradation"] = args.degradation
    if args.no_amp:
        cfg["train"]["amp"] = False
    if args.mode == "baseline":
        cfg["train"]["aux_weight"] = 0.0
    cfg["mode"] = args.mode
    tcfg, dcfg = cfg["train"], cfg["data"]
    for key in ("steps", "batch_size", "accumulate", "save_every", "validate_every", "log_every"):
        if tcfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if tcfg["aux_weight"] < 0 or tcfg["lesion_multiplier"] < 1:
        raise ValueError("Invalid auxiliary loss settings")
    if args.init and args.resume:
        raise ValueError("Use --init OR --resume")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU. Data preparation and unit tests can run on CPU, not this training entry point.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("This runner is single GPU. Set CUDA_VISIBLE_DEVICES=0 (or another ONE device).")
    run = Path(args.run).resolve()
    if run.exists() and any(run.iterdir()) and not args.resume:
        raise ValueError(f"Nonempty run directory: {run}. Use a new run name or --resume.")
    run.mkdir(parents=True, exist_ok=True)
    seed_all(int(tcfg["seed"]))
    torch.backends.cudnn.benchmark = False
    from plantseg_seg.audit import verify_prepared
    verify_prepared(dcfg["prepared"],verify_content=True)
    trainset = PlantPairs(**{k: dcfg[k] for k in ("prepared", "hr_size", "scale", "degradation")},
                          split="train", training=True, augment=dcfg["augment"], seed=tcfg["seed"])
    valset = PlantPairs(**{k: dcfg[k] for k in ("prepared", "hr_size", "scale", "degradation")},
                        split="val", training=False, seed=tcfg["seed"], limit=tcfg["val_limit"])
    if len(trainset) < tcfg["batch_size"]:
        raise ValueError("Training set smaller than batch size")
    manifest_hash = file_hash(Path(dcfg["prepared"]) / "manifest.csv")
    metadata_hash = file_hash(Path(dcfg["prepared"]) / "metadata.json")
    backend = ResShiftBackend(cfg, checkpoint=args.resume or args.init)
    backend.model.requires_grad_(True)
    if tcfg["trainable"] == "out":
        backend.model.requires_grad_(False)
        backend.model.out.requires_grad_(True)
    params = [p for p in backend.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=bool(tcfg["amp"]), init_scale=1024.0)
    step, best = 0, float("inf")
    if args.resume:
        saved = safe_load(args.resume)
        if "optimizer" not in saved:
            raise ValueError("--resume requires last.pth, not the inference-only best.pth")
        if saved["manifest_sha256"] != manifest_hash or saved["metadata_sha256"] != metadata_hash:
            raise ValueError("Dataset/mask metadata differs from the resumed experiment")
        old = saved["config"]
        for key in ("lr", "accumulate", "batch_size", "trainable", "aux_weight", "lesion_multiplier", "amp"):
            if old["train"][key] != tcfg[key]:
                raise ValueError(f"Resume config changed: train.{key}. Use --init for a new phase.")
        if old["data"] != dcfg:
            raise ValueError("Resume data configuration changed; use --init for a new experiment")
        if saved["backend_lock"] != backend.lock:
            raise ValueError("Backend provenance differs from the resumed experiment")
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        step, best = int(saved["step"]), float(saved["best_score"])
        if "torch_rng" in saved:
            torch.set_rng_state(saved["torch_rng"])
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        del saved
        print("Resumed model/optimizer/scaler. DataLoader shuffle/crop position restarts; resume is not bitwise identical.")
    if step >= tcfg["steps"]:
        raise ValueError("Requested total --steps must exceed saved step")
    write_json(run / "config.json", cfg)
    write_json(run / "provenance.json", {"backend": backend.lock, "manifest_sha256": manifest_hash,
               "metadata_sha256": metadata_hash, "initialization": args.init or cfg["backend"]["diffusion_weights"],
               "resume": args.resume, "torch": str(torch.__version__), "python": platform.python_version(),
               "gpu": torch.cuda.get_device_name(0), "trainable_parameters": sum(p.numel() for p in params),
               "checkpoint_selection": "monitoring-val lesion RGB MAE, fallback total RGB MAE; NOT lesion Dice"})
    loader = DataLoader(trainset, batch_size=tcfg["batch_size"], shuffle=True,
                        num_workers=dcfg["workers"], pin_memory=True, drop_last=True,
                        worker_init_fn=seed_worker, persistent_workers=dcfg["workers"] > 0)
    iterator = iter(loader)

    def next_batch():
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    def checkpoint(full: bool):
        # CPU copies make checkpoints portable. No EMA is silently substituted.
        value = {"state_dict": {k: v.detach().cpu() for k, v in backend.model.state_dict().items()},
                 "step": step, "best_score": best, "config": cfg,
                 "backend_lock": backend.lock, "manifest_sha256": manifest_hash, "metadata_sha256": metadata_hash}
        if full:
            value.update(optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        return value

    print(f"Mode={args.mode}; training originals={len(trainset)}; monitor patches={len(valset)}; effective batch={tcfg['batch_size'] * tcfg['accumulate']}", flush=True)
    while step < tcfg["steps"]:
        backend.model.train()
        optimizer.zero_grad(set_to_none=True)
        lr_now = tcfg["lr"] * min(1.0, (step + 1) / max(1, tcfg["warmup"]))
        for group in optimizer.param_groups:
            group["lr"] = lr_now
        logs = {"native_mse": 0.0, "weighted_l1": 0.0, "total": 0.0}
        for micro in range(tcfg["accumulate"]):
            b = {k: v.to(backend.device, non_blocking=True) if torch.is_tensor(v) else v for k, v in next_batch().items()}
            ts = torch.randint(0, backend.diffusion.num_timesteps, (b["hr"].shape[0],), device=backend.device)
            with backend.autocast(tcfg["amp"]):
                native, predicted_latent = backend.loss_prediction(b["hr"], b["lr"], ts)
            aux = torch.zeros_like(native)
            if tcfg["aux_weight"] > 0:
                pred_rgb = backend.decode_with_grad(predicted_latent, checkpoint=tcfg["decoder_checkpoint"])
                aux = disease_weighted_l1(pred_rgb, b["hr"], b["lesion"], b["valid"], tcfg["lesion_multiplier"])
                if not aux.requires_grad:
                    raise RuntimeError("Auxiliary loss is detached")
            total = (native + tcfg["aux_weight"] * aux).mean()
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite loss. Check inputs; try --no-amp and a smaller learning rate.")
            scaler.scale(total / tcfg["accumulate"]).backward()
            for key, value in (("native_mse", native.mean()), ("weighted_l1", aux.mean()), ("total", total)):
                logs[key] += float(value.detach()) / tcfg["accumulate"]
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradient. Check the last batch and rerun with --no-amp.")
        if step == 0 and not any(p.grad is not None and p.grad.abs().sum() > 0 for p in params):
            raise RuntimeError("No nonzero denoiser gradient")
        if any(p.grad is not None for p in backend.ae.parameters()):
            raise RuntimeError("Frozen autoencoder unexpectedly accumulated parameter gradients")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        entry = {"step": step, "lr": lr_now, "grad_norm": float(grad_norm), **logs}
        with (run / "training.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        if step % tcfg["log_every"] == 0 or step == 1:
            print(entry, flush=True)
        if step % tcfg["validate_every"] == 0 or step == tcfg["steps"]:
            metrics, per_patch = evaluate_pairs(backend, valset, seed=tcfg["seed"], amp=tcfg["amp"],
                    preview_dir=run / "validation" / f"step_{step:06d}", preview_count=tcfg.get("preview_count",12), include_codec=False)
            from common.runtime import write_csv as write_rows,append_json
            write_rows(run / "validation" / f"step_{step:06d}" / "per_patch.csv",per_patch)
            append_json(run / "validation.jsonl",{"step":step,"patches":len(valset),**metrics})
            score = metrics["diffusion"]["lesion_mae"]
            score = metrics["diffusion"]["mae"] if score is None else score
            improved = score < best
            if improved:
                best = float(score)
                atomic_save(checkpoint(False), run / "best.pth")
            write_json(run / f"validation_{step:06d}.json", {"step": step, "monitoring_patch_protocol_not_full_image_test": True, **metrics})
            print(f"Validation step {step}: {metrics}; new best={improved}", flush=True)
        if step % tcfg["save_every"] == 0 or step == tcfg["steps"] or step % tcfg["validate_every"] == 0:
            atomic_save(checkpoint(True), run / "last.pth")
    print(f"Finished {step} updates. Checkpoints: {run}/best.pth and last.pth", flush=True)


if __name__ == "__main__":
    main()
