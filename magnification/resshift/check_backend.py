#!/usr/bin/env python
"""Actual GPU smoke test: official weights, auxiliary gradient, backward, and sampling."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf
from leafdiff.backend import ResShiftBackend
from leafdiff.common import seed_all, to_pil, panel, write_json
from leafdiff.data import PlantPairs
from leafdiff.losses import disease_weighted_l1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/plantseg.yaml")
    p.add_argument("--output", default="runs/backend_check")
    p.add_argument("--no-amp", action="store_true")
    a = p.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(a.config), resolve=True)
    seed_all(cfg["train"]["seed"])
    d = cfg["data"]
    ds = PlantPairs(**{k: d[k] for k in ("prepared", "hr_size", "scale", "degradation")}, split="train", training=False)
    # Select a useful smoke-test crop; this selection is not a validation/test score.
    b = ds[0]
    for i in range(min(len(ds), 30)):
        candidate = ds[i]
        if candidate["lesion"].sum() > 0:
            b = candidate
            break
    backend = ResShiftBackend(cfg)
    backend.model.train()
    x = {k: v[None].to(backend.device) for k, v in b.items() if torch.is_tensor(v)}
    ts = torch.tensor([backend.diffusion.num_timesteps - 1], device=backend.device)
    amp = not a.no_amp
    with backend.autocast(amp):
        native, latent = backend.loss_prediction(x["hr"], x["lr"], ts)
    rgb = backend.decode_with_grad(latent, checkpoint=True)
    aux = disease_weighted_l1(rgb, x["hr"], x["lesion"], x["valid"], 5.0).mean()
    dz = torch.autograd.grad(aux, latent, retain_graph=True)[0]
    if not torch.isfinite(dz).all() or dz.abs().sum() == 0:
        raise RuntimeError("Auxiliary lesion loss did not propagate a finite nonzero gradient through the frozen decoder")
    total = native.mean() + 0.1 * aux
    total.backward()
    grads = [p.grad for p in backend.model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("Denoiser gradients are missing or non-finite; try --no-amp")
    if any(p.grad is not None for p in backend.ae.parameters()):
        raise RuntimeError("Autoencoder parameters should remain frozen")
    report = {"native_loss": float(native.mean().detach()), "auxiliary_loss": float(aux.detach()),
              "auxiliary_latent_grad_abs_sum": float(dz.abs().sum()),
              "denoiser_grad_abs_sum": float(sum(g.detach().float().abs().sum() for g in grads)),
              "codec_parameters_frozen": True, "checkpoint_steps": backend.diffusion.num_timesteps,
              "gpu": torch.cuda.get_device_name(), "backend_commit": backend.lock["commit"]}
    # Release the training graph before reverse sampling to lower peak live memory.
    del grads, total, native, aux, dz, latent, rgb
    backend.model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    result = backend.sample(x["lr"], amp=amp)
    rt = backend.codec_roundtrip(x["hr"])
    if tuple(result.shape) != (1, 3, 256, 256) or not torch.isfinite(result).all():
        raise RuntimeError("Unexpected sampling output")
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    panel([to_pil(x["bicubic"]), to_pil(result), to_pil(x["hr"]), to_pil(rt)],
          ["Bicubic", "Pretrained ResShift", "Original HR target", "Frozen codec roundtrip"], out / "smoke.png")
    write_json(out / "smoke.json", report)
    print(f"GPU smoke test PASSED. Inspect {out}/smoke.png before a longer run.")
    print(report)


if __name__ == "__main__":
    main()
