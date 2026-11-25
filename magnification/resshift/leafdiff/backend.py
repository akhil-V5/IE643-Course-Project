from __future__ import annotations
import importlib
import sys
from contextlib import nullcontext
from pathlib import Path
import torch
from omegaconf import OmegaConf
from .common import file_hash, read_json, safe_load
import torch.nn.functional as F

def instantiate(config):
    target = config["target"]
    module, name = target.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)(**dict(config.get("params", {})))


def state_dict_from(obj):
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict) or not obj or not all(isinstance(v, torch.Tensor) for v in obj.values()):
        raise ValueError("Expected a tensor state_dict. This is not a supported checkpoint format.")
    # Exact key loading after removal of a DDP prefix; no silent partial loads.
    return {k.removeprefix("module."): v for k, v in obj.items()}


class ResShiftBackend:
    def __init__(self, config: dict, checkpoint: str | None = None, device: str = "cuda"):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Preparation/tests work on CPU; actual ResShift training is intended for an NVIDIA GPU.")
        b = config["backend"]
        self.repo = Path(b["repo"]).resolve()
        if not (self.repo / "models/script_util.py").exists():
            raise FileNotFoundError("Run python scripts/bootstrap.py from the project root first")
        lock = read_json(b["lock_file"])
        for rel, expected in lock["source_sha256"].items():
            if file_hash(self.repo / rel) != expected:
                raise RuntimeError(f"Upstream code changed after bootstrap: {rel}")
        for name,meta in lock.get("weights",{}).items():
            key="autoencoder_weights" if name.startswith("autoencoder") else "diffusion_weights"
            if file_hash(b[key])!=meta["sha256"]:
                raise RuntimeError(f"Official weight file changed since bootstrap: {b[key]}")
        self.lock = lock
        sys.path.insert(0, str(self.repo))
        # Fail on another project's top-level models/ldm modules already being imported.
        for name in ("models", "ldm"):
            module = sys.modules.get(name)
            if module is not None:
                paths = list(getattr(module, "__path__", []))
                if paths and not any(Path(p).resolve().is_relative_to(self.repo) for p in paths):
                    raise RuntimeError(f"Conflicting Python module {name}; run this script in a clean process")
        cfg = OmegaConf.to_container(OmegaConf.load(b["config"]), resolve=True)
        if cfg["diffusion"]["params"]["predict_type"] != "xstart" or cfg["diffusion"]["params"]["steps"] != 15:
            raise ValueError("This wrapper expects the v1 xstart 15-step configuration, not v3 or another task")
        if cfg["diffusion"]["params"]["sf"] != 4 or not cfg["model"]["params"]["cond_lq"]:
            raise ValueError("Expected x4 low-resolution-conditioned configuration")
        self.diffusion = instantiate(cfg["diffusion"])
        self.model = instantiate(cfg["model"]).to(self.device)
        selected = checkpoint or b["diffusion_weights"]
        self.model.load_state_dict(state_dict_from(safe_load(selected)), strict=True)
        self.ae = instantiate(cfg["autoencoder"]).to(self.device).float().eval()
        self.ae.load_state_dict(state_dict_from(safe_load(b["autoencoder_weights"])), strict=True)
        self.ae.requires_grad_(False)
        self.cfg = cfg

    def autocast(self, enabled: bool = True):
        return torch.autocast("cuda", dtype=torch.float16) if enabled and self.device.type == "cuda" else nullcontext()

    def loss_prediction(self, hr, lr, timestep):
        # The native API returns (terms, noisy_latent, predicted_clean_latent).
        # Native terms contains "mse", NOT necessarily a "loss" key.
        result = self.diffusion.training_losses(self.model, hr, lr, timestep,
                    first_stage_model=self.ae, model_kwargs={"lq": lr})
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError("Unexpected upstream training_losses API; inspect the recorded backend version")
        terms, _, predicted_latent = result
        if "mse" not in terms:
            raise RuntimeError("Missing native ResShift mse term")
        if torch.is_grad_enabled() and not predicted_latent.requires_grad:
            raise RuntimeError("Detached clean prediction: auxiliary lesion loss cannot train the denoiser")
        return terms["mse"].float(), predicted_latent

    def decode_with_grad(self, latent: torch.Tensor, checkpoint: bool = True):
        # Decoder weights are frozen, but the decode operation must NOT use no_grad.
        # Keep VQ distance/argmin in float32 to avoid half-precision overflow.
        def decode(z):
            with torch.autocast(self.device.type, enabled=False):
                return self.ae.decode(z.float() / float(self.diffusion.scale_factor))
        if checkpoint and torch.is_grad_enabled() and latent.requires_grad:
            from torch.utils.checkpoint import checkpoint as ckpt
            return ckpt(decode, latent, use_reentrant=False)
        return decode(latent)

    @torch.no_grad()
    def sample(self, lr: torch.Tensor, amp: bool = True):
        """Reconstruct the entire LR tensor; no tiling or extra downsampling."""
        if (lr.ndim != 4 or lr.shape[1] != 3 or lr.shape[0] < 1
                or min(lr.shape[-2:]) < 1 or not lr.is_floating_point()):
            raise ValueError("Expected a floating-point N,3,H,W tensor")

        h, w = lr.shape[-2:]
        pad_h, pad_w = (-h) % 64, (-w) % 64

        if pad_h or pad_w:
            # Border padding only; the image is never split into patches.
            padding_mode = (
                "reflect" if pad_h < h and pad_w < w else "replicate"
            )
            lr = F.pad(
                lr, (0, pad_w, 0, pad_h), mode=padding_mode
            )

        self.model.eval()
        final = None

        with self.autocast(amp):
            for state in self.diffusion.p_sample_loop_progressive(
                y=lr,
                model=self.model,
                first_stage_model=self.ae,
                model_kwargs={"lq": lr},
                clip_denoised=False,
                noise_repeat=False,
                progress=False,
                device=self.device,
            ):
                final = state["sample"]

        if final is None:
            raise RuntimeError("Diffusion sampler produced no states")

        restored = self.decode_with_grad(final, checkpoint=False)

        expected = (
            lr.shape[0], 3, lr.shape[-2] * 4, lr.shape[-1] * 4
        )
        if tuple(restored.shape) != expected:
            raise RuntimeError(
                f"Output shape {tuple(restored.shape)} != {expected}"
            )

        # Remove temporary padding: return exactly x4 the original LR size.
        restored = restored[:, :, :h * 4, :w * 4]

        if not torch.isfinite(restored).all():
            raise RuntimeError("ResShift produced non-finite RGB values")

        return restored.clamp(-1, 1)

    @torch.no_grad()
    def codec_roundtrip(self, hr: torch.Tensor):
        z = self.diffusion.encode_first_stage(hr, self.ae, up_sample=False)
        return self.decode_with_grad(z, checkpoint=False).clamp(-1, 1)
