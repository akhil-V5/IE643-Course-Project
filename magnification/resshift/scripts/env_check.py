#!/usr/bin/env python
from __future__ import annotations
import importlib
import platform
import sys
import torch

print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("Torch CUDA build:", torch.version.cuda)
print("CUDA usable:", torch.cuda.is_available())
for name in ("torchvision", "numpy", "PIL", "timm", "omegaconf", "einops", "loralib", "skimage"):
    try:
        module = importlib.import_module(name)
        print(name, getattr(module, "__version__", "import OK"))
    except Exception as e:
        print(name, "FAILED:", str(e))
        sys.exit(1)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {prop.name}, VRAM {prop.total_memory/2**30:.1f} GiB")
else:
    print("Only data preparation/CPU unit tests are supported in this environment.")
