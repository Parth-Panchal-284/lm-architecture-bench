#!/usr/bin/env python3
from __future__ import annotations

import importlib
import platform
import sys


def version(name: str) -> str:
    module = importlib.import_module(name)
    return getattr(module, "__version__", "unknown")


print(f"python: {sys.version.split()[0]}")
print(f"platform: {platform.platform()}")

for package in ["torch", "numpy", "yaml", "pandas", "matplotlib", "scipy"]:
    try:
        print(f"{package}: {version(package)}")
    except Exception as exc:
        print(f"{package}: ERROR ({exc})")

try:
    import torch

    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu_count: {torch.cuda.device_count()}")
        for index in range(torch.cuda.device_count()):
            print(f"gpu_{index}: {torch.cuda.get_device_name(index)}")
except Exception as exc:
    print(f"torch_device_check: ERROR ({exc})")

try:
    import nanolm

    print(f"nanolm: {nanolm.__file__}")
except Exception as exc:
    print(f"nanolm: ERROR ({exc})")
    raise SystemExit(1)
