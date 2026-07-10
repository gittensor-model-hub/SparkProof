"""Blackwell GPU gate for SparkProof dataset validation (RTX PRO 6000 SM12x target)."""

from __future__ import annotations

import os
from typing import Any

BLACKWELL_SM_MAJOR = frozenset({10, 12})
TARGET_GPU_FAMILY = "blackwell"
WORKSTATION_PROFILE = "workstation"
DATACENTER_PROFILE = "datacenter"


def blackwell_profile() -> str:
    name = os.environ.get("SPARKPROOF_BLACKWELL_PROFILE", WORKSTATION_PROFILE).strip().lower()
    if name not in {WORKSTATION_PROFILE, DATACENTER_PROFILE}:
        raise ValueError(
            f"unknown SPARKPROOF_BLACKWELL_PROFILE={name!r}; "
            f"expected {WORKSTATION_PROFILE!r} or {DATACENTER_PROFILE!r}"
        )
    return name


def is_blackwell_capability(major: int, _minor: int) -> bool:
    return major in BLACKWELL_SM_MAJOR


def require_blackwell_gpu(device_index: int = 0) -> dict[str, Any]:
    """Fail unless the selected CUDA device is Blackwell (SM10x datacenter or SM12x workstation)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required — SparkProof Triton datasets must be validated on Blackwell hardware"
        )
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device_index} not found")

    props = torch.cuda.get_device_properties(device_index)
    major, minor = props.major, props.minor
    if not is_blackwell_capability(major, minor):
        raise RuntimeError(
            f"GPU {props.name} (SM {major}.{minor}) is not Blackwell — "
            "SparkProof production bundles require RTX PRO 6000 Blackwell (SM12x) "
            "or datacenter Blackwell (SM10x)"
        )

    profile = blackwell_profile()
    if profile == WORKSTATION_PROFILE and major != 12:
        raise RuntimeError(
            f"SPARKPROOF_BLACKWELL_PROFILE=workstation requires SM12x (got SM{major}.{minor} on {props.name})"
        )

    return {
        "family": TARGET_GPU_FAMILY,
        "profile": profile,
        "device_index": device_index,
        "name": props.name,
        "capability": [major, minor],
        "total_memory_gb": round(props.total_memory / (1024**3), 2),
    }
