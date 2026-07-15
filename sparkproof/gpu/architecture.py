"""GPU architecture detection and allow-listing for SparkProof dataset generation.

SparkProof datasets may be generated/validated on Blackwell or Hopper H100/H200
confidential-computing GPUs. This module is the single source of truth for which
architectures are accepted, how they map to a CUDA device, and how they're
labeled in generated prompts — so prompt text always matches the hardware that
actually validated the kernel.
"""

from __future__ import annotations

from typing import Any

from sparkproof.blackwell.gpu import (
    BLACKWELL_SM_MAJOR,
    WORKSTATION_PROFILE,
    blackwell_profile,
)

ARCH_BLACKWELL = "blackwell"
ARCH_HOPPER_H100 = "hopper-h100"
ARCH_HOPPER_H200 = "hopper-h200"
SUPPORTED_ARCHITECTURES = frozenset({ARCH_BLACKWELL, ARCH_HOPPER_H100, ARCH_HOPPER_H200})

HOPPER_SM_MAJOR = 9

# Prompt-facing label per architecture — what teacher prompts say the kernel targets.
SM_LABEL: dict[str, str] = {
    ARCH_BLACKWELL: "Blackwell SM12x",
    ARCH_HOPPER_H100: "Hopper SM90",
    ARCH_HOPPER_H200: "Hopper SM90",
}

# Blackwell introduced native FP4 tensor cores; Hopper's Transformer Engine tops
# out at FP8 — fp4 prompt/API units are excluded when targeting Hopper.
FP4_SUPPORTED: dict[str, bool] = {
    ARCH_BLACKWELL: True,
    ARCH_HOPPER_H100: False,
    ARCH_HOPPER_H200: False,
}

# H100 (80GB) and H200 (141GB) are both SM 9.0 — torch exposes no separate
# compute-capability minor version for them, so memory size (with device name
# as a tie-breaker) is what distinguishes the two.
_H200_MIN_MEMORY_GB = 120.0


def sm_label(gpu_architecture: str) -> str:
    """Prompt-facing GPU/SM label for an architecture, defaulting to Blackwell."""
    return SM_LABEL.get(gpu_architecture, SM_LABEL[ARCH_BLACKWELL])


def fp4_supported(gpu_architecture: str) -> bool:
    return FP4_SUPPORTED.get(gpu_architecture, False)


def detect_gpu_architecture(*, name: str, major: int, minor: int, total_memory_gb: float) -> str:
    """Map a detected CUDA device to a supported architecture, or raise."""
    if major in BLACKWELL_SM_MAJOR:
        return ARCH_BLACKWELL
    if major == HOPPER_SM_MAJOR:
        if total_memory_gb >= _H200_MIN_MEMORY_GB or "h200" in name.lower():
            return ARCH_HOPPER_H200
        return ARCH_HOPPER_H100
    raise RuntimeError(
        f"GPU {name} (SM {major}.{minor}) is not a supported architecture — "
        f"SparkProof datasets require one of {sorted(SUPPORTED_ARCHITECTURES)!r} "
        "(Blackwell RTX PRO 6000 / B100 / B200, or Hopper H100 / H200)"
    )


def require_supported_gpu(device_index: int = 0) -> dict[str, Any]:
    """Fail unless the selected CUDA device is Blackwell or Hopper H100/H200.

    Returns a gpu_profile dict carrying both the legacy `family` key
    ("blackwell" or "hopper", for existing corroboration checks) and the new
    `gpu_architecture` key (the specific supported value) that manifests and
    prompts key off of.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required — SparkProof Triton datasets must be validated on "
            "Blackwell or Hopper H100/H200 hardware"
        )
    if device_index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {device_index} not found")

    props = torch.cuda.get_device_properties(device_index)
    major, minor = props.major, props.minor
    total_memory_gb = round(props.total_memory / (1024**3), 2)
    gpu_architecture = detect_gpu_architecture(
        name=props.name, major=major, minor=minor, total_memory_gb=total_memory_gb
    )

    profile: dict[str, Any] = {
        "family": "blackwell" if gpu_architecture == ARCH_BLACKWELL else "hopper",
        "gpu_architecture": gpu_architecture,
        "device_index": device_index,
        "name": props.name,
        "capability": [major, minor],
        "total_memory_gb": total_memory_gb,
    }
    if gpu_architecture == ARCH_BLACKWELL:
        profile["profile"] = blackwell_profile()
        if profile["profile"] == WORKSTATION_PROFILE and major != 12:
            raise RuntimeError(
                f"SPARKPROOF_BLACKWELL_PROFILE=workstation requires SM12x (got SM{major}.{minor} on {props.name})"
            )
    return profile
