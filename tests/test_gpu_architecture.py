"""GPU architecture detection, allow-listing, and prompt parametrization."""

import pytest

from sparkproof.gpu.architecture import (
    ARCH_BLACKWELL,
    ARCH_HOPPER_H100,
    ARCH_HOPPER_H200,
    SUPPORTED_ARCHITECTURES,
    detect_gpu_architecture,
    fp4_supported,
    sm_label,
)


def test_detects_blackwell_workstation_and_datacenter():
    assert detect_gpu_architecture(name="RTX PRO 6000 Blackwell", major=12, minor=0, total_memory_gb=96) == ARCH_BLACKWELL
    assert detect_gpu_architecture(name="NVIDIA B200", major=10, minor=0, total_memory_gb=180) == ARCH_BLACKWELL


def test_detects_hopper_h100_by_memory_size():
    arch = detect_gpu_architecture(name="NVIDIA H100 80GB HBM3", major=9, minor=0, total_memory_gb=79.1)
    assert arch == ARCH_HOPPER_H100


def test_detects_hopper_h200_by_memory_size():
    arch = detect_gpu_architecture(name="NVIDIA H200", major=9, minor=0, total_memory_gb=140.0)
    assert arch == ARCH_HOPPER_H200


def test_detects_hopper_h200_by_name_when_memory_ambiguous():
    # A partially-visible/MIG-sliced H200 might report low free memory; the
    # device name is the tie-breaker.
    arch = detect_gpu_architecture(name="NVIDIA H200 (partial)", major=9, minor=0, total_memory_gb=40.0)
    assert arch == ARCH_HOPPER_H200


@pytest.mark.parametrize(
    "name,major,minor",
    [
        ("NVIDIA A100", 8, 0),
        ("NVIDIA RTX 4090", 8, 9),
        ("NVIDIA V100", 7, 0),
    ],
)
def test_rejects_unsupported_architectures(name, major, minor):
    with pytest.raises(RuntimeError, match="not a supported architecture"):
        detect_gpu_architecture(name=name, major=major, minor=minor, total_memory_gb=40.0)


def test_supported_architectures_are_exactly_three():
    assert SUPPORTED_ARCHITECTURES == frozenset({ARCH_BLACKWELL, ARCH_HOPPER_H100, ARCH_HOPPER_H200})


def test_sm_label_per_architecture():
    assert sm_label(ARCH_BLACKWELL) == "Blackwell SM12x"
    assert sm_label(ARCH_HOPPER_H100) == "Hopper SM90"
    assert sm_label(ARCH_HOPPER_H200) == "Hopper SM90"


def test_sm_label_defaults_to_blackwell_for_unknown():
    assert sm_label("unknown-arch") == "Blackwell SM12x"


def test_fp4_supported_only_on_blackwell():
    assert fp4_supported(ARCH_BLACKWELL) is True
    assert fp4_supported(ARCH_HOPPER_H100) is False
    assert fp4_supported(ARCH_HOPPER_H200) is False
