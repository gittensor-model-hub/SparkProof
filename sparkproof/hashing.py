"""Canonical hashing helpers for manifests and samples."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_attestation_nonce(prompts_sha256: str, trajectories_sha256: str) -> str:
    """Derive a GPU-attestation nonce bound to this exact prompts+trajectories content.

    Used as the NVIDIA NRAS attestation nonce (echoed back signed as ``eat_nonce``)
    so a bundle's ``trajectories_raw.jsonl`` can't be swapped after attestation
    without invalidating the nonce match checked in ``verify_gpu_attestation``.
    """
    return sha256_hex(f"sparkproof-attest:{prompts_sha256}:{trajectories_sha256}".encode())


def sample_leaf_hash(record: dict[str, Any]) -> str:
    """Hash the fields that must not change without detection (sparkproof-1)."""
    payload = {
        "prompt": record["prompt"],
        "response": record["response"],
        "provider": record["provider"],
        "model": record["model"],
        "system": record.get("system"),
        "reasoning": record.get("reasoning"),
        "gateway": record.get("gateway"),
        "api_base": record.get("api_base"),
        "request_url": record.get("request_url"),
        "gateway_model": record.get("gateway_model") or record.get("openrouter_model"),
        "openrouter_model": record.get("openrouter_model") or record.get("gateway_model"),
        "request_sha256": record.get("request_sha256"),
        "response_sha256": record.get("response_sha256"),
    }
    return sha256_hex(canonical_json_bytes(payload))


def verified_sample_leaf_hash(record: dict[str, Any]) -> str:
    """sparkproof-2: teacher fields + Blackwell Triton validation proof."""
    validation = record.get("sparkproof_validation")
    if not validation or not validation.get("passed"):
        raise ValueError("verified_sample_leaf_hash requires sparkproof_validation.passed=true")
    payload = {
        "prompt": record["prompt"],
        "response": record["response"],
        "provider": record["provider"],
        "model": record["model"],
        "system": record.get("system"),
        "reasoning": record.get("reasoning"),
        "gateway": record.get("gateway"),
        "api_base": record.get("api_base"),
        "request_url": record.get("request_url"),
        "gateway_model": record.get("gateway_model") or record.get("openrouter_model"),
        "openrouter_model": record.get("openrouter_model") or record.get("gateway_model"),
        "request_sha256": record.get("request_sha256"),
        "response_sha256": record.get("response_sha256"),
        "sparkproof_validation": {
            "passed": True,
            "triton_version": validation["triton_version"],
            "code_sha256": validation["code_sha256"],
            "stages": validation["stages"],
            "benchmark": validation.get("benchmark"),
        },
    }
    return sha256_hex(canonical_json_bytes(payload))
