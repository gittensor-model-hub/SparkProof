"""Verify SparkProof dataset bundles (OpenRouter xhigh + Blackwell GPU CC)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sparkproof.hashing import sample_leaf_hash, sha256_file, verified_sample_leaf_hash
from sparkproof.manifest import build_manifest, build_manifest_v2
from sparkproof.merkle import merkle_root
from sparkproof.gateways import ALLOWED_GATEWAYS
from sparkproof.policy import (
    ALLOWED_MODELS,
    REQUIRED_REASONING_EFFORT,
    allowed_teachers_manifest,
    validate_gateway_trajectory,
    validate_provider_model,
)
from sparkproof.teacher_request import manifest_generation_config, verify_request_sha256


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _load_trajectories(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def verify_manifest_policy(manifest: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    version = manifest.get("version")
    if version not in {"sparkproof-1", "sparkproof-2"}:
        issues.append(f"unsupported manifest version: {version!r}")
        return issues
    if manifest.get("gateway") not in ALLOWED_GATEWAYS:
        issues.append(f"gateway must be one of {sorted(ALLOWED_GATEWAYS)!r}, got {manifest.get('gateway')!r}")
    allowed = manifest.get("allowed_teachers") or []
    expected = allowed_teachers_manifest(manifest.get("gateway", "openrouter"))
    if allowed != expected:
        issues.append(f"allowed_teachers mismatch: {allowed!r}")
    if version == "sparkproof-2":
        if manifest.get("dataset_kind") != "triton-3.7.1-blackwell":
            issues.append(f"unexpected dataset_kind: {manifest.get('dataset_kind')!r}")
        gpu = manifest.get("gpu_profile") or {}
        if gpu.get("family") != "blackwell":
            issues.append("gpu_profile.family must be blackwell")
    issues.extend(verify_generation_config(manifest))
    return issues


def verify_generation_config(manifest: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    try:
        gen = manifest_generation_config(manifest)
    except ValueError:
        issues.append("missing generation_config in manifest")
        return issues
    if gen.get("reasoning_effort") != REQUIRED_REASONING_EFFORT:
        issues.append(
            f"generation_config.reasoning_effort must be {REQUIRED_REASONING_EFFORT!r}, "
            f"got {gen.get('reasoning_effort')!r}"
        )
    for teacher in manifest.get("allowed_teachers") or []:
        if teacher.get("reasoning_effort") != REQUIRED_REASONING_EFFORT:
            issues.append(
                f"allowed_teacher {teacher.get('provider')!r} missing reasoning_effort={REQUIRED_REASONING_EFFORT!r}"
            )
    return issues


def verify_trajectory_request_hashes(
    trajectories: list[dict[str, Any]], manifest: dict[str, Any]
) -> list[str]:
    issues: list[str] = []
    try:
        gen = manifest_generation_config(manifest)
    except ValueError:
        issues.append("missing generation_config in manifest")
        return issues
    for i, record in enumerate(trajectories):
        try:
            verify_request_sha256(record, gen)
        except ValueError as e:
            issues.append(f"trajectory[{i}]: {e}")
    return issues


def verify_trajectories_v2(trajectories: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if len(trajectories) != manifest.get("sample_count"):
        issues.append(
            f"sample_count mismatch: manifest={manifest.get('sample_count')} trajectories={len(trajectories)}"
        )

    leaves = []
    for i, record in enumerate(trajectories):
        try:
            validate_provider_model(record["provider"], record["model"])
            validate_gateway_trajectory(record)
            validation = record.get("sparkproof_validation")
            if not validation or not validation.get("passed"):
                issues.append(f"trajectory[{i}]: missing sparkproof_validation.passed")
            leaves.append(verified_sample_leaf_hash(record))
        except ValueError as e:
            issues.append(f"trajectory[{i}]: {e}")

    root = merkle_root(leaves)
    if root != manifest.get("merkle_root"):
        issues.append(f"merkle_root mismatch: expected {manifest.get('merkle_root')} got {root}")
    return issues


def verify_gpu_attestation(bundle_dir: Path, manifest: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    path = bundle_dir / "gpu_attestation.json"
    if not path.exists():
        issues.append("missing gpu_attestation.json (required for sparkproof-2)")
        return issues
    att = json.loads(path.read_text())
    if not att.get("passed"):
        issues.append("gpu_attestation.passed is false")
    token_sha = att.get("token_sha256") or ""
    if manifest.get("attestation_hash") and token_sha != manifest.get("attestation_hash"):
        issues.append("manifest.attestation_hash does not match gpu_attestation.token_sha256")
    gpu = att.get("gpu_profile") or {}
    manifest_gpu = manifest.get("gpu_profile") or {}
    for key in ("family", "profile", "name"):
        if gpu.get(key) != manifest_gpu.get(key):
            issues.append(f"gpu_profile.{key} mismatch between manifest and gpu_attestation")
    return issues


def verify_trajectories(trajectories: list[dict[str, Any]], manifest: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if len(trajectories) != manifest.get("sample_count"):
        issues.append(
            f"sample_count mismatch: manifest={manifest.get('sample_count')} trajectories={len(trajectories)}"
        )

    leaves = []
    for i, record in enumerate(trajectories):
        try:
            validate_provider_model(record["provider"], record["model"])
            validate_gateway_trajectory(record)
        except ValueError as e:
            issues.append(f"trajectory[{i}]: {e}")
        if record["provider"] not in ALLOWED_MODELS:
            issues.append(f"trajectory[{i}]: unsupported provider {record['provider']!r}")
        leaves.append(sample_leaf_hash(record))

    root = merkle_root(leaves)
    if root != manifest.get("merkle_root"):
        issues.append(f"merkle_root mismatch: expected {manifest.get('merkle_root')} got {root}")
    return issues


def verify_bundle(bundle_dir: Path, *, require_gpu_attestation: bool = True) -> dict[str, Any]:
    manifest_path = bundle_dir / "manifest.json"
    trajectories_path = bundle_dir / "trajectories.jsonl"
    prompts_path = bundle_dir / "prompts.jsonl"

    if not manifest_path.exists() or not trajectories_path.exists():
        raise FileNotFoundError("bundle must contain manifest.json and trajectories.jsonl")

    manifest = _load_json(manifest_path)
    trajectories = _load_trajectories(trajectories_path)
    issues: list[str] = []
    issues.extend(verify_manifest_policy(manifest))
    issues.extend(verify_trajectory_request_hashes(trajectories, manifest))

    if manifest.get("version") == "sparkproof-2":
        issues.extend(verify_trajectories_v2(trajectories, manifest))
        if require_gpu_attestation:
            issues.extend(verify_gpu_attestation(bundle_dir, manifest))
        validation_report = bundle_dir / "validation_report.jsonl"
        if not validation_report.exists():
            issues.append("missing validation_report.jsonl")
        if not (bundle_dir / "trajectories_raw.jsonl").exists():
            issues.append("missing trajectories_raw.jsonl archive")
    else:
        issues.extend(verify_trajectories(trajectories, manifest))
        if require_gpu_attestation:
            issues.append("production bundles must be sparkproof-2 (run sparkproof-prove on Blackwell)")

    try:
        gen_config = manifest_generation_config(manifest)
        gateway_name = manifest.get("gateway", "openrouter")
        if manifest.get("version") == "sparkproof-2":
            recomputed = build_manifest_v2(
                trajectories,
                prompts_sha256=manifest["prompts_sha256"],
                gpu_profile=manifest["gpu_profile"],
                raw_sample_count=manifest["raw_sample_count"],
                benchmark_enabled=bool((manifest.get("validation") or {}).get("benchmark_enabled")),
                openrouter_generation_config=gen_config,
                gateway=gateway_name,
                attestation_hash=manifest.get("attestation_hash"),
            )
            stable_keys = (
                "version",
                "generator_version",
                "dataset_kind",
                "gateway",
                "allowed_teachers",
                "openrouter_generation_config",
                "sample_count",
                "raw_sample_count",
                "merkle_root",
                "prompts_sha256",
                "gpu_profile",
                "validation",
            )
        else:
            recomputed = build_manifest(
                trajectories,
                prompts_sha256=manifest["prompts_sha256"],
                openrouter_generation_config=gen_config,
                gateway=gateway_name,
            )
            stable_keys = (
                "version",
                "generator_version",
                "gateway",
                "allowed_teachers",
                "openrouter_generation_config",
                "sample_count",
                "merkle_root",
                "prompts_sha256",
            )
        recomputed_dict = recomputed.to_dict()
        for key in stable_keys:
            if recomputed_dict.get(key) != manifest.get(key):
                issues.append(f"manifest.{key} mismatch on recomputation")
    except ValueError as e:
        issues.append(str(e))

    if prompts_path.exists():
        if sha256_file(str(prompts_path)) != manifest.get("prompts_sha256"):
            issues.append("prompts.jsonl sha256 does not match manifest.prompts_sha256")

    return {
        "verified": len(issues) == 0,
        "issues": issues,
        "sample_count": len(trajectories),
        "merkle_root": manifest.get("merkle_root"),
        "gateway": manifest.get("gateway"),
        "manifest_version": manifest.get("version"),
        "gpu_attested": (bundle_dir / "gpu_attestation.json").exists(),
    }
