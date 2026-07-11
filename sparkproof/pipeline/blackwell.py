"""Prove teacher trajectories on Blackwell: compile, execute, optional benchmark, GPU CC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sparkproof.blackwell.gpu import require_blackwell_gpu
from sparkproof.gpu.attestation import GpuAttestationResult, attest_blackwell_gpu
from sparkproof.hashing import canonical_json_bytes, dataset_attestation_nonce, sha256_file, sha256_hex
from sparkproof.manifest import build_manifest_v2
from sparkproof.triton.validator import TritonKernelValidator


def _load_trajectories(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _has_validation_stage(records: list[dict[str, Any]], stage: str) -> bool:
    return any(
        stage in ((record.get("sparkproof_validation") or {}).get("stages") or {})
        for record in records
    )


def prove_blackwell_bundle(
    bundle_dir: Path,
    *,
    gpu_index: int = 0,
    benchmark: bool = False,
    strict_validate: bool = False,
    capture_ir: bool = False,
    attest_gpu: bool = True,
    min_pass_rate: float = 0.0,
) -> dict[str, Any]:
    """Validate trajectories on Blackwell, filter to passing samples, re-seal manifest.

    Expects `trajectories.jsonl` (raw teacher outputs). Writes:
      - trajectories_raw.jsonl (archive)
      - trajectories.jsonl (verified-only)
      - validation_report.jsonl
      - gpu_attestation.json (when attest_gpu=True)
      - manifest.json (sparkproof-2)
    """
    bundle_dir = Path(bundle_dir)
    trajectories_path = bundle_dir / "trajectories.jsonl"
    manifest_path = bundle_dir / "manifest.json"
    prompts_path = bundle_dir / "prompts.jsonl"

    if not trajectories_path.exists():
        raise FileNotFoundError(f"missing {trajectories_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")

    prior_manifest = json.loads(manifest_path.read_text())
    prompts_sha256 = prior_manifest.get("prompts_sha256")
    if prompts_path.exists() and not prompts_sha256:
        prompts_sha256 = sha256_file(str(prompts_path))

    gpu_profile = require_blackwell_gpu(gpu_index)

    raw = _load_trajectories(trajectories_path)
    _write_jsonl(bundle_dir / "trajectories_raw.jsonl", raw)
    # Never silently downgrade evidence already required during generation.
    strict_validate = strict_validate or _has_validation_stage(raw, "anti_cheat")
    capture_ir = capture_ir or _has_validation_stage(raw, "ir_artifacts")

    gpu_attestation: GpuAttestationResult | None = None
    if attest_gpu:
        # Bind this attested GPU session to exactly the raw trajectory content about
        # to be validated (hashed canonically, since trajectories_raw.jsonl is a
        # re-serialized copy of `raw` and won't be byte-identical to it), so the
        # archive can't be swapped afterward without invalidating the nonce match
        # checked in verify_gpu_attestation.
        nonce = dataset_attestation_nonce(prompts_sha256 or "", sha256_hex(canonical_json_bytes(raw)))
        gpu_attestation = attest_blackwell_gpu(gpu_profile=gpu_profile, nonce=nonce)

    validator = TritonKernelValidator(gpu_index=gpu_index)
    validation_reports: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []

    for i, record in enumerate(raw):
        validation = validator.validate_response(
            record["response"],
            run_benchmark=benchmark,
            strict=strict_validate,
            capture_ir=capture_ir,
        )
        validation_reports.append(
            {
                "index": i,
                "provider": record["provider"],
                "prompt_sha256": __import__("hashlib")
                .sha256(record["prompt"].encode())
                .hexdigest(),
                "validation": validation,
            }
        )
        if validation["passed"]:
            stamped = dict(record)
            stamped["sparkproof_validation"] = validation
            verified.append(stamped)

    pass_rate = len(verified) / len(raw) if raw else 0.0
    if pass_rate < min_pass_rate:
        raise RuntimeError(
            f"pass rate {pass_rate:.1%} below minimum {min_pass_rate:.1%} "
            f"({len(verified)}/{len(raw)} trajectories verified on Blackwell)"
        )

    attestation_hash = gpu_attestation.token_sha256() if gpu_attestation and gpu_attestation.passed else None
    gen_config = prior_manifest.get("generation_config") or prior_manifest.get("openrouter_generation_config") or {
        "reasoning_effort": "xhigh",
        "max_tokens": 2048,
        "temperature": 0.7,
    }
    manifest = build_manifest_v2(
        verified,
        prompts_sha256=prompts_sha256 or "",
        gpu_profile=gpu_profile,
        raw_sample_count=len(raw),
        benchmark_enabled=benchmark,
        openrouter_generation_config=gen_config,
        gateway=prior_manifest.get("gateway"),
        attestation_hash=attestation_hash,
    ).to_dict()

    _write_jsonl(bundle_dir / "validation_report.jsonl", validation_reports)
    _write_jsonl(bundle_dir / "trajectories.jsonl", verified)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    if gpu_attestation is not None:
        (bundle_dir / "gpu_attestation.json").write_text(
            json.dumps(gpu_attestation.to_dict(), indent=2) + "\n"
        )

    return {
        "verified_count": len(verified),
        "raw_count": len(raw),
        "pass_rate": pass_rate,
        "merkle_root": manifest["merkle_root"],
        "gpu_profile": gpu_profile,
        "gpu_attested": bool(gpu_attestation and gpu_attestation.passed),
        "benchmark": benchmark,
        "strict_validate": strict_validate,
        "capture_ir": capture_ir,
    }


def prove_blackwell_trajectories(
    trajectories: list[dict[str, Any]],
    *,
    prompts_sha256: str,
    gpu_index: int = 0,
    benchmark: bool = False,
    strict_validate: bool = False,
    capture_ir: bool = False,
    attest_gpu: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    """Validate in-memory trajectories (used right after generation)."""
    strict_validate = strict_validate or _has_validation_stage(trajectories, "anti_cheat")
    capture_ir = capture_ir or _has_validation_stage(trajectories, "ir_artifacts")
    gpu_profile = require_blackwell_gpu(gpu_index)
    gpu_attestation = None
    if attest_gpu:
        nonce = dataset_attestation_nonce(prompts_sha256, sha256_hex(canonical_json_bytes(trajectories)))
        gpu_attestation = attest_blackwell_gpu(gpu_profile=gpu_profile, nonce=nonce)

    validator = TritonKernelValidator(gpu_index=gpu_index)
    verified: list[dict[str, Any]] = []
    for record in trajectories:
        validation = validator.validate_response(
            record["response"],
            run_benchmark=benchmark,
            strict=strict_validate,
            capture_ir=capture_ir,
        )
        if validation["passed"]:
            stamped = dict(record)
            stamped["sparkproof_validation"] = validation
            verified.append(stamped)

    attestation_hash = gpu_attestation.token_sha256() if gpu_attestation and gpu_attestation.passed else None
    from sparkproof.openrouter_request import openrouter_generation_config

    manifest = build_manifest_v2(
        verified,
        prompts_sha256=prompts_sha256,
        gpu_profile=gpu_profile,
        raw_sample_count=len(trajectories),
        benchmark_enabled=benchmark,
        openrouter_generation_config=openrouter_generation_config(max_tokens=2048, temperature=0.7),
        attestation_hash=attestation_hash,
    ).to_dict()

    gpu_dict = gpu_attestation.to_dict() if gpu_attestation else None
    return verified, manifest, gpu_dict
