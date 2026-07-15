"""Dataset manifest builder for SparkProof bundles."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

from sparkproof.gpu.architecture import ARCH_BLACKWELL
from sparkproof.hashing import canonical_json_bytes, sample_leaf_hash, sha256_hex, verified_sample_leaf_hash
from sparkproof.merkle import merkle_root
from sparkproof.policy import (
    DATASET_KIND_TRITON_BLACKWELL,
    GENERATOR_VERSION,
    TRITON_VERSION,
    allowed_teachers_manifest,
    validate_gateway_trajectory,
    validate_provider_model,
)


@dataclass(frozen=True)
class DatasetManifest:
    version: str
    generator_version: str
    gateway: str
    allowed_teachers: list[dict[str, str]]
    openrouter_generation_config: dict[str, Any]
    sample_count: int
    merkle_root: str
    prompts_sha256: str
    created_at: str
    attestation_hash: str | None = None
    sampling: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version": self.version,
            "generator_version": self.generator_version,
            "gateway": self.gateway,
            "allowed_teachers": self.allowed_teachers,
            "generation_config": self.openrouter_generation_config,
            "openrouter_generation_config": self.openrouter_generation_config,
            "sample_count": self.sample_count,
            "merkle_root": self.merkle_root,
            "prompts_sha256": self.prompts_sha256,
            "created_at": self.created_at,
        }
        if self.attestation_hash is not None:
            out["attestation_hash"] = self.attestation_hash
        if self.sampling is not None:
            out["sampling"] = self.sampling
        return out


def build_manifest(
    trajectories: list[dict[str, Any]],
    *,
    prompts_sha256: str,
    openrouter_generation_config: dict[str, Any],
    gateway: str | None = None,
    attestation_hash: str | None = None,
    sampling: dict[str, Any] | None = None,
) -> DatasetManifest:
    gateway_name = gateway or (trajectories[0].get("gateway") if trajectories else None) or "openrouter"
    for record in trajectories:
        validate_provider_model(record["provider"], record["model"])
        validate_gateway_trajectory(record)

    leaves = [sample_leaf_hash(record) for record in trajectories]
    return DatasetManifest(
        version="sparkproof-1",
        generator_version=GENERATOR_VERSION,
        gateway=gateway_name,
        allowed_teachers=allowed_teachers_manifest(gateway_name),
        openrouter_generation_config=openrouter_generation_config,
        sample_count=len(trajectories),
        merkle_root=merkle_root(leaves),
        prompts_sha256=prompts_sha256,
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        attestation_hash=attestation_hash,
        sampling=sampling,
    )


def manifest_id(manifest: dict[str, Any]) -> str:
    return sha256_hex(canonical_json_bytes(manifest))


@dataclass(frozen=True)
class BlackwellDatasetManifest:
    version: str
    generator_version: str
    dataset_kind: str
    gateway: str
    allowed_teachers: list[dict[str, str]]
    openrouter_generation_config: dict[str, Any]
    sample_count: int
    raw_sample_count: int
    merkle_root: str
    prompts_sha256: str
    created_at: str
    gpu_profile: dict[str, Any]
    validation: dict[str, Any]
    gpu_architecture: str = ARCH_BLACKWELL
    attestation_hash: str | None = None
    sampling: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version": self.version,
            "generator_version": self.generator_version,
            "dataset_kind": self.dataset_kind,
            "gateway": self.gateway,
            "allowed_teachers": self.allowed_teachers,
            "generation_config": self.openrouter_generation_config,
            "openrouter_generation_config": self.openrouter_generation_config,
            "sample_count": self.sample_count,
            "raw_sample_count": self.raw_sample_count,
            "merkle_root": self.merkle_root,
            "prompts_sha256": self.prompts_sha256,
            "created_at": self.created_at,
            "gpu_profile": self.gpu_profile,
            "gpu_architecture": self.gpu_architecture,
            "validation": self.validation,
        }
        if self.attestation_hash is not None:
            out["attestation_hash"] = self.attestation_hash
        if self.sampling is not None:
            out["sampling"] = self.sampling
        return out


def build_manifest_v2(
    trajectories: list[dict[str, Any]],
    *,
    prompts_sha256: str,
    gpu_profile: dict[str, Any],
    raw_sample_count: int,
    benchmark_enabled: bool,
    openrouter_generation_config: dict[str, Any],
    gateway: str | None = None,
    attestation_hash: str | None = None,
    sampling: dict[str, Any] | None = None,
) -> BlackwellDatasetManifest:
    gateway_name = gateway or (trajectories[0].get("gateway") if trajectories else None) or "openrouter"
    for record in trajectories:
        validate_provider_model(record["provider"], record["model"])
        validate_gateway_trajectory(record)
        validation = record.get("sparkproof_validation")
        if not validation or not validation.get("passed"):
            raise ValueError("sparkproof-2 manifest requires verified trajectories only")

    leaves = [verified_sample_leaf_hash(record) for record in trajectories]
    return BlackwellDatasetManifest(
        version="sparkproof-2",
        generator_version=GENERATOR_VERSION,
        dataset_kind=DATASET_KIND_TRITON_BLACKWELL,
        gateway=gateway_name,
        allowed_teachers=allowed_teachers_manifest(gateway_name),
        openrouter_generation_config=openrouter_generation_config,
        sample_count=len(trajectories),
        raw_sample_count=raw_sample_count,
        merkle_root=merkle_root(leaves),
        prompts_sha256=prompts_sha256,
        created_at=datetime.datetime.now(datetime.UTC).isoformat(),
        gpu_profile=gpu_profile,
        gpu_architecture=gpu_profile.get("gpu_architecture", ARCH_BLACKWELL),
        validation={
            "triton_version": TRITON_VERSION,
            "stages": ["syntax", "triton_api", "compile_execute"]
            + (["benchmark"] if benchmark_enabled else []),
            "benchmark_enabled": benchmark_enabled,
        },
        attestation_hash=attestation_hash,
        sampling=sampling,
    )
