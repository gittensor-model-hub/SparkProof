import json
from pathlib import Path

import pytest

from sparkproof.bundle import write_bundle
from sparkproof.hashing import sha256_file
from sparkproof.manifest import build_manifest
from sparkproof.policy import (
    GATEWAY,
    allowed_teachers_manifest,
    validate_openrouter_trajectory,
)
from sparkproof.hashing import canonical_json_bytes, dataset_attestation_nonce, sha256_hex
from sparkproof.verify import verify_bundle, verify_gpu_attestation, verify_manifest_policy
from tests.conftest_helpers import TEST_GEN_CONFIG, make_trajectory


def test_verify_gpu_attestation_accepts_matching_nonce(tmp_path: Path):
    raw = [make_trajectory("anthropic", "claude-fable-5")]
    (tmp_path / "trajectories_raw.jsonl").write_text(json.dumps(raw[0]) + "\n")
    manifest = {"prompts_sha256": "prompts-hash", "gpu_profile": {}}
    nonce = dataset_attestation_nonce("prompts-hash", sha256_hex(canonical_json_bytes(raw)))
    (tmp_path / "gpu_attestation.json").write_text(
        json.dumps({"passed": True, "nonce": nonce, "gpu_profile": {}})
    )

    assert verify_gpu_attestation(tmp_path, manifest) == []


def test_verify_gpu_attestation_flags_swapped_trajectories_raw(tmp_path: Path):
    raw = [make_trajectory("anthropic", "claude-fable-5")]
    manifest = {"prompts_sha256": "prompts-hash", "gpu_profile": {}}
    nonce = dataset_attestation_nonce("prompts-hash", sha256_hex(canonical_json_bytes(raw)))
    (tmp_path / "gpu_attestation.json").write_text(
        json.dumps({"passed": True, "nonce": nonce, "gpu_profile": {}})
    )
    # trajectories_raw.jsonl written AFTER attestation, with different content than
    # what the nonce was bound to — must be flagged, not silently accepted.
    swapped = [make_trajectory("openai", "gpt-5.6")]
    (tmp_path / "trajectories_raw.jsonl").write_text(json.dumps(swapped[0]) + "\n")

    issues = verify_gpu_attestation(tmp_path, manifest)
    assert any("nonce" in issue for issue in issues)


def test_verify_local_bundle_with_allow_unattested(tmp_path: Path):
    trajectories = [make_trajectory("anthropic", "claude-fable-5"), make_trajectory("openai", "gpt-5.6")]
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "2+2?"}) + "\n")
    prompts_sha256 = sha256_file(str(prompts))
    manifest = build_manifest(
        trajectories,
        prompts_sha256=prompts_sha256,
        openrouter_generation_config=TEST_GEN_CONFIG,
    ).to_dict()
    bundle_dir = tmp_path / "bundle"
    write_bundle(out_dir=bundle_dir, trajectories=trajectories, manifest=manifest, prompts_path=prompts)

    report = verify_bundle(bundle_dir, require_gpu_attestation=False)
    assert report["verified"] is True
    assert report["issues"] == []
    assert report["gateway"] == GATEWAY


def test_verify_rejects_wrong_model(tmp_path: Path):
    trajectories = [make_trajectory("openai", "gpt-5")]
    manifest = {
        "version": "sparkproof-1",
        "generator_version": "0.2.0",
        "gateway": GATEWAY,
        "allowed_teachers": allowed_teachers_manifest(),
        "openrouter_generation_config": TEST_GEN_CONFIG,
        "sample_count": 1,
        "merkle_root": "0" * 64,
        "prompts_sha256": "abc",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    bundle_dir = tmp_path / "bad"
    write_bundle(out_dir=bundle_dir, trajectories=trajectories, manifest=manifest)
    report = verify_bundle(bundle_dir, require_gpu_attestation=False)
    assert report["verified"] is False


def test_verify_rejects_direct_api_trajectory(tmp_path: Path):
    record = make_trajectory("anthropic", "claude-fable-5")
    record["gateway"] = "direct"
    with pytest.raises(ValueError, match="gateway"):
        validate_openrouter_trajectory(record)


def _sparkproof2_manifest(**overrides):
    base = {
        "version": "sparkproof-2",
        "gateway": "openrouter",
        "allowed_teachers": allowed_teachers_manifest("openrouter"),
        "dataset_kind": "triton-3.7.1-blackwell",
        "gpu_profile": {"family": "blackwell", "gpu_architecture": "blackwell"},
        "gpu_architecture": "blackwell",
    }
    base.update(overrides)
    return base


def test_verify_manifest_policy_accepts_blackwell():
    issues = verify_manifest_policy(_sparkproof2_manifest())
    assert not any("gpu_architecture" in i for i in issues)


def test_verify_manifest_policy_accepts_hopper_h100():
    manifest = _sparkproof2_manifest(
        gpu_profile={"family": "hopper", "gpu_architecture": "hopper-h100"},
        gpu_architecture="hopper-h100",
    )
    issues = verify_manifest_policy(manifest)
    assert not any("gpu_architecture" in i for i in issues)


def test_verify_manifest_policy_rejects_unsupported_architecture():
    manifest = _sparkproof2_manifest(
        gpu_profile={"family": "ampere", "gpu_architecture": "ampere-a100"},
        gpu_architecture="ampere-a100",
    )
    issues = verify_manifest_policy(manifest)
    assert any("gpu_architecture" in i for i in issues)


def test_verify_manifest_policy_rejects_missing_architecture():
    manifest = _sparkproof2_manifest(gpu_profile={}, gpu_architecture=None)
    issues = verify_manifest_policy(manifest)
    assert any("gpu_architecture" in i for i in issues)


def test_verify_manifest_policy_reads_architecture_from_gpu_profile_fallback():
    # Older callers might only set gpu_profile.gpu_architecture without the
    # top-level manifest field — still accepted via fallback.
    manifest = _sparkproof2_manifest(gpu_architecture=None, gpu_profile={"gpu_architecture": "hopper-h200"})
    issues = verify_manifest_policy(manifest)
    assert not any("gpu_architecture" in i for i in issues)
