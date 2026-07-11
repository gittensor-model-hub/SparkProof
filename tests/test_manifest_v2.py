import json
from pathlib import Path

from sparkproof.bundle import write_bundle
from sparkproof.hashing import sha256_file
from sparkproof.manifest import build_manifest_v2
from sparkproof.openrouter_request import build_chat_body, request_sha256
from sparkproof.policy import REQUIRED_REASONING_EFFORT
from sparkproof.verify import verify_bundle
from tests.conftest_helpers import TEST_GEN_CONFIG


def _verified_traj(provider: str, model: str, openrouter_model: str) -> dict:
    prompt = "Write vector_add Triton kernel"
    body = build_chat_body(
        provider=provider,
        prompt=prompt,
        system=None,
        max_tokens=TEST_GEN_CONFIG["max_tokens"],
        temperature=TEST_GEN_CONFIG["temperature"],
    )
    return {
        "prompt": prompt,
        "response": "```python\nimport triton\n```",
        "provider": provider,
        "model": model,
        "gateway": "openrouter",
        "api_base": "https://openrouter.ai/api/v1",
        "request_url": "https://openrouter.ai/api/v1/chat/completions",
        "gateway_model": openrouter_model,
        "openrouter_model": openrouter_model,
        "request_sha256": request_sha256(body),
        "response_sha256": "b" * 64,
        "metadata": {
            "gateway_reasoning_effort": REQUIRED_REASONING_EFFORT,
            "openrouter_reasoning_effort": REQUIRED_REASONING_EFFORT,
            "openrouter_max_tokens": TEST_GEN_CONFIG["max_tokens"],
            "openrouter_temperature": TEST_GEN_CONFIG["temperature"],
        },
        "sparkproof_validation": {
            "passed": True,
            "triton_version": "3.7.1",
            "code_sha256": "c" * 64,
            "stages": {"syntax": {"passed": True}, "compile_execute": {"passed": True}},
        },
    }


def test_build_manifest_v2_merkle(tmp_path: Path):
    trajectories = [
        _verified_traj("anthropic", "claude-fable-5", "anthropic/claude-fable-5"),
    ]
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "x"}) + "\n")
    manifest = build_manifest_v2(
        trajectories,
        prompts_sha256=sha256_file(str(prompts)),
        gpu_profile={
            "family": "blackwell",
            "profile": "workstation",
            "name": "NVIDIA RTX PRO 6000 Blackwell",
            "capability": [12, 0],
        },
        raw_sample_count=3,
        benchmark_enabled=False,
        openrouter_generation_config=TEST_GEN_CONFIG,
        attestation_hash="d" * 64,
    ).to_dict()
    assert manifest["version"] == "sparkproof-2"
    assert manifest["sample_count"] == 1
    assert manifest["raw_sample_count"] == 3
    assert manifest["openrouter_generation_config"]["reasoning_effort"] == "xhigh"


def test_build_manifest_v2_carries_sampling_provenance(tmp_path: Path):
    trajectories = [_verified_traj("anthropic", "claude-fable-5", "anthropic/claude-fable-5")]
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "x"}) + "\n")
    sampling = {"policy": "stratified-v1", "run_seed": "abc", "catalog_sha256": "e" * 64}
    manifest = build_manifest_v2(
        trajectories,
        prompts_sha256=sha256_file(str(prompts)),
        gpu_profile={"family": "blackwell", "profile": "workstation", "name": "x", "capability": [12, 0]},
        raw_sample_count=1,
        benchmark_enabled=False,
        openrouter_generation_config=TEST_GEN_CONFIG,
        sampling=sampling,
    ).to_dict()
    assert manifest["sampling"] == sampling


def test_verify_sparkproof_v2_bundle(tmp_path: Path):
    trajectories = [
        _verified_traj("anthropic", "claude-fable-5", "anthropic/claude-fable-5"),
    ]
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "x"}) + "\n")
    prompts_sha = sha256_file(str(prompts))
    gpu_profile = {
        "family": "blackwell",
        "profile": "workstation",
        "name": "NVIDIA RTX PRO 6000 Blackwell",
        "capability": [12, 0],
        "device_index": 0,
    }
    manifest = build_manifest_v2(
        trajectories,
        prompts_sha256=prompts_sha,
        gpu_profile=gpu_profile,
        raw_sample_count=1,
        benchmark_enabled=False,
        openrouter_generation_config=TEST_GEN_CONFIG,
        attestation_hash="e" * 64,
    ).to_dict()
    bundle = tmp_path / "bundle"
    write_bundle(out_dir=bundle, trajectories=trajectories, manifest=manifest, prompts_path=prompts)
    (bundle / "trajectories_raw.jsonl").write_text((bundle / "trajectories.jsonl").read_text())
    (bundle / "validation_report.jsonl").write_text(
        json.dumps({"index": 0, "validation": trajectories[0]["sparkproof_validation"]}) + "\n"
    )
    (bundle / "gpu_attestation.json").write_text(
        json.dumps(
            {
                "passed": True,
                "environment": "REMOTE",
                "token": "tok",
                "claims": {},
                "gpu_profile": gpu_profile,
                "token_sha256": "e" * 64,
            }
        )
    )
    report = verify_bundle(bundle, require_gpu_attestation=False)
    assert report["verified"] is True
    assert report["manifest_version"] == "sparkproof-2"
    assert report["gpu_attested"] is True
