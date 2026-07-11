import json
import sys
import types
from pathlib import Path

import pytest

from sparkproof.publish.hf_dataset import (
    PROOF_ARTIFACTS,
    load_trajectories_jsonl,
    publish_bundle_to_hf,
    trajectory_to_messages_record,
    upload_proof_artifacts,
)


def test_trajectory_to_messages_record_skips_empty_response():
    assert trajectory_to_messages_record({"response": ""}) is None
    assert trajectory_to_messages_record({}) is None


def test_trajectory_to_messages_record_skips_failed_validation():
    trajectory = {"response": "code", "sparkproof_validation": {"passed": False}}
    assert trajectory_to_messages_record(trajectory) is None


def test_trajectory_to_messages_record_builds_messages_and_metadata():
    trajectory = {
        "response": "kernel code",
        "prompt": "write a kernel",
        "provider": "anthropic",
        "gateway": "openrouter",
        "sparkproof_validation": {"passed": True, "benchmark": {"composite_score": 0.8}},
        "metadata": {"prompt_meta": {"task_id": "t1", "category": "translation"}},
    }
    record = trajectory_to_messages_record(trajectory)
    assert record["messages"][0]["role"] == "system"
    assert record["messages"][1] == {"role": "user", "content": "write a kernel"}
    assert record["messages"][2] == {"role": "assistant", "content": "kernel code"}
    assert record["metadata"] == {
        "provider": "anthropic",
        "gateway": "openrouter",
        "task_id": "t1",
        "category": "translation",
        "validation_score": 0.8,
    }


def test_trajectory_to_messages_record_wraps_reasoning_in_think_block():
    trajectory = {"response": "code", "prompt": "p", "reasoning": "  step by step  "}
    record = trajectory_to_messages_record(trajectory)
    assert record["messages"][2]["content"] == "<think>\nstep by step\n</think>\n\ncode"


def test_load_trajectories_jsonl_skips_blank_lines(tmp_path: Path):
    path = tmp_path / "trajectories.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n')
    assert load_trajectories_jsonl(path) == [{"a": 1}, {"a": 2}]


def test_upload_proof_artifacts_skips_missing_and_uploads_present(tmp_path: Path):
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "gpu_attestation.json").write_text("{}")
    # every other PROOF_ARTIFACTS name is intentionally absent

    calls = []

    class FakeApi:
        def upload_file(self, **kwargs):
            calls.append(kwargs)

    uploaded = upload_proof_artifacts(api=FakeApi(), bundle_dir=tmp_path, repo_id="org/ds")

    assert uploaded == ["manifest.json", "gpu_attestation.json"]
    assert {c["path_in_repo"] for c in calls} == {"proof/manifest.json", "proof/gpu_attestation.json"}
    assert all(c["repo_id"] == "org/ds" and c["repo_type"] == "dataset" for c in calls)


def _install_fake_hf_modules(monkeypatch, *, call_order: list):
    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

        def push_to_hub(self, repo_id, split="train", commit_message=None):
            call_order.append(("push_to_hub", repo_id))

    class FakeApi:
        def create_repo(self, **kwargs):
            call_order.append(("create_repo", kwargs.get("repo_id")))

        def upload_file(self, **kwargs):
            call_order.append(("upload_file", kwargs["path_in_repo"]))

    datasets_module = types.ModuleType("datasets")
    datasets_module.Dataset = FakeDataset
    hub_module = types.ModuleType("huggingface_hub")
    hub_module.HfApi = FakeApi
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)


def _write_bundle(tmp_path: Path, *, verified: bool = True) -> Path:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    trajectory = {
        "response": "code",
        "prompt": "p",
        "sparkproof_validation": {"passed": verified},
    }
    (bundle_dir / "trajectories.jsonl").write_text(json.dumps(trajectory) + "\n")
    (bundle_dir / "manifest.json").write_text("{}")
    (bundle_dir / "gpu_attestation.json").write_text("{}")
    return bundle_dir


def test_publish_bundle_to_hf_requires_trajectories_file(tmp_path: Path, monkeypatch):
    _install_fake_hf_modules(monkeypatch, call_order=[])
    with pytest.raises(FileNotFoundError):
        publish_bundle_to_hf(bundle_dir=tmp_path, repo_id="org/ds")


def test_publish_bundle_to_hf_requires_verified_trajectories(tmp_path: Path, monkeypatch):
    call_order: list = []
    _install_fake_hf_modules(monkeypatch, call_order=call_order)
    bundle_dir = _write_bundle(tmp_path, verified=False)

    with pytest.raises(ValueError, match="no verified trajectories"):
        publish_bundle_to_hf(bundle_dir=bundle_dir, repo_id="org/ds")


def test_publish_bundle_to_hf_uploads_proof_before_dataset_rows_go_public(tmp_path: Path, monkeypatch):
    call_order: list = []
    _install_fake_hf_modules(monkeypatch, call_order=call_order)
    bundle_dir = _write_bundle(tmp_path)

    url = publish_bundle_to_hf(bundle_dir=bundle_dir, repo_id="org/ds")

    assert url == "https://huggingface.co/datasets/org/ds"
    kinds = [c[0] for c in call_order]
    assert kinds.index("push_to_hub") > kinds.index("upload_file"), (
        "dataset rows must not go public before proof artifacts are confirmed uploaded"
    )
    # trajectories.jsonl, manifest.json, gpu_attestation.json are all present in the
    # fixture and all three are in PROOF_ARTIFACTS.
    assert kinds.count("upload_file") == 3


def test_publish_bundle_to_hf_skips_proof_when_disabled(tmp_path: Path, monkeypatch):
    call_order: list = []
    _install_fake_hf_modules(monkeypatch, call_order=call_order)
    bundle_dir = _write_bundle(tmp_path)

    publish_bundle_to_hf(bundle_dir=bundle_dir, repo_id="org/ds", include_proof=False)

    assert "upload_file" not in [c[0] for c in call_order]


def test_proof_artifacts_list_matches_known_bundle_files():
    assert set(PROOF_ARTIFACTS) == {
        "manifest.json",
        "dataset_manifest.json",
        "gpu_attestation.json",
        "novelty_report.json",
        "validation_report.jsonl",
        "prompts.jsonl",
        "trajectories.jsonl",
        "trajectories_raw.jsonl",
    }
