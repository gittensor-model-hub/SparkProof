import json
import sys
import types
from pathlib import Path

import pytest

from sparkproof.triton_dataset.registry_snapshot import (
    HF_SNAPSHOT_FILENAME,
    HF_TASK_IDS_FILENAME,
    MANIFEST_SNAPSHOT_ROWS_KEY,
    MANIFEST_SNAPSHOT_SHA256_KEY,
    MANIFEST_TASK_IDS_ROWS_KEY,
    MANIFEST_TASK_IDS_SHA256_KEY,
    RegistrySnapshotDownload,
    count_jsonl_rows,
    download_registry_snapshot,
    resolve_registry_snapshot_path,
    sha256_file,
    verify_snapshot_against_manifest,
)


def _traj(task_id: str, prompt: str) -> dict:
    return {
        "prompt": prompt,
        "response": f"```python\n# {task_id}\n```",
        "metadata": {"prompt_meta": {"task_id": task_id, "origin": "torch_op", "split": "train"}},
        "sparkproof_validation": {"passed": True},
    }


def test_verify_snapshot_against_manifest_detects_sha_and_row_mismatch(tmp_path: Path):
    snapshot = tmp_path / HF_SNAPSHOT_FILENAME
    snapshot.write_text(json.dumps(_traj("a", "prompt")) + "\n", encoding="utf-8")
    manifest = {
        MANIFEST_SNAPSHOT_SHA256_KEY: "f" * 64,
        MANIFEST_SNAPSHOT_ROWS_KEY: 99,
    }
    issues = verify_snapshot_against_manifest(snapshot, manifest)
    assert any(MANIFEST_SNAPSHOT_SHA256_KEY in issue for issue in issues)
    assert any(MANIFEST_SNAPSHOT_ROWS_KEY in issue for issue in issues)


def test_verify_snapshot_against_manifest_checks_task_ids_pin(tmp_path: Path):
    snapshot = tmp_path / HF_SNAPSHOT_FILENAME
    snapshot.write_text(json.dumps(_traj("a", "prompt")) + "\n", encoding="utf-8")
    task_ids = tmp_path / HF_TASK_IDS_FILENAME
    task_ids.write_text(json.dumps({"task_ids_total": 1, "task_ids": ["a"]}) + "\n", encoding="utf-8")
    manifest = {
        MANIFEST_SNAPSHOT_SHA256_KEY: sha256_file(snapshot),
        MANIFEST_SNAPSHOT_ROWS_KEY: 1,
        MANIFEST_TASK_IDS_SHA256_KEY: "a" * 64,
        MANIFEST_TASK_IDS_ROWS_KEY: 1,
    }
    issues = verify_snapshot_against_manifest(snapshot, manifest, task_ids_path=task_ids)
    assert any(MANIFEST_TASK_IDS_SHA256_KEY in issue for issue in issues)


def test_download_registry_snapshot_verifies_pins(tmp_path: Path, monkeypatch):
    snapshot = tmp_path / HF_SNAPSHOT_FILENAME
    snapshot.write_text(json.dumps(_traj("task_a", "prompt A")) + "\n", encoding="utf-8")
    task_ids = tmp_path / HF_TASK_IDS_FILENAME
    task_ids.write_text(json.dumps({"task_ids_total": 1, "task_ids": ["task_a"]}) + "\n", encoding="utf-8")
    manifest = {
        MANIFEST_SNAPSHOT_SHA256_KEY: sha256_file(snapshot),
        MANIFEST_SNAPSHOT_ROWS_KEY: 1,
        MANIFEST_TASK_IDS_SHA256_KEY: sha256_file(task_ids),
        MANIFEST_TASK_IDS_ROWS_KEY: 1,
    }

    def fake_download(repo_id, repo_type, filename, token=None, cache_dir=None):
        if filename == "mix_manifest.json":
            manifest_path = tmp_path / "mix_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            return str(manifest_path)
        if filename == HF_SNAPSHOT_FILENAME:
            return str(snapshot)
        if filename == HF_TASK_IDS_FILENAME:
            return str(task_ids)
        raise AssertionError(filename)

    fake_hub = types.ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    report = download_registry_snapshot(repo_id="org/mining", out_dir=tmp_path / "out")
    assert isinstance(report, RegistrySnapshotDownload)
    assert report.rows_total == 1
    assert (tmp_path / "out" / HF_SNAPSHOT_FILENAME).exists()


def test_resolve_registry_snapshot_path_verifies_local_file(tmp_path: Path, monkeypatch):
    snapshot = tmp_path / "local.jsonl"
    snapshot.write_text(json.dumps(_traj("task_a", "prompt A")) + "\n", encoding="utf-8")
    manifest = {
        MANIFEST_SNAPSHOT_SHA256_KEY: sha256_file(snapshot),
        MANIFEST_SNAPSHOT_ROWS_KEY: count_jsonl_rows(snapshot),
    }

    monkeypatch.setattr(
        "sparkproof.triton_dataset.registry_snapshot.download_mining_manifest",
        lambda repo_id, hf_token=None: manifest,
    )

    resolved = resolve_registry_snapshot_path(
        registry_snapshot=snapshot,
        mining_repo="org/mining",
    )
    assert resolved == snapshot.resolve()


def test_resolve_registry_snapshot_path_rejects_bad_local_pin(tmp_path: Path, monkeypatch):
    snapshot = tmp_path / "local.jsonl"
    snapshot.write_text(json.dumps(_traj("task_a", "prompt A")) + "\n", encoding="utf-8")
    manifest = {MANIFEST_SNAPSHOT_SHA256_KEY: "f" * 64, MANIFEST_SNAPSHOT_ROWS_KEY: 1}
    monkeypatch.setattr(
        "sparkproof.triton_dataset.registry_snapshot.download_mining_manifest",
        lambda repo_id, hf_token=None: manifest,
    )
    with pytest.raises(ValueError, match="pin verification failed"):
        resolve_registry_snapshot_path(registry_snapshot=snapshot, mining_repo="org/mining")
