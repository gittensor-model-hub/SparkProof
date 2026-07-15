"""Download and verify SparkDistill's accepted-registry snapshot for miner dedupe checks.

SparkDistill publishes ``accepted_registry_snapshot.jsonl`` on the canonical mining HF
repo and pins ``accepted_registry_snapshot_sha256`` in ``mix_manifest.json``. Miners pass
the snapshot to the release gate via ``--registry-snapshot`` so ``novelty_report.json``
includes cross-registry duplicates before opening a dataset PR.

This module mirrors the miner-facing half of SparkDistill's
``eval.export_registry_snapshot`` (download + pin verify). Building the snapshot from
``datasets/registry.jsonl`` remains a SparkDistill maintainer/CI concern.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MINING_DATASET_REPO = "gittensor-model-hub/sparkproof-mining"
MINING_MANIFEST_PATH = "mix_manifest.json"
HF_SNAPSHOT_FILENAME = "accepted_registry_snapshot.jsonl"
HF_TASK_IDS_FILENAME = "accepted_task_ids.json"
MANIFEST_SNAPSHOT_SHA256_KEY = "accepted_registry_snapshot_sha256"
MANIFEST_SNAPSHOT_ROWS_KEY = "accepted_registry_snapshot_rows_total"
MANIFEST_TASK_IDS_SHA256_KEY = "accepted_task_ids_sha256"
MANIFEST_TASK_IDS_ROWS_KEY = "accepted_task_ids_total"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def count_jsonl_rows(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def verify_snapshot_against_manifest(
    snapshot_path: Path,
    manifest: dict[str, Any],
    *,
    task_ids_path: Path | None = None,
) -> list[str]:
    """Confirm a local snapshot file matches mix_manifest pins."""
    issues: list[str] = []
    expected_sha = manifest.get(MANIFEST_SNAPSHOT_SHA256_KEY)
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        issues.append(f"mix_manifest missing {MANIFEST_SNAPSHOT_SHA256_KEY}")
        return issues

    if not snapshot_path.exists():
        issues.append(f"snapshot file not found: {snapshot_path}")
        return issues

    actual_sha = sha256_file(snapshot_path)
    if actual_sha != expected_sha:
        issues.append(
            f"{MANIFEST_SNAPSHOT_SHA256_KEY} mismatch: manifest={expected_sha} file={actual_sha}"
        )

    expected_rows = manifest.get(MANIFEST_SNAPSHOT_ROWS_KEY)
    if expected_rows is not None:
        actual_rows = count_jsonl_rows(snapshot_path)
        if int(expected_rows) != actual_rows:
            issues.append(
                f"{MANIFEST_SNAPSHOT_ROWS_KEY} mismatch: manifest={expected_rows} file={actual_rows}"
            )

    expected_task_ids_sha = manifest.get(MANIFEST_TASK_IDS_SHA256_KEY)
    if isinstance(expected_task_ids_sha, str) and len(expected_task_ids_sha) == 64:
        if task_ids_path is None or not task_ids_path.exists():
            issues.append(f"mix_manifest pins {MANIFEST_TASK_IDS_SHA256_KEY} but task_ids file is missing")
        else:
            actual_task_ids_sha = sha256_file(task_ids_path)
            if actual_task_ids_sha != expected_task_ids_sha:
                issues.append(
                    f"{MANIFEST_TASK_IDS_SHA256_KEY} mismatch: manifest={expected_task_ids_sha} file={actual_task_ids_sha}"
                )
            expected_task_rows = manifest.get(MANIFEST_TASK_IDS_ROWS_KEY)
            if expected_task_rows is not None:
                payload = json.loads(task_ids_path.read_text(encoding="utf-8"))
                actual_task_rows = int(payload.get("task_ids_total") or len(payload.get("task_ids") or []))
                if int(expected_task_rows) != actual_task_rows:
                    issues.append(
                        f"{MANIFEST_TASK_IDS_ROWS_KEY} mismatch: manifest={expected_task_rows} file={actual_task_rows}"
                    )

    return issues


@dataclass(frozen=True)
class RegistrySnapshotDownload:
    repo_id: str
    manifest_path: Path
    snapshot_path: Path
    task_ids_path: Path | None
    manifest: dict[str, Any]
    rows_total: int
    sha256: str

    def verify(self) -> list[str]:
        return verify_snapshot_against_manifest(
            self.snapshot_path,
            self.manifest,
            task_ids_path=self.task_ids_path,
        )


def download_mining_manifest(
    repo_id: str = DEFAULT_MINING_DATASET_REPO,
    *,
    hf_token: str | None = None,
) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    manifest_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=MINING_MANIFEST_PATH,
            token=hf_token,
        )
    )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def download_registry_snapshot(
    *,
    repo_id: str = DEFAULT_MINING_DATASET_REPO,
    out_dir: Path | None = None,
    hf_token: str | None = None,
) -> RegistrySnapshotDownload:
    """Download mix_manifest + accepted snapshot from HF and verify pins."""
    from huggingface_hub import hf_hub_download

    manifest = download_mining_manifest(repo_id, hf_token=hf_token)
    expected_sha = manifest.get(MANIFEST_SNAPSHOT_SHA256_KEY)
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError(
            f"{repo_id}/{MINING_MANIFEST_PATH} missing {MANIFEST_SNAPSHOT_SHA256_KEY} "
            "(mining repo may predate accepted-registry snapshots)"
        )

    cache_dir = str(out_dir) if out_dir is not None else None
    snapshot_hf_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=HF_SNAPSHOT_FILENAME,
            token=hf_token,
            cache_dir=cache_dir,
        )
    )

    task_ids_path: Path | None = None
    if manifest.get(MANIFEST_TASK_IDS_SHA256_KEY):
        task_ids_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=HF_TASK_IDS_FILENAME,
                token=hf_token,
                cache_dir=cache_dir,
            )
        )

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        local_snapshot = out_dir / HF_SNAPSHOT_FILENAME
        local_snapshot.write_bytes(snapshot_hf_path.read_bytes())
        snapshot_path = local_snapshot
        if task_ids_path is not None:
            local_task_ids = out_dir / HF_TASK_IDS_FILENAME
            local_task_ids.write_bytes(task_ids_path.read_bytes())
            task_ids_path = local_task_ids
    else:
        snapshot_path = snapshot_hf_path

    download = RegistrySnapshotDownload(
        repo_id=repo_id,
        manifest_path=Path(snapshot_hf_path).parent / MINING_MANIFEST_PATH,
        snapshot_path=snapshot_path,
        task_ids_path=task_ids_path,
        manifest=manifest,
        rows_total=count_jsonl_rows(snapshot_path),
        sha256=sha256_file(snapshot_path),
    )
    issues = download.verify()
    if issues:
        raise ValueError(f"registry snapshot pin verification failed: {'; '.join(issues)}")
    return download


def resolve_registry_snapshot_path(
    *,
    registry_snapshot: Path | None = None,
    mining_repo: str | None = None,
    cache_dir: Path | None = None,
    hf_token: str | None = None,
    verify_manifest: dict[str, Any] | None = None,
) -> Path | None:
    """Resolve the snapshot path for release-gate novelty checks."""
    if registry_snapshot is not None:
        path = registry_snapshot.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if verify_manifest is not None:
            issues = verify_snapshot_against_manifest(path, verify_manifest)
            if issues:
                raise ValueError(f"registry snapshot pin verification failed: {'; '.join(issues)}")
        elif mining_repo is not None:
            manifest = download_mining_manifest(mining_repo, hf_token=hf_token)
            issues = verify_snapshot_against_manifest(path, manifest)
            if issues:
                raise ValueError(f"registry snapshot pin verification failed: {'; '.join(issues)}")
        return path

    if mining_repo is not None:
        return download_registry_snapshot(
            repo_id=mining_repo,
            out_dir=cache_dir,
            hf_token=hf_token,
        ).snapshot_path

    return None
