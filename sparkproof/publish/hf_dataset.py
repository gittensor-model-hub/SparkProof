"""Publish verified trajectories to Hugging Face datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def trajectory_to_messages_record(trajectory: dict[str, Any]) -> dict[str, Any] | None:
    if not trajectory.get("response"):
        return None
    validation = trajectory.get("sparkproof_validation") or {}
    if validation and not validation.get("passed"):
        return None

    system = trajectory.get("system") or (
        "You are a Triton 3.7.1 GPU kernel expert for Blackwell SM12x."
    )
    reasoning = trajectory.get("reasoning")
    assistant = trajectory.get("response", "")
    if reasoning:
        assistant = f"<think>\n{reasoning.strip()}\n</think>\n\n{assistant}"

    meta = {
        "provider": trajectory.get("provider"),
        "gateway": trajectory.get("gateway"),
        "task_id": (trajectory.get("metadata") or {}).get("prompt_meta", {}).get("task_id"),
        "category": (trajectory.get("metadata") or {}).get("prompt_meta", {}).get("category"),
    }
    if validation:
        meta["validation_score"] = (validation.get("benchmark") or {}).get("composite_score")

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": trajectory["prompt"]},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": meta,
    }


def load_trajectories_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# Bundle artifacts mirrored into the HF dataset repo under proof/ so a validator can
# re-run `sparkproof-verify` (and the SparkDistill dataset gate) from the HF link alone.
PROOF_ARTIFACTS = (
    "manifest.json",
    "dataset_manifest.json",
    "gpu_attestation.json",
    "validation_report.jsonl",
    "prompts.jsonl",
    "trajectories.jsonl",
    "trajectories_raw.jsonl",
)


def upload_proof_artifacts(*, api: Any, bundle_dir: Path, repo_id: str) -> list[str]:
    uploaded: list[str] = []
    for name in PROOF_ARTIFACTS:
        path = bundle_dir / name
        if not path.exists():
            continue
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"proof/{name}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"SparkProof bundle artifact: {name}",
        )
        uploaded.append(name)
    return uploaded


def publish_bundle_to_hf(
    *,
    bundle_dir: Path,
    repo_id: str,
    private: bool = False,
    split: str = "train",
    include_proof: bool = True,
) -> str:
    from datasets import Dataset
    from huggingface_hub import HfApi

    traj_path = bundle_dir / "trajectories.jsonl"
    if not traj_path.exists():
        raise FileNotFoundError(f"missing {traj_path}")

    rows: list[dict[str, Any]] = []
    for traj in load_trajectories_jsonl(traj_path):
        row = trajectory_to_messages_record(traj)
        if row:
            rows.append(row)

    if not rows:
        raise ValueError("no verified trajectories to publish")

    ds = Dataset.from_list(rows)
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=private)
    ds.push_to_hub(repo_id, split=split, commit_message="SparkProof verified Triton trajectories")
    if include_proof:
        upload_proof_artifacts(api=api, bundle_dir=bundle_dir, repo_id=repo_id)
    return f"https://huggingface.co/datasets/{repo_id}"
