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
    from sparkproof.triton_dataset.episodes import episode_to_messages, trajectory_has_episode
    from sparkproof.triton_dataset.training_cot import normalize_training_reasoning

    # Prefer the original mining-task prompt when top-level prompt is a repair /
    # CoT-recovery wrapper (prompt_meta is stamped by generate_best_of_n).
    prompt_meta = (trajectory.get("metadata") or {}).get("prompt_meta") or {}
    user_prompt = prompt_meta.get("prompt") or trajectory.get("prompt") or ""

    meta = {
        "provider": trajectory.get("provider"),
        "model": trajectory.get("model"),
        "gateway": trajectory.get("gateway"),
        "task_id": prompt_meta.get("task_id"),
        "category": prompt_meta.get("category"),
    }
    cot_recovery = (trajectory.get("metadata") or {}).get("cot_recovery")
    if cot_recovery:
        meta["cot_recovery"] = cot_recovery
    if (trajectory.get("metadata") or {}).get("cot_provider"):
        meta["cot_provider"] = trajectory["metadata"]["cot_provider"]
    if (trajectory.get("metadata") or {}).get("multi_turn"):
        meta["multi_turn"] = True
        meta["episode_version"] = (trajectory.get("metadata") or {}).get("episode_version")
        episode = (trajectory.get("metadata") or {}).get("episode") or {}
        meta["repairs_used"] = episode.get("repairs_used")
        meta["optimize_improved"] = episode.get("optimize_improved")
    gateway_model = trajectory.get("gateway_model") or trajectory.get("openrouter_model")
    if gateway_model:
        meta["gateway_model"] = gateway_model
    if validation:
        meta["validation_score"] = (validation.get("benchmark") or {}).get("composite_score")
    tier = (trajectory.get("metadata") or {}).get("tier")
    if tier:
        meta["tier"] = tier

    # Multi-turn episodes are the high-value SFT form (fail→critique→fix→optimize).
    if trajectory_has_episode(trajectory):
        messages = episode_to_messages((trajectory.get("metadata") or {})["episode"])
        # Ensure system matches trajectory system when present.
        if system and messages and messages[0].get("role") == "system":
            messages[0] = {"role": "system", "content": system}
        return {"messages": messages, "metadata": meta}

    reasoning = normalize_training_reasoning(trajectory.get("reasoning"))
    assistant = trajectory.get("response", "")
    if reasoning:
        assistant = f"<think>\n{reasoning.strip()}\n</think>\n\n{assistant}"

    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
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
    "novelty_report.json",
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
    # Upload proof artifacts before the dataset rows go live: if this fails partway,
    # nothing publicly consumable has been published yet, so a failed exit code means
    # what it says instead of leaving a public, partially-unverifiable dataset behind.
    if include_proof:
        upload_proof_artifacts(api=api, bundle_dir=bundle_dir, repo_id=repo_id)
    ds.push_to_hub(repo_id, split=split, commit_message="SparkProof verified Triton trajectories")
    return f"https://huggingface.co/datasets/{repo_id}"
