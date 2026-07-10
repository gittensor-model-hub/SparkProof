"""Pre-publish release gate for verified Triton trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sparkproof.triton_dataset.decontaminate import TritonDecontaminator, extract_python_from_response
from sparkproof.triton_dataset.task_policy import FORBIDDEN_TRAINING_ORIGINS


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_trajectory_row(traj: dict[str, Any], decon: TritonDecontaminator) -> list[str]:
    issues: list[str] = []
    meta = (traj.get("metadata") or {}).get("prompt_meta") or {}
    origin = meta.get("origin") or meta.get("source")
    if origin in FORBIDDEN_TRAINING_ORIGINS:
        issues.append(f"benchmark origin {origin!r}")
    if meta.get("split") in {"test", "eval"}:
        issues.append("eval split")
    issues.extend(decon.check_task(meta))
    validation = traj.get("sparkproof_validation") or {}
    if validation.get("passed") is not True:
        issues.append("missing or failed sparkproof validation")
    code = extract_python_from_response(traj.get("response", ""))
    if decon.is_contaminated_code(code):
        issues.append("code structure matches eval benchmark")
    blob = json.dumps(traj)
    for needle in ("sk-", "/home/", "YUNWU_API_KEY", "OPENROUTER_API_KEY"):
        if needle in blob:
            issues.append(f"suspicious content: {needle}")
    return issues


def build_manifest(
    *,
    trajectories: list[dict[str, Any]],
    dataset_version: str,
    bundle_dir: Path,
) -> dict[str, Any]:
    gold = silver = repair = dpo = 0
    for t in trajectories:
        tier = (t.get("metadata") or {}).get("tier") or (t.get("sparkproof_validation") or {}).get("tier")
        if tier == "silver":
            silver += 1
        elif tier == "repair":
            repair += 1
        else:
            gold += 1
        if (t.get("metadata") or {}).get("dpo_pair"):
            dpo += 1

    manifest = {
        "dataset_version": dataset_version,
        "triton_version": "3.7.1",
        "gpu_targets": ["blackwell"],
        "rows_total": len(trajectories),
        "gold_rows": gold,
        "silver_rows": silver,
        "repair_rows": repair,
        "dpo_pairs": dpo,
    }
    traj_path = bundle_dir / "trajectories.jsonl"
    if traj_path.exists():
        manifest["trajectories_sha256"] = _sha256_file(traj_path)
    return manifest


def run_release_gate(
    bundle_dir: Path,
    *,
    dataset_version: str = "triton-distill-v0.2",
    problems_dir: Path | None = None,
    benchmark_py_dir: Path | None = None,
) -> dict[str, Any]:
    from sparkproof.publish.hf_dataset import load_trajectories_jsonl
    from sparkproof.verify import verify_bundle

    verification = verify_bundle(bundle_dir, require_gpu_attestation=True)
    if not verification.get("verified"):
        issues = verification.get("issues") or ["bundle verification failed"]
        raise ValueError(f"release gate requires a valid GPU-attested sparkproof-2 bundle: {issues}")

    traj_path = bundle_dir / "trajectories.jsonl"
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)

    trajectories = load_trajectories_jsonl(traj_path)
    decon = TritonDecontaminator(
        problems_dir=problems_dir,
        benchmark_py_dir=benchmark_py_dir,
        require_eval_corpus=True,
    )
    blocked: list[dict[str, Any]] = []
    for i, traj in enumerate(trajectories):
        issues = check_trajectory_row(traj, decon)
        if issues:
            blocked.append({"index": i, "task_id": (traj.get("metadata") or {}).get("prompt_meta", {}).get("task_id"), "issues": issues})

    manifest = build_manifest(trajectories=trajectories, dataset_version=dataset_version, bundle_dir=bundle_dir)
    manifest["blocked_rows"] = len(blocked)
    manifest["passed"] = len(blocked) == 0

    manifest_path = bundle_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    if blocked:
        (bundle_dir / "release_gate_blocked.json").write_text(json.dumps(blocked[:50], indent=2))
        raise ValueError(f"release gate failed: {len(blocked)} rows blocked (see release_gate_blocked.json)")

    return manifest
