"""Orchestrate evolution, generation, failure mining, and decontamination."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sparkproof.triton_dataset.decontaminate import TritonDecontaminator, extract_python_from_response
from sparkproof.triton_dataset.failure_miner import mine_failure_to_tasks, record_failure
from sparkproof.triton_dataset.multi_candidate import generate_best_candidate
from sparkproof.triton_dataset.self_evolve import evolve_parent
from sparkproof.triton_dataset.task_policy import assert_trainable_task, normalize_train_task


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_dataset_generation_step(
    base_task: dict[str, Any],
    *,
    client,
    validator,
    decontaminator: TritonDecontaminator,
    evolve_depth: int = 1,
    run_id: str = "orchestrate",
    run_benchmark: bool = False,
    run_seed: str | None = None,
    debug_split: Path | None = None,
    mined_split: Path | None = None,
) -> dict[str, Any]:
    """Single-task pipeline: evolve → generate → mine failures / accept clean rows."""
    task = normalize_train_task(base_task)
    assert_trainable_task(task)

    evolved_tasks = [task]
    if evolve_depth:
        evolved_tasks.extend(evolve_parent(task, depth=evolve_depth, run_seed=run_seed))

    results: list[dict[str, Any]] = []
    for evolved in evolved_tasks:
        issues = decontaminator.check_task(evolved)
        if issues:
            results.append({"task_id": evolved["task_id"], "status": "rejected_task", "issues": issues})
            continue

        outcome = generate_best_candidate(
            evolved, client=client, validator=validator, run_benchmark=run_benchmark
        )
        if outcome.get("passed"):
            code = extract_python_from_response(outcome["response"])
            if decontaminator.is_contaminated_code(code):
                results.append({"task_id": evolved["task_id"], "status": "contaminated_code"})
                continue
            results.append(
                {
                    "task_id": evolved["task_id"],
                    "prompt": evolved.get("prompt", ""),
                    "system": evolved.get("system", ""),
                    "status": "accepted",
                    "trajectory": outcome["trajectory"],
                    "tier": outcome.get("tier", "gold"),
                    "candidates": outcome.get("candidates", []),
                }
            )
        else:
            failure = record_failure(
                run_id=run_id,
                task=evolved,
                model=outcome.get("provider") or getattr(client, "model", "teacher"),
                validation=outcome.get("validation") or {},
                response=outcome.get("response", ""),
            )
            if debug_split:
                _append_jsonl(debug_split, failure)
            mined = mine_failure_to_tasks(failure)
            if mined_split:
                for m in mined:
                    _append_jsonl(mined_split, m)
            results.append(
                {
                    "task_id": evolved["task_id"],
                    "prompt": evolved.get("prompt", ""),
                    "system": evolved.get("system", ""),
                    "status": "mined_failure",
                    "failure_class": failure.get("failure_class"),
                    "mined_tasks": len(mined),
                    "candidates": outcome.get("candidates", []),
                }
            )
    return {"base_task_id": task["task_id"], "results": results}
