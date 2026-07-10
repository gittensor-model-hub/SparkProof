"""TritonBench YAML problems — EVAL ONLY (never training prompts)."""

from __future__ import annotations

from pathlib import Path

from sparkproof.triton_dataset.yaml_problems import iter_yaml_problem_prompts as _iter_yaml


def iter_eval_problem_prompts(
    *,
    problems_dir: Path | None = None,
    levels: list[int] | None = None,
) -> list[dict]:
    """Load the complete held-out corpus unless explicit levels are requested."""
    records = _iter_yaml(problems_dir=problems_dir, levels=levels or [1, 2, 3, 4])
    for rec in records:
        rec["origin"] = "tritonbench"
        rec["split"] = "eval"
        rec["source"] = "tritonbench"
    return records
