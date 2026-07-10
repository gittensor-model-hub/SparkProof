"""TritonBench YAML problems — EVAL ONLY (never training prompts)."""

from __future__ import annotations

from pathlib import Path

from sparkproof.triton_dataset.yaml_problems import iter_yaml_problem_prompts as _iter_yaml


def iter_eval_problem_prompts(*, problems_dir: Path | None = None) -> list[dict]:
    records = _iter_yaml(problems_dir=problems_dir)
    for rec in records:
        rec["origin"] = "tritonbench"
        rec["split"] = "eval"
        rec["source"] = "tritonbench"
    return records
