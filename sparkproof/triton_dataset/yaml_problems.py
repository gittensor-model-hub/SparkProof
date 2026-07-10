"""Load TritonBench YAML problems as SparkProof prompt records."""

from __future__ import annotations

from pathlib import Path
import yaml

_LEVEL_DIRS = {
    1: "level1_basic",
    2: "level2_intermediate",
    3: "level3_advanced",
    4: "level4_expert",
}

SYSTEM = (
    "You are a Triton 3.7.1 GPU kernel expert for Blackwell SM12x. "
    "Output complete runnable Python: @triton.jit kernel, launcher, torch.allclose test. "
    "End execution successfully so validation can run the code."
)


def _problems_root(explicit: Path | None) -> Path | None:
    if explicit is not None and explicit.exists():
        return explicit
    sibling = Path(__file__).resolve().parents[2].parent / "SparkDistill" / "tritonbench" / "tritonbench" / "problems"
    if sibling.exists():
        return sibling
    return None


def iter_yaml_problem_prompts(
    *,
    problems_dir: Path | None = None,
    levels: list[int] | None = None,
    include_bugfix: bool = True,
) -> list[dict]:
    root = _problems_root(problems_dir)
    if root is None:
        return []

    levels = levels or [1]
    out: list[dict] = []

    for level in levels:
        level_dir = root / _LEVEL_DIRS.get(level, f"level{level}")
        if not level_dir.exists():
            continue
        for path in sorted(level_dir.glob("*.yaml")):
            prob = yaml.safe_load(path.read_text())
            out.append(_yaml_to_record(prob, path.stem, level))

    if include_bugfix:
        bugfix = root / "bugfix"
        if bugfix.exists():
            for path in sorted(bugfix.glob("*.yaml")):
                prob = yaml.safe_load(path.read_text())
                rec = _yaml_to_record(prob, path.stem, "bugfix")
                if prob.get("input_code"):
                    rec["prompt"] = prob["prompt"].strip() + "\n\n```python\n" + prob["input_code"].strip() + "\n```"
                if prob.get("expected_fix"):
                    rec["ground_truth_hint"] = prob["expected_fix"]
                out.append(rec)

    return out


def _yaml_to_record(prob: dict, stem: str, level: object) -> dict:
    return {
        "task_id": prob.get("id", stem),
        "source": "yaml",
        "category": prob.get("category", "kernel_generation"),
        "prompt": prob["prompt"].strip(),
        "system": SYSTEM,
        "level": level,
        "title": prob.get("title", stem),
    }
