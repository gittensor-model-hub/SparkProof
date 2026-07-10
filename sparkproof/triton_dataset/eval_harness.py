"""TritonBench eval harness — isolated from training dataset paths."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _tritonbench_root() -> Path | None:
    sibling = Path(__file__).resolve().parents[2].parent / "SparkDistill" / "tritonbench"
    if sibling.exists():
        return sibling
    return None


def _collect_environment() -> dict[str, str]:
    env: dict[str, str] = {"gpu": "NVIDIA Blackwell", "triton": "3.7.1"}
    try:
        import torch

        if torch.cuda.is_available():
            env["compute_capability"] = f"{torch.cuda.get_device_capability(0)[0]}.{torch.cuda.get_device_capability(0)[1]}"
            env["cuda"] = torch.version.cuda or "unknown"
        env["pytorch"] = torch.__version__
    except ImportError:
        pass
    return env


def normalize_report(raw: dict[str, Any], *, run_id: str, model_name: str, split: str = "test") -> dict[str, Any]:
    summary = raw.get("summary") or {}
    details = raw.get("details") or []
    by_category: dict[str, dict[str, float]] = {}
    for row in details:
        cat = row.get("category") or row.get("id", "unknown").split("_")[0]
        by_category.setdefault(cat, {"count": 0, "correctness_sum": 0.0, "exec_pass": 0})
        by_category[cat]["count"] += 1
        by_category[cat]["correctness_sum"] += float(row.get("correctness", 0.0))
        by_category[cat]["exec_pass"] += int(bool(row.get("exec_pass")))

    for cat, data in by_category.items():
        n = data["count"] or 1
        data["correctness_rate"] = data["correctness_sum"] / n
        data["exec_pass_rate"] = data["exec_pass"] / n
        del data["correctness_sum"]
        del data["exec_pass"]

    n = raw.get("num_problems") or len(details) or 1
    return {
        "run_id": run_id,
        "model": model_name,
        "split": split,
        "timestamp": raw.get("timestamp") or datetime.utcnow().isoformat() + "Z",
        "environment": _collect_environment(),
        "metrics": {
            "tasks_total": n,
            "response_rate": 1.0,
            "parse_rate": summary.get("syntax_pass_rate", 0.0),
            "compile_rate": summary.get("exec_pass_rate", 0.0),
            "runtime_rate": summary.get("exec_pass_rate", 0.0),
            "correctness_rate": summary.get("avg_correctness", 0.0),
            "performance_pass_rate": summary.get("avg_perf_awareness", 0.0),
            "avg_composite": summary.get("avg_composite", 0.0),
        },
        "by_level": raw.get("by_level") or {},
        "by_category": by_category,
        "raw": raw,
    }


class TritonBenchHarness:
    """Invoke TritonBench runner; write eval JSON outside training dirs."""

    def __init__(self, bench_root: Path | None = None) -> None:
        self.bench_root = bench_root or _tritonbench_root()

    def run_eval_cycle(
        self,
        *,
        endpoint: str,
        model_name: str,
        out_path: Path,
        config: str = "configs/eval_quick.yaml",
        levels: list[int] | None = None,
    ) -> dict[str, Any]:
        if self.bench_root is None or not self.bench_root.exists():
            return {"status": "error", "message": "TritonBench root not found (expected SparkDistill/tritonbench)"}

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        run_id = f"tb_{datetime.utcnow():%Y%m%d_%H%M%S}"

        cmd = [
            sys.executable,
            "-m",
            "tritonbench.cli",
            "eval",
            "--config",
            config,
            "--endpoint",
            endpoint,
            "--model",
            model_name,
            "--output",
            str(out_path.parent / "_tb_run"),
        ]
        if levels:
            cmd.extend(["--levels", *[str(x) for x in levels]])

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(self.bench_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                return {
                    "status": "error",
                    "message": proc.stderr[-4000:] or proc.stdout[-4000:] or f"exit {proc.returncode}",
                }
        except OSError as exc:
            return {"status": "error", "message": str(exc)}

        raw_reports = sorted((out_path.parent / "_tb_run").glob("tritonbench_*.json"))
        if not raw_reports:
            return {"status": "error", "message": "TritonBench produced no results JSON"}

        raw = json.loads(raw_reports[-1].read_text())
        report = normalize_report(raw, run_id=run_id, model_name=model_name)
        out_path.write_text(json.dumps(report, indent=2))
        return report
