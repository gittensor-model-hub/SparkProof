"""Summarize doc/generation bundle pass rates by source and task metadata."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prompt_meta(record: dict[str, Any]) -> dict[str, Any]:
    return (record.get("metadata") or {}).get("prompt_meta") or {}


def _task_key(record: dict[str, Any]) -> str:
    meta = _prompt_meta(record)
    return str(meta.get("task_id") or record.get("task_id") or "unknown")


def summarize_adjudication(adjudication: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(adjudication)
    winners = sum(1 for row in adjudication if row.get("winner_provider"))
    by_source: Counter[str] = Counter()
    win_by_source: Counter[str] = Counter()
    for row in adjudication:
        src = str(row.get("source") or "unknown")
        by_source[src] += 1
        if row.get("winner_provider"):
            win_by_source[src] += 1
    return {
        "prompts_run": total,
        "teacher_winners": winners,
        "teacher_pass_rate": winners / total if total else 0.0,
        "by_source": {k: {"run": by_source[k], "winners": win_by_source[k]} for k in sorted(by_source)},
    }


def summarize_validation(
    validation_rows: list[dict[str, Any]],
    *,
    prompts_by_task: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    total = len(validation_rows)
    passed = sum(1 for row in validation_rows if (row.get("validation") or {}).get("passed"))
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"run": 0, "passed": 0})
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"run": 0, "passed": 0})

    for row in validation_rows:
        task_id = str(row.get("task_id") or "")
        prompt = prompts_by_task.get(task_id, {})
        src = str(prompt.get("source") or row.get("source") or "unknown")
        cat = str(prompt.get("category") or row.get("category") or "unknown")
        ok = bool((row.get("validation") or {}).get("passed"))
        by_source[src]["run"] += 1
        by_category[cat]["run"] += 1
        if ok:
            by_source[src]["passed"] += 1
            by_category[cat]["passed"] += 1

    return {
        "validated": total,
        "proved_passed": passed,
        "prove_pass_rate": passed / total if total else 0.0,
        "by_source": dict(by_source),
        "by_category": dict(by_category),
    }


def summarize_bundle(bundle_dir: Path) -> dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    prompts = _load_jsonl(bundle_dir / "prompts.jsonl")
    prompts_by_task = {str(p.get("task_id")): p for p in prompts if p.get("task_id")}

    adjudication = _load_jsonl(bundle_dir / "adjudication.jsonl")
    for row in adjudication:
        tid = row.get("task_id")
        if tid and tid in prompts_by_task:
            row.setdefault("source", prompts_by_task[tid].get("source"))
            row.setdefault("category", prompts_by_task[tid].get("category"))

    verified = _load_jsonl(bundle_dir / "trajectories.jsonl")
    raw = _load_jsonl(bundle_dir / "trajectories_raw.jsonl")
    if not raw:
        raw = _load_jsonl(bundle_dir / "trajectories.jsonl")
    validation = _load_jsonl(bundle_dir / "validation_report.jsonl")

    if raw and validation and len(raw) == len(validation):
        for traj, val in zip(raw, validation):
            val["task_id"] = _task_key(traj)

    manifest: dict[str, Any] = {}
    manifest_path = bundle_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())

    out: dict[str, Any] = {
        "bundle": str(bundle_dir.resolve()),
        "prompts_in_bundle": len(prompts),
        "verified_trajectories": len(verified),
        "raw_trajectories": len(raw),
        "manifest_sample_count": manifest.get("sample_count"),
        "manifest_version": manifest.get("version"),
    }

    if adjudication:
        out["teacher"] = summarize_adjudication(adjudication)
    if validation:
        out["prove"] = summarize_validation(validation, prompts_by_task=prompts_by_task)

    if prompts and not adjudication and not validation:
        out["prompts_by_source"] = dict(Counter(p.get("source", "unknown") for p in prompts))
        out["prompts_by_category"] = dict(Counter(p.get("category", "unknown") for p in prompts))

    return out


def format_summary(report: dict[str, Any]) -> str:
    lines = [
        f"Bundle: {report.get('bundle')}",
        f"  prompts in bundle:     {report.get('prompts_in_bundle', 0)}",
        f"  verified trajectories: {report.get('verified_trajectories', 0)}",
        f"  manifest samples:      {report.get('manifest_sample_count', '—')} ({report.get('manifest_version', '—')})",
    ]
    teacher = report.get("teacher")
    if teacher:
        rate = teacher["teacher_pass_rate"]
        lines.append(
            f"  teacher winners:       {teacher['teacher_winners']}/{teacher['prompts_run']} ({rate:.1%})"
        )
        for src, stats in teacher.get("by_source", {}).items():
            w, r = stats["winners"], stats["run"]
            lines.append(f"    {src}: {w}/{r} ({(w / r if r else 0):.1%})")
    prove = report.get("prove")
    if prove:
        rate = prove["prove_pass_rate"]
        lines.append(
            f"  Blackwell proved:      {prove['proved_passed']}/{prove['validated']} ({rate:.1%})"
        )
        for src, stats in prove.get("by_source", {}).items():
            p, r = stats["passed"], stats["run"]
            lines.append(f"    {src}: {p}/{r} ({(p / r if r else 0):.1%})")
        lines.append("  by category:")
        for cat, stats in prove.get("by_category", {}).items():
            p, r = stats["passed"], stats["run"]
            lines.append(f"    {cat}: {p}/{r} ({(p / r if r else 0):.1%})")
    if report.get("prompts_by_source"):
        lines.append(f"  prompts_by_source: {report['prompts_by_source']}")
    return "\n".join(lines)
