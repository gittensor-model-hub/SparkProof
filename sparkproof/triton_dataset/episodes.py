"""Multi-turn training episodes: attempt → fail → critique → fix → optimize → accept.

A single Prompt→Answer row teaches little. An episode that records real validator
failures, teacher repairs, and measured optimization is dramatically more valuable
for SparkDistill SFT.
"""

from __future__ import annotations

from typing import Any

from sparkproof.triton_dataset.training_cot import (
    normalize_training_reasoning,
    prose_rationale_from_response,
)

EPISODE_VERSION = "sparkproof-episode-v1"


def _validation_tail(validation: dict[str, Any]) -> str:
    stages = validation.get("stages") or {}
    for stage in ("compile_execute", "syntax", "triton_api", "anti_cheat"):
        detail = stages.get(stage) or {}
        tail = detail.get("output_tail") or detail.get("stderr") or ""
        if tail:
            return str(tail)[-1500:]
    return ""


def validator_feedback_content(validation: dict[str, Any]) -> str:
    """User-turn text injected after a failed attempt (hardware critique)."""
    fail = validation.get("fail_reason") or "unknown"
    tail = _validation_tail(validation)
    lines = [
        "[sparkproof-validator]",
        f"Status: FAILED ({fail})",
        "Your previous answer did not pass hardware validation.",
        "Diagnose the root cause, then return a complete corrected runnable Python",
        "kernel + launcher + torch.allclose test.",
    ]
    if tail:
        lines.extend(["", "Trace tail:", tail])
    return "\n".join(lines)


def optimize_feedback_content(
    *,
    original_task: str,
    code: str,
    validation: dict[str, Any],
) -> str:
    """User-turn text requesting a measured optimization pass after a correct kernel."""
    bench = validation.get("benchmark") or {}
    speed = bench.get("normalized_speedup")
    composite = bench.get("composite_score")
    metrics = []
    if speed is not None:
        metrics.append(f"normalized_speedup={speed}")
    if composite is not None:
        metrics.append(f"composite_score={composite}")
    metric_line = ", ".join(metrics) if metrics else "correctness=passed (no speedup metric)"
    return (
        "[sparkproof-optimizer]\n"
        "The following Triton 3.7.1 kernel PASSED hardware validation and is correct.\n"
        f"Measured: {metric_line}\n\n"
        "Produce an OPTIMIZED complete runnable Python solution (kernel + launcher + "
        "torch.allclose) that remains numerically correct and aims for higher throughput "
        "(tiling, num_warps/num_stages, fusion, memory coalescing). Explain the "
        "engineering rationale, then return the full code.\n\n"
        f"## Original task\n{original_task}\n\n"
        f"## Baseline verified solution\n```python\n{code}\n```\n"
    )


def assistant_content_from_record(record: dict[str, Any]) -> str:
    """Build assistant turn text with optional ``<think>`` rationale."""
    response = (record.get("response") or "").strip()
    reasoning = normalize_training_reasoning(record.get("reasoning"))
    if not reasoning:
        reasoning = prose_rationale_from_response(response)
    if reasoning:
        # Avoid duplicating prose already sitting before a code fence.
        body = response
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{body}"
    return response


def _turn(
    *,
    role: str,
    kind: str,
    content: str,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {"role": role, "kind": kind, "content": content}
    row.update({k: v for k, v in extra.items() if v is not None})
    return row


def build_episode(
    *,
    task_prompt: str,
    system: str | None,
    provider: str,
    turns: list[dict[str, Any]],
    accepted: bool,
    repairs_used: int = 0,
    optimize_used: bool = False,
    optimize_improved: bool = False,
    final_speedup: float | None = None,
) -> dict[str, Any]:
    return {
        "version": EPISODE_VERSION,
        "task_prompt": task_prompt,
        "system": system,
        "provider": provider,
        "turns": turns,
        "accepted": accepted,
        "repairs_used": repairs_used,
        "optimize_used": optimize_used,
        "optimize_improved": optimize_improved,
        "final_speedup": final_speedup,
        "attempt_count": sum(1 for t in turns if t.get("role") == "assistant"),
    }


def stamp_episode(record: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    """Attach episode metadata; keep leaf-hash fields on the final accepted answer."""
    stamped = dict(record)
    stamped["prompt"] = episode["task_prompt"]
    meta = dict(stamped.get("metadata") or {})
    meta["episode"] = episode
    meta["episode_version"] = EPISODE_VERSION
    meta["multi_turn"] = True
    if episode.get("optimize_improved"):
        meta["tier"] = "optimized"
    stamped["metadata"] = meta
    return stamped


def episode_to_messages(episode: dict[str, Any]) -> list[dict[str, str]]:
    """Convert an episode into chat messages for SFT."""
    system = episode.get("system") or (
        "You are a Triton 3.7.1 GPU kernel expert. Debug with real validator feedback "
        "and optimize only when measurements justify it."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for turn in episode.get("turns") or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    return messages


def trajectory_has_episode(trajectory: dict[str, Any]) -> bool:
    episode = (trajectory.get("metadata") or {}).get("episode")
    return isinstance(episode, dict) and bool(episode.get("turns"))
