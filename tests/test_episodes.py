"""Tests for multi-turn SparkProof episodes (fail→critique→fix→optimize)."""

from __future__ import annotations

from sparkproof.publish.hf_dataset import trajectory_to_messages_record
from sparkproof.triton_dataset.episodes import (
    EPISODE_VERSION,
    episode_to_messages,
    validator_feedback_content,
)
from sparkproof.triton_dataset.multi_candidate import generate_with_repair


PASS_CODE = "```python\nprint('SPARKPROOF_TRITON_PASS')\n```"
FAIL_CODE = "```python\nraise RuntimeError('boom')\n```"
OPT_CODE = "```python\nprint('SPARKPROOF_TRITON_PASS')\n# optimized\n```"


class _SeqValidator:
    def __init__(self, outcomes: list[dict]):
        self.outcomes = list(outcomes)
        self.calls = 0

    def validate_response(self, response, **kwargs):
        idx = min(self.calls, len(self.outcomes) - 1)
        self.calls += 1
        return self.outcomes[idx]


def test_validator_feedback_includes_failure():
    text = validator_feedback_content(
        {"fail_reason": "runtime_error", "stages": {"compile_execute": {"output_tail": "boom"}}}
    )
    assert "FAILED" in text
    assert "runtime_error" in text
    assert "boom" in text


def test_generate_with_repair_records_multi_turn_episode(monkeypatch):
    prompts: list[str] = []

    def fake_generate(*, prompt, provider, **kwargs):
        prompts.append(prompt)
        n = len(prompts)
        if n == 1:
            response = FAIL_CODE
        elif n == 2:
            response = PASS_CODE
        else:
            response = OPT_CODE
        return {
            "prompt": prompt,
            "response": response,
            "provider": provider,
            "model": "claude-fable-5",
            "reasoning": "step by step reasoning about tiling and masks for this kernel.",
            "request_sha256": f"req{n}",
            "response_sha256": f"resp{n}",
            "metadata": {"usage": {"completion_tokens": 10}},
        }

    monkeypatch.setattr(
        "sparkproof.triton_dataset.multi_candidate.generate_via_gateway",
        fake_generate,
    )

    validator = _SeqValidator(
        [
            {"passed": False, "fail_reason": "runtime_error", "stages": {"compile_execute": {"output_tail": "boom"}}},
            {
                "passed": True,
                "stages": {},
                "benchmark": {"correctness_pass_rate": 1.0, "normalized_speedup": 1.0, "composite_score": 0.6},
            },
            {
                "passed": True,
                "stages": {},
                "benchmark": {"correctness_pass_rate": 1.0, "normalized_speedup": 2.7, "composite_score": 0.8},
            },
        ]
    )

    result = generate_with_repair(
        gateway="openrouter",
        api_key="key",
        provider="anthropic",
        prompt="write a fused add kernel",
        system="sys",
        max_tokens=1024,
        temperature=0.7,
        max_repairs=2,
        validator=validator,
        run_benchmark=True,
        record_episode=True,
        enable_optimize=True,
    )
    assert result is not None
    assert result.validation.get("passed") is True
    episode = result.record["metadata"]["episode"]
    assert episode["version"] == EPISODE_VERSION
    assert episode["accepted"] is True
    assert episode["repairs_used"] == 1
    assert episode["optimize_used"] is True
    assert episode["optimize_improved"] is True
    assert result.record["metadata"]["tier"] == "optimized"
    assert result.record["prompt"] == "write a fused add kernel"

    kinds = [t["kind"] for t in episode["turns"]]
    assert kinds[0] == "task"
    assert "attempt" in kinds
    assert "validator_feedback" in kinds
    assert "repair" in kinds
    assert "optimize_feedback" in kinds
    assert "optimize" in kinds

    messages = episode_to_messages(episode)
    assert messages[0]["role"] == "system"
    roles = [m["role"] for m in messages[1:]]
    assert roles[0] == "user"
    assert "assistant" in roles
    assert any("<think>" in m["content"] for m in messages if m["role"] == "assistant")

    sft = trajectory_to_messages_record(result.record)
    assert sft is not None
    assert sft["metadata"]["multi_turn"] is True
    assert len(sft["messages"]) >= 5  # system + task + attempt + feedback + repair (+ opt…)


def test_episode_disabled_keeps_single_turn_export(monkeypatch):
    def fake_generate(*, prompt, provider, **kwargs):
        return {
            "prompt": prompt,
            "response": PASS_CODE,
            "provider": provider,
            "model": "claude-fable-5",
            "reasoning": "short rationale text for the kernel.",
            "metadata": {},
        }

    monkeypatch.setattr(
        "sparkproof.triton_dataset.multi_candidate.generate_via_gateway",
        fake_generate,
    )

    class Ok:
        def validate_response(self, response, **kwargs):
            return {"passed": True, "stages": {}}

    result = generate_with_repair(
        gateway="openrouter",
        api_key="key",
        provider="anthropic",
        prompt="task",
        system=None,
        max_tokens=512,
        temperature=0.7,
        max_repairs=0,
        validator=Ok(),
        run_benchmark=False,
        record_episode=False,
        enable_optimize=False,
    )
    assert result is not None
    assert "episode" not in (result.record.get("metadata") or {})
    sft = trajectory_to_messages_record(result.record)
    assert sft is not None
    assert len(sft["messages"]) == 3
