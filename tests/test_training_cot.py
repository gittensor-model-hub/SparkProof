"""Tests for Sol→Fable training-CoT recovery."""

from __future__ import annotations

import json

from sparkproof.publish.hf_dataset import trajectory_to_messages_record
from sparkproof.teacher_request import generation_config, verify_request_sha256
from sparkproof.triton_dataset.multi_candidate import CandidateResult, generate_best_of_n
from sparkproof.triton_dataset.training_cot import (
    has_usable_training_reasoning,
    normalize_training_reasoning,
    recover_openai_winner_cot,
)
from tests.conftest_helpers import gateway_record_from_prompt, gateway_trajectory_fields


ENCRYPTED_DETAILS = json.dumps(
    [{"type": "reasoning.encrypted", "data": "abc123encryptedblob"}],
    ensure_ascii=False,
)

FABLE_REASONING = (
    "Use a 1D grid over the vector length, load with a BLOCK-sized mask, "
    "add elementwise, and store under the same mask so tails stay correct."
)

SOL_KERNEL = """```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask), mask=mask)

def launch(x, y):
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(x.numel(), 128),)](x, y, out, x.numel(), BLOCK=128)
    return out
print("SPARKPROOF_TRITON_PASS")
```"""

FABLE_RESOLVE_RESPONSE = (
    "Tile along the leading dimension with masked loads/stores.\n\n" + SOL_KERNEL
)


def test_encrypted_reasoning_is_not_usable_for_training():
    assert has_usable_training_reasoning(None) is False
    assert has_usable_training_reasoning(ENCRYPTED_DETAILS) is False
    assert normalize_training_reasoning(ENCRYPTED_DETAILS) is None
    assert has_usable_training_reasoning(FABLE_REASONING) is True


def test_sft_export_skips_encrypted_reasoning_json():
    record = trajectory_to_messages_record(
        {
            "prompt": "write add kernel",
            "response": SOL_KERNEL,
            "provider": "openai",
            "model": "gpt-5.6",
            "reasoning": ENCRYPTED_DETAILS,
            "sparkproof_validation": {"passed": True},
            "metadata": {"prompt_meta": {"prompt": "write add kernel", "task_id": "t1"}},
        }
    )
    assert record is not None
    assistant = record["messages"][2]["content"]
    assert "<think>" not in assistant
    assert "encrypted" not in assistant


def test_recover_fable_resolve_replaces_sol_winner(monkeypatch):
    calls: list[str] = []

    def fake_generate(*, provider, prompt, max_tokens=2048, temperature=0.7, system=None, **kwargs):
        calls.append(provider)
        assert provider == "anthropic"
        assert "Reference verified solution" in prompt
        return gateway_record_from_prompt(
            gateway="openrouter",
            provider=provider,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            model="claude-fable-5",
            response=FABLE_RESOLVE_RESPONSE,
            reasoning=FABLE_REASONING,
            metadata={"usage": {"completion_tokens": 100}},
        )

    class FakeValidator:
        def validate_response(self, response, **kwargs):
            return {"passed": True, "stages": {}, "benchmark": {"correctness_pass_rate": 1.0}}

    monkeypatch.setattr(
        "sparkproof.triton_dataset.training_cot.generate_via_gateway",
        fake_generate,
    )

    sol_winner = CandidateResult(
        provider="openai",
        record={
            "prompt": "write add kernel",
            "response": SOL_KERNEL,
            "model": "gpt-5.6",
            "reasoning": ENCRYPTED_DETAILS,
            "metadata": {"prompt_meta": {"prompt": "write add kernel", "task_id": "t1"}},
            **gateway_trajectory_fields("openai"),
        },
        validation={"passed": True, "stages": {}},
        repairs_used=0,
        score=110.0,
    )
    recovered = recover_openai_winner_cot(
        sol_winner,
        gateway="openrouter",
        api_key="key",
        original_prompt="write add kernel",
        system=None,
        max_tokens=2048,
        temperature=0.7,
        validator=FakeValidator(),
        run_benchmark=False,
        strict_validate=False,
        capture_ir=False,
    )
    assert calls == ["anthropic"]
    assert recovered.provider == "anthropic"
    assert recovered.record["metadata"]["cot_recovery"] == "fable_resolve"
    assert recovered.record["prompt"] == "write add kernel"
    assert has_usable_training_reasoning(recovered.record["reasoning"])
    sft = trajectory_to_messages_record(recovered.record)
    assert sft is not None
    assert "<think>" in sft["messages"][2]["content"]
    assert FABLE_REASONING[:40] in sft["messages"][2]["content"]
    verify_request_sha256(
        recovered.record,
        generation_config(max_tokens=2048, temperature=0.7),
    )
    assert recovered.record["metadata"].get("cot_request_sha256")


def test_recover_fable_explain_keeps_sol_code_when_resolve_fails(monkeypatch):
    calls: list[str] = []

    def fake_generate(*, provider, prompt, max_tokens=2048, temperature=0.7, system=None, **kwargs):
        calls.append("resolve" if "Reference verified" in prompt else "explain")
        if "Reference verified" in prompt:
            return gateway_record_from_prompt(
                gateway="yunwu",
                provider=provider,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                model="claude-fable-5",
                response="sorry I failed\n```python\nraise RuntimeError('nope')\n```",
                reasoning=FABLE_REASONING,
            )
        return gateway_record_from_prompt(
            gateway="yunwu",
            provider=provider,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            model="claude-fable-5",
            response="Detailed rationale about masks and grids.\n\n" + SOL_KERNEL,
            reasoning=FABLE_REASONING,
        )

    class FakeValidator:
        def validate_response(self, response, **kwargs):
            if "raise RuntimeError" in response:
                return {"passed": False, "fail_reason": "runtime_error", "stages": {}}
            return {"passed": True, "stages": {}}

    monkeypatch.setattr(
        "sparkproof.triton_dataset.training_cot.generate_via_gateway",
        fake_generate,
    )

    sol_winner = CandidateResult(
        provider="openai",
        record={
            "prompt": "write add kernel",
            "response": SOL_KERNEL,
            "model": "gpt-5.6",
            "reasoning": ENCRYPTED_DETAILS,
            "metadata": {"prompt_meta": {"prompt": "write add kernel"}},
            "sparkproof_validation": {"passed": True},
            **gateway_trajectory_fields("openai", gateway="yunwu"),
        },
        validation={"passed": True, "stages": {}},
        repairs_used=0,
        score=110.0,
    )
    recovered = recover_openai_winner_cot(
        sol_winner,
        gateway="yunwu",
        api_key="key",
        original_prompt="write add kernel",
        system=None,
        max_tokens=2048,
        temperature=0.7,
        validator=FakeValidator(),
        run_benchmark=False,
        strict_validate=False,
        capture_ir=False,
    )
    assert calls == ["resolve", "explain"]
    assert recovered.provider == "openai"
    assert recovered.record["response"] == SOL_KERNEL
    assert recovered.record["metadata"]["cot_recovery"] == "fable_explain"
    assert recovered.record["metadata"]["cot_provider"] == "anthropic"
    assert recovered.record["reasoning"] == FABLE_REASONING
    verify_request_sha256(
        recovered.record,
        generation_config(max_tokens=2048, temperature=0.7),
    )


def test_generate_best_of_n_recovers_cot_when_sol_wins(monkeypatch):
    def fake_generate(*, provider, prompt, gateway="openrouter", max_tokens=2048, temperature=0.7, system=None, **kwargs):
        if provider == "openai":
            return gateway_record_from_prompt(
                gateway=gateway,
                provider=provider,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                model="gpt-5.6",
                response=SOL_KERNEL,
                reasoning=ENCRYPTED_DETAILS,
                metadata={"usage": {"completion_tokens": 10}},
            )
        if "Reference verified" in prompt or "VERIFIED correct" in prompt:
            return gateway_record_from_prompt(
                gateway=gateway,
                provider=provider,
                prompt=prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                model="claude-fable-5",
                response=FABLE_RESOLVE_RESPONSE,
                reasoning=FABLE_REASONING,
                metadata={"usage": {"completion_tokens": 20}},
            )
        return gateway_record_from_prompt(
            gateway=gateway,
            provider=provider,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            model="claude-fable-5",
            response="```python\nraise RuntimeError('bad')\n```",
            reasoning=FABLE_REASONING,
            metadata={"usage": {"completion_tokens": 5}},
        )

    class FakeValidator:
        def validate_response(self, response, **kwargs):
            if "raise RuntimeError" in response:
                return {"passed": False, "fail_reason": "runtime_error", "stages": {}}
            return {
                "passed": True,
                "stages": {},
                "benchmark": {"correctness_pass_rate": 1.0, "normalized_speedup": 1.0},
            }

    monkeypatch.setattr(
        "sparkproof.triton_dataset.multi_candidate.generate_via_gateway",
        fake_generate,
    )
    monkeypatch.setattr(
        "sparkproof.triton_dataset.training_cot.generate_via_gateway",
        fake_generate,
    )

    winner, candidates = generate_best_of_n(
        gateway="openrouter",
        api_key="key",
        prompt_record={"task_id": "t1", "prompt": "original task", "source": "torch_op"},
        providers=["anthropic", "openai"],
        validator=FakeValidator(),
    )
    assert len(candidates) == 2
    assert winner is not None
    assert winner.provider == "anthropic"
    assert winner.record["metadata"]["cot_recovery"] == "fable_resolve"
    assert winner.record["prompt"] == "original task"
