"""Best-of-N teacher generation + self-repair on Blackwell."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sparkproof.generate.gateway_client import generate_via_gateway
from sparkproof.policy import SUPPORTED_PROVIDERS
from sparkproof.triton.validator import TritonKernelValidator


@dataclass(frozen=True)
class CandidateResult:
    provider: str
    record: dict[str, Any]
    validation: dict[str, Any]
    repairs_used: int
    score: float


def _client_value(client: Any, name: str, default: Any = None) -> Any:
    if isinstance(client, dict):
        return client.get(name, default)
    return getattr(client, name, default)


def _benchmark_score(validation: dict[str, Any]) -> float:
    bench = validation.get("benchmark") or {}
    if bench:
        return float(bench.get("composite_score", 0.0))
    return 1.0 if validation.get("passed") else 0.0


def acceptance_score(validation: dict[str, Any], *, output_tokens: int = 0) -> float:
    bench = validation.get("benchmark") or {}
    correctness = float(bench.get("correctness_pass_rate", bench.get("composite_score", 0.0)))
    if validation.get("passed") and not bench:
        correctness = 1.0
    compile_pass = 1.0 if validation.get("passed") or validation.get("fail_reason") != "syntax_error" else 0.0
    # Only a harness-controlled metric may populate normalized_speedup.
    # Candidate-controlled/self-reported timing diagnostics are intentionally
    # ignored here.
    speed = float(bench.get("normalized_speedup", 0.0)) if bench else 0.0
    return 100.0 * correctness + 10.0 * compile_pass + 5.0 * speed - 0.01 * output_tokens


def assign_tier(validation: dict[str, Any], *, repairs_used: int = 0) -> str:
    if not validation.get("passed"):
        return "reject"
    if repairs_used > 0:
        return "repair"
    bench = validation.get("benchmark") or {}
    if bench and bench.get("composite_score", 0.0) >= 0.5:
        return "gold"
    return "silver"


def _repair_prompt(broken_response: str, validation: dict[str, Any]) -> str:
    stages = validation.get("stages", {})
    tail = ""
    for stage in ("compile_execute", "syntax", "triton_api"):
        detail = stages.get(stage, {})
        if detail.get("output_tail"):
            tail = detail["output_tail"]
            break
    fail = validation.get("fail_reason", "unknown")
    return (
        "Your prior Triton 3.7.1 answer failed hardware validation.\n"
        f"Failure: {fail}\n"
        f"Trace tail:\n{tail[-1500:]}\n\n"
        "Return corrected **complete runnable Python** (kernel + launcher + torch.allclose test).\n\n"
        f"```python\n{extract_code(broken_response)}\n```"
    )


def extract_code(response: str) -> str:
    for pattern in (r"```python\n(.*?)```", r"```\n(.*?)```"):
        matches = re.findall(pattern, response, re.DOTALL)
        if matches:
            return "\n\n".join(matches)
    return response


def generate_with_repair(
    *,
    gateway: str,
    api_key: str,
    provider: str,
    prompt: str,
    system: str | None,
    max_tokens: int,
    temperature: float,
    max_repairs: int,
    validator: TritonKernelValidator,
    run_benchmark: bool,
    strict_validate: bool = False,
    capture_ir: bool = False,
    prompt_meta: dict[str, Any] | None = None,
) -> CandidateResult | None:
    repair_user = prompt
    repairs = 0
    last_record: dict[str, Any] | None = None
    last_validation: dict[str, Any] | None = None

    for attempt in range(max_repairs + 1):
        record = generate_via_gateway(
            gateway=gateway,
            api_key=api_key,
            provider=provider,
            prompt=repair_user,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if prompt_meta:
            record.setdefault("metadata", {})
            record["metadata"]["prompt_meta"] = prompt_meta

        validation = validator.validate_response(
            record["response"],
            run_benchmark=run_benchmark,
            strict=strict_validate,
            capture_ir=capture_ir,
            prompt_meta=prompt_meta,
        )
        last_record, last_validation = record, validation
        if validation["passed"]:
            stamped = dict(record)
            stamped["sparkproof_validation"] = validation
            tier = assign_tier(validation, repairs_used=repairs)
            stamped.setdefault("metadata", {})
            stamped["metadata"]["tier"] = tier
            usage = (record.get("metadata") or {}).get("usage") or {}
            output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
            score = acceptance_score(validation, output_tokens=output_tokens)
            return CandidateResult(
                provider=provider,
                record=stamped,
                validation=validation,
                repairs_used=repairs,
                score=score,
            )
        if attempt < max_repairs:
            repairs += 1
            repair_user = _repair_prompt(record["response"], validation)

    if last_record is None or last_validation is None:
        return None
    return CandidateResult(
        provider=provider,
        record=last_record,
        validation=last_validation,
        repairs_used=repairs,
        score=0.0,
    )


def generate_best_of_n(
    *,
    gateway: str,
    api_key: str,
    prompt_record: dict[str, Any],
    providers: list[str],
    max_tokens: int = 2048,
    temperature: float = 0.7,
    max_repairs: int = 2,
    gpu_index: int = 0,
    run_benchmark: bool = False,
    strict_validate: bool = False,
    capture_ir: bool = False,
    validator: TritonKernelValidator | None = None,
    recover_training_cot: bool = True,
) -> tuple[CandidateResult | None, list[CandidateResult]]:
    unknown = set(providers) - SUPPORTED_PROVIDERS
    if unknown:
        raise ValueError(f"unsupported providers {sorted(unknown)}")

    validator = validator or TritonKernelValidator(gpu_index=gpu_index)
    prompt = prompt_record["prompt"]
    system = prompt_record.get("system")
    # Keep the original (pre-repair) prompt in metadata so checkpoint-based
    # recovery (dpo_export.enrich_adjudication_with_responses) can backfill
    # it later; per-attempt repair prompts never overwrite this copy.
    meta = {k: prompt_record[k] for k in prompt_record if k != "system"}

    candidates: list[CandidateResult] = []
    for provider in providers:
        result = generate_with_repair(
            gateway=gateway,
            api_key=api_key,
            provider=provider,
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            max_repairs=max_repairs,
            validator=validator,
            run_benchmark=run_benchmark,
            strict_validate=strict_validate,
            capture_ir=capture_ir,
            prompt_meta=meta,
        )
        if result is not None:
            candidates.append(result)

    winners = [c for c in candidates if c.validation.get("passed")]
    winners.sort(key=lambda c: c.score, reverse=True)
    winner = winners[0] if winners else None

    # When GPT Sol wins with encrypted/empty CoT, recover a train-able Fable
    # rationale (re-solve + validate, else explain + keep Sol gold answer).
    if winner is not None and recover_training_cot:
        from sparkproof.triton_dataset.training_cot import recover_openai_winner_cot

        winner = recover_openai_winner_cot(
            winner,
            gateway=gateway,
            api_key=api_key,
            original_prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            validator=validator,
            run_benchmark=run_benchmark,
            strict_validate=strict_validate,
            capture_ir=capture_ir,
        )
    return winner, candidates


def generate_best_candidate(
    prompt_record: dict[str, Any],
    *,
    client: Any,
    validator: TritonKernelValidator | None = None,
    run_benchmark: bool = False,
    strict_validate: bool = False,
    capture_ir: bool = False,
) -> dict[str, Any]:
    """Orchestrator-friendly wrapper around best-of-N generation."""
    gateway = _client_value(client, "gateway")
    api_key = _client_value(client, "api_key")
    providers = _client_value(client, "providers", ["anthropic", "openai"])
    gpu_index = _client_value(client, "gpu_index", 0)
    max_tokens = _client_value(client, "max_tokens", 4096)
    max_repairs = _client_value(client, "max_repairs", 2)
    temperature = _client_value(client, "temperature", 0.7)

    strict_validate = strict_validate or bool(_client_value(client, "strict_validate", False))
    capture_ir = capture_ir or bool(_client_value(client, "capture_ir", False))

    recover_training_cot = bool(_client_value(client, "recover_training_cot", True))
    winner, all_candidates = generate_best_of_n(
        gateway=gateway,
        api_key=api_key,
        prompt_record=prompt_record,
        providers=list(providers),
        max_tokens=max_tokens,
        temperature=temperature,
        max_repairs=max_repairs,
        gpu_index=gpu_index,
        run_benchmark=run_benchmark,
        strict_validate=strict_validate,
        capture_ir=capture_ir,
        validator=validator,
        recover_training_cot=recover_training_cot,
    )
    candidate_rows = [
        {
            "provider": candidate.provider,
            "passed": candidate.validation.get("passed", False),
            "score": candidate.score,
            "repairs_used": candidate.repairs_used,
            "validation": candidate.validation,
            "response": candidate.record.get("response", ""),
        }
        for candidate in all_candidates
    ]
    if winner is not None:
        return {
            "passed": True,
            "response": winner.record["response"],
            "validation": winner.validation,
            "tier": winner.record.get("metadata", {}).get("tier", "gold"),
            "trajectory": winner.record,
            "provider": winner.provider,
            "candidates": candidate_rows,
        }
    failed = all_candidates[0] if all_candidates else None
    return {
        "passed": False,
        "response": failed.record["response"] if failed else "",
        "validation": failed.validation if failed else {"passed": False, "fail_reason": "no_candidates"},
        "tier": "reject",
        "provider": failed.provider if failed else None,
        "candidates": candidate_rows,
    }
