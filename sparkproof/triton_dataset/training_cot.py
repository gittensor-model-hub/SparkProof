"""Recover train-able CoT when GPT 5.6 Sol wins with encrypted/empty reasoning.

GPT Sol often returns only encrypted ``reasoning_details``, which cannot be used
for distillation. Best practice for SparkProof datasets:

1. **Re-solve** with Claude Fable 5, hinted by Sol's verified kernel, then
   re-validate on GPU. If Fable passes with plaintext reasoning, prefer that
   trajectory for SFT.
2. **Explain fallback**: if re-solve fails, ask Fable for engineering rationale
   while keeping Sol's verified response as the gold answer, and attach Fable's
   plaintext CoT under ``reasoning`` (with honest provenance metadata).

This does **not** decrypt Sol's private CoT — it produces a Fable-authored
inspectable rationale suitable for student training.
"""

from __future__ import annotations

import json
import re
from typing import Any

from sparkproof.generate.gateway_client import generate_via_gateway
from sparkproof.teacher_request import rebind_leaf_prompt
from sparkproof.triton.validator import TritonKernelValidator


COT_PROVIDER = "anthropic"
_MIN_REASONING_CHARS = 32
_MIN_PLAINTEXT_CHARS = 8


def extract_plaintext_reasoning_details(details: Any) -> str | None:
    """Pull usable text from OpenRouter/yunwu ``reasoning_details`` (skip encrypted)."""
    if not isinstance(details, list):
        return None
    chunks: list[str] = []
    for item in details:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type") or "")
        if "encrypted" in kind:
            continue
        if kind == "reasoning.text":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                chunks.append(text.strip())
        elif kind == "reasoning.summary":
            summary = item.get("summary") or item.get("text")
            if isinstance(summary, str) and summary.strip():
                chunks.append(summary.strip())
    joined = "\n\n".join(chunks).strip()
    return joined or None


def has_usable_training_reasoning(reasoning: Any) -> bool:
    """True when ``reasoning`` is plaintext suitable for ``<think>`` SFT tags."""
    return normalize_training_reasoning(reasoning) is not None


def normalize_training_reasoning(reasoning: Any) -> str | None:
    """Return plaintext CoT or None (drops encrypted JSON dumps)."""
    if not isinstance(reasoning, str) or not reasoning.strip():
        return None
    text = reasoning.strip()
    if text[0] in "[{":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return text if len(text) >= _MIN_PLAINTEXT_CHARS else None
        extracted = extract_plaintext_reasoning_details(parsed)
        if extracted and len(extracted) >= _MIN_REASONING_CHARS:
            return extracted
        return None
    return text if len(text) >= _MIN_PLAINTEXT_CHARS else None


def prose_rationale_from_response(response: str) -> str | None:
    """Use prose before the first fenced code block as a fallback rationale."""
    if not response:
        return None
    before = re.split(r"```(?:python)?\n", response, maxsplit=1)[0].strip()
    if len(before) >= _MIN_REASONING_CHARS:
        return before
    return None


def resolve_cot_prompt(*, original_prompt: str, verified_code: str) -> str:
    return (
        "A verified-correct Triton 3.7.1 solution for the task below already exists.\n"
        "Produce YOUR OWN complete solution with inspectable engineering rationale "
        "(decomposition/grid, tile selection, pointer and stride equations, masking, "
        "accumulation precision, expected bottleneck), then the complete runnable "
        "Python (kernel + launcher + torch.allclose test).\n"
        "You may use the reference as a guide, but return a full self-contained answer.\n\n"
        f"## Task\n{original_prompt}\n\n"
        f"## Reference verified solution\n```python\n{verified_code}\n```\n"
    )


def explain_cot_prompt(*, original_prompt: str, verified_code: str) -> str:
    return (
        "The following Triton 3.7.1 solution is VERIFIED correct on hardware.\n"
        "Write a detailed engineering rationale for HOW and WHY this kernel works "
        "(decomposition/grid, tiles, strides, masks, precision, bottleneck).\n"
        "Then return the COMPLETE same runnable Python solution unchanged "
        "(kernel + launcher + torch.allclose test). Do not substitute a different algorithm.\n\n"
        f"## Task\n{original_prompt}\n\n"
        f"## Verified solution\n```python\n{verified_code}\n```\n"
    )


def _original_task_prompt(record: dict[str, Any], fallback: str) -> str:
    meta = record.get("metadata") or {}
    prompt_meta = meta.get("prompt_meta") or {}
    return str(prompt_meta.get("prompt") or fallback)


def _stamp_cot_meta(
    record: dict[str, Any],
    *,
    mode: str,
    cot_record: dict[str, Any],
    sol_provider: str,
) -> dict[str, Any]:
    stamped = dict(record)
    meta = dict(stamped.get("metadata") or {})
    meta["cot_recovery"] = mode
    meta["cot_provider"] = cot_record.get("provider") or COT_PROVIDER
    meta["cot_model"] = cot_record.get("model")
    meta["sol_winner_provider"] = sol_provider
    if cot_record.get("request_sha256") and "cot_request_sha256" not in meta:
        meta["cot_request_sha256"] = cot_record["request_sha256"]
    if cot_record.get("response_sha256"):
        meta["cot_response_sha256"] = cot_record["response_sha256"]
    stamped["metadata"] = meta
    return stamped


def recover_openai_winner_cot(
    winner: Any,
    *,
    gateway: str,
    api_key: str,
    original_prompt: str,
    system: str | None,
    max_tokens: int,
    temperature: float,
    validator: TritonKernelValidator,
    run_benchmark: bool,
    strict_validate: bool,
    capture_ir: bool,
) -> Any:
    """If an OpenAI winner lacks usable CoT, recover via Fable re-solve / explain.

    Returns a ``CandidateResult`` (imported lazily to avoid import cycles).
    """
    from sparkproof.triton_dataset.multi_candidate import (
        CandidateResult,
        acceptance_score,
        assign_tier,
        extract_code,
    )

    if winner.provider != "openai":
        return winner
    if has_usable_training_reasoning(winner.record.get("reasoning")):
        normalized = normalize_training_reasoning(winner.record.get("reasoning"))
        if normalized and normalized != winner.record.get("reasoning"):
            record = dict(winner.record)
            record["reasoning"] = normalized
            return CandidateResult(
                provider=winner.provider,
                record=record,
                validation=winner.validation,
                repairs_used=winner.repairs_used,
                score=winner.score,
            )
        return winner

    task_prompt = _original_task_prompt(winner.record, original_prompt)
    verified_code = extract_code(winner.record.get("response") or "")
    if not verified_code.strip():
        return winner

    # --- 1) Re-solve with Fable, hinted by Sol's verified kernel ----------------
    resolve_record = generate_via_gateway(
        gateway=gateway,
        api_key=api_key,
        provider=COT_PROVIDER,
        prompt=resolve_cot_prompt(original_prompt=task_prompt, verified_code=verified_code),
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # Keep mining-task prompt for SFT / novelty (not the CoT-recovery wrapper).
    resolve_record = dict(resolve_record)
    resolve_record = rebind_leaf_prompt(
        resolve_record,
        task_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        preserve_prior_request_sha256_as="cot_request_sha256",
    )
    if (winner.record.get("metadata") or {}).get("prompt_meta"):
        resolve_record.setdefault("metadata", {})
        resolve_record["metadata"]["prompt_meta"] = winner.record["metadata"]["prompt_meta"]

    resolve_validation = validator.validate_response(
        resolve_record["response"],
        run_benchmark=run_benchmark,
        strict=strict_validate,
        capture_ir=capture_ir,
        prompt_meta=(winner.record.get("metadata") or {}).get("prompt_meta"),
    )
    resolve_reasoning = normalize_training_reasoning(resolve_record.get("reasoning"))
    if not resolve_reasoning:
        resolve_reasoning = prose_rationale_from_response(resolve_record.get("response") or "")

    if resolve_validation.get("passed") and has_usable_training_reasoning(resolve_reasoning):
        stamped = _stamp_cot_meta(
            resolve_record,
            mode="fable_resolve",
            cot_record=resolve_record,
            sol_provider="openai",
        )
        stamped["reasoning"] = resolve_reasoning
        stamped["sparkproof_validation"] = resolve_validation
        stamped.setdefault("metadata", {})
        stamped["metadata"]["tier"] = assign_tier(resolve_validation, repairs_used=0)
        usage = (resolve_record.get("metadata") or {}).get("usage") or {}
        output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        score = acceptance_score(resolve_validation, output_tokens=output_tokens)
        return CandidateResult(
            provider=COT_PROVIDER,
            record=stamped,
            validation=resolve_validation,
            repairs_used=0,
            score=max(score, winner.score),
        )

    # --- 2) Explain fallback: keep Sol gold answer, attach Fable rationale ------
    explain_record = generate_via_gateway(
        gateway=gateway,
        api_key=api_key,
        provider=COT_PROVIDER,
        prompt=explain_cot_prompt(original_prompt=task_prompt, verified_code=verified_code),
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    explain_reasoning = normalize_training_reasoning(explain_record.get("reasoning"))
    if not explain_reasoning:
        explain_reasoning = prose_rationale_from_response(explain_record.get("response") or "")

    if not has_usable_training_reasoning(explain_reasoning):
        cleaned = rebind_leaf_prompt(
            dict(winner.record),
            task_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        cleaned["reasoning"] = None
        cleaned.setdefault("metadata", {})
        cleaned["metadata"]["cot_recovery"] = "failed"
        return CandidateResult(
            provider=winner.provider,
            record=cleaned,
            validation=winner.validation,
            repairs_used=winner.repairs_used,
            score=winner.score,
        )

    stamped = _stamp_cot_meta(
        winner.record,
        mode="fable_explain",
        cot_record=explain_record,
        sol_provider="openai",
    )
    stamped = rebind_leaf_prompt(
        stamped,
        task_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    stamped["reasoning"] = explain_reasoning
    return CandidateResult(
        provider=winner.provider,
        record=stamped,
        validation=winner.validation,
        repairs_used=winner.repairs_used,
        score=winner.score,
    )
