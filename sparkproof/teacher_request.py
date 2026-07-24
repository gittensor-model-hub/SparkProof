"""Canonical teacher chat request bodies for SparkProof attestation."""

from __future__ import annotations

from typing import Any

from sparkproof.gateways import gateway_model_for, get_gateway
from sparkproof.hashing import canonical_json_bytes, sha256_hex
from sparkproof.policy import REQUIRED_REASONING_EFFORT, trajectory_gateway_model


def build_messages(*, prompt: str, system: str | None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def build_chat_body(
    *,
    gateway: str,
    provider: str,
    prompt: str,
    system: str | None,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str = REQUIRED_REASONING_EFFORT,
) -> dict[str, Any]:
    """Pinned request shape — must match what generators send for `gateway`."""
    if reasoning_effort != REQUIRED_REASONING_EFFORT:
        raise ValueError(
            f"SparkProof production requires reasoning effort {REQUIRED_REASONING_EFFORT!r}, got {reasoning_effort!r}"
        )

    policy = get_gateway(gateway)
    body: dict[str, Any] = {
        "model": gateway_model_for(gateway, provider),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": build_messages(prompt=prompt, system=system),
    }
    if policy.reasoning_in_body == "nested":
        body["reasoning"] = {"effort": reasoning_effort}
    else:
        body["reasoning_effort"] = reasoning_effort
    return body


def request_sha256(body: dict[str, Any]) -> str:
    return sha256_hex(canonical_json_bytes(body))


def generation_config(*, max_tokens: int, temperature: float) -> dict[str, Any]:
    return {
        "reasoning_effort": REQUIRED_REASONING_EFFORT,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def manifest_generation_config(manifest: dict[str, Any]) -> dict[str, Any]:
    gen = manifest.get("generation_config") or manifest.get("openrouter_generation_config")
    if not gen:
        raise ValueError("manifest missing generation_config")
    return gen


def verify_request_sha256(record: dict[str, Any], generation_config: dict[str, Any]) -> None:
    """Prove the committed request hash matches the pinned xhigh teacher call."""
    gateway = record["gateway"]
    expected_effort = generation_config.get("reasoning_effort", REQUIRED_REASONING_EFFORT)
    if expected_effort != REQUIRED_REASONING_EFFORT:
        raise ValueError(f"generation_config.reasoning_effort must be {REQUIRED_REASONING_EFFORT!r}")

    body = build_chat_body(
        gateway=gateway,
        provider=record["provider"],
        prompt=record["prompt"],
        system=record.get("system"),
        max_tokens=int(generation_config["max_tokens"]),
        temperature=float(generation_config["temperature"]),
        reasoning_effort=expected_effort,
    )
    committed = record.get("request_sha256")
    recomputed = request_sha256(body)
    if committed != recomputed:
        routed = trajectory_gateway_model(record)
        raise ValueError(
            f"request_sha256 mismatch — miner did not call {gateway!r} with pinned "
            f"{routed!r} + reasoning.effort={expected_effort!r}"
        )

    meta = record.get("metadata") or {}
    for key in ("gateway_reasoning_effort", "openrouter_reasoning_effort"):
        meta_effort = meta.get(key)
        if meta_effort and meta_effort != REQUIRED_REASONING_EFFORT:
            raise ValueError(f"metadata.{key} must be {REQUIRED_REASONING_EFFORT!r}")


def rebind_leaf_prompt(
    record: dict[str, Any],
    task_prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    preserve_prior_request_sha256_as: str | None = None,
) -> dict[str, Any]:
    """Reset the leaf ``prompt`` to the mining task and align ``request_sha256``.

    Repair, optimize, and CoT-recovery calls use wrapper prompts but exported rows
    must fingerprint the original mining task for SFT/novelty. Recompute the leaf
    hash from the mining-task body so ``verify_request_sha256`` matches release gate.
    """
    rebound = dict(record)
    prior_hash = rebound.get("request_sha256")
    if preserve_prior_request_sha256_as and prior_hash:
        meta = dict(rebound.get("metadata") or {})
        meta[preserve_prior_request_sha256_as] = prior_hash
        rebound["metadata"] = meta
    rebound["prompt"] = task_prompt
    body = build_chat_body(
        gateway=rebound["gateway"],
        provider=rebound["provider"],
        prompt=task_prompt,
        system=rebound.get("system"),
        max_tokens=max_tokens,
        temperature=temperature,
    )
    rebound["request_sha256"] = request_sha256(body)
    return rebound
