"""Canonical OpenRouter chat request bodies — use sparkproof.teacher_request."""

from __future__ import annotations

from sparkproof.gateways import GATEWAY_OPENROUTER
from sparkproof.policy import openrouter_model_for
from sparkproof.teacher_request import (
    build_chat_body as _build_chat_body,
    build_messages,
    generation_config,
    manifest_generation_config,
    request_sha256,
    verify_request_sha256,
)


def build_chat_body(
    *,
    provider: str,
    prompt: str,
    system: str | None,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str | None = None,
    **kwargs: object,
) -> dict:
    del kwargs
    from sparkproof.policy import REQUIRED_REASONING_EFFORT

    effort = reasoning_effort or REQUIRED_REASONING_EFFORT
    return _build_chat_body(
        gateway=GATEWAY_OPENROUTER,
        provider=provider,
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=effort,
    )


def openrouter_generation_config(*, max_tokens: int, temperature: float) -> dict:
    return generation_config(max_tokens=max_tokens, temperature=temperature)


__all__ = [
    "build_chat_body",
    "build_messages",
    "generation_config",
    "manifest_generation_config",
    "openrouter_generation_config",
    "openrouter_model_for",
    "request_sha256",
    "verify_request_sha256",
]
