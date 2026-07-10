"""OpenRouter teacher client — use sparkproof.generate.gateway_client."""

from __future__ import annotations

from sparkproof.generate.gateway_client import generate_via_gateway, generation_config_for_run
from sparkproof.gateways import GATEWAY_OPENROUTER, resolve_api_key
from sparkproof.policy import REQUIRED_REASONING_EFFORT


def generate_via_openrouter(
    *,
    api_key: str,
    provider: str,
    prompt: str,
    system: str | None,
    max_tokens: int,
    temperature: float = 0.7,
    reasoning_effort: str = REQUIRED_REASONING_EFFORT,
) -> dict:
    return generate_via_gateway(
        gateway=GATEWAY_OPENROUTER,
        api_key=api_key or resolve_api_key(GATEWAY_OPENROUTER),
        provider=provider,
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )


__all__ = ["generate_via_openrouter", "generation_config_for_run"]
