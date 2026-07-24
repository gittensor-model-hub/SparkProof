"""Shared test helpers for teacher gateway attestation."""

from __future__ import annotations

from sparkproof.gateways import GATEWAY_OPENROUTER, gateway_model_for, get_gateway
from sparkproof.policy import REQUIRED_REASONING_EFFORT
from sparkproof.teacher_request import build_chat_body, generation_config, request_sha256

TEST_GEN_CONFIG = generation_config(max_tokens=2048, temperature=0.7)


def gateway_trajectory_fields(
    provider: str,
    *,
    gateway: str = GATEWAY_OPENROUTER,
    system: str | None = None,
) -> dict:
    policy = get_gateway(gateway)
    routed_model = gateway_model_for(gateway, provider)
    fields = {
        "gateway": gateway,
        "provider": provider,
        "system": system,
        "api_base": policy.api_base,
        "request_url": policy.chat_url,
        "gateway_model": routed_model,
    }
    if gateway == GATEWAY_OPENROUTER:
        fields["openrouter_model"] = routed_model
    return fields


def gateway_record_from_prompt(
    *,
    gateway: str,
    provider: str,
    prompt: str,
    system: str | None,
    max_tokens: int,
    temperature: float,
    model: str,
    response: str,
    reasoning: str | None = None,
    metadata: dict | None = None,
) -> dict:
    body = build_chat_body(
        gateway=gateway,
        provider=provider,
        prompt=prompt,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return {
        "prompt": prompt,
        "response": response,
        "model": model,
        "reasoning": reasoning,
        "request_sha256": request_sha256(body),
        "response_sha256": "c" * 64,
        "metadata": metadata or {},
        **gateway_trajectory_fields(provider, gateway=gateway, system=system),
    }


def make_trajectory(
    provider: str,
    model: str,
    *,
    gateway: str = GATEWAY_OPENROUTER,
    prompt: str = "2+2?",
    system: str | None = None,
) -> dict:
    routed_model = gateway_model_for(gateway, provider)
    body = build_chat_body(
        gateway=gateway,
        provider=provider,
        prompt=prompt,
        system=system,
        max_tokens=TEST_GEN_CONFIG["max_tokens"],
        temperature=TEST_GEN_CONFIG["temperature"],
    )
    metadata = {
        "gateway_generation_id": "gen-test123",
        "gateway_response_model": routed_model,
        "gateway_reasoning_effort": REQUIRED_REASONING_EFFORT,
        "gateway_max_tokens": TEST_GEN_CONFIG["max_tokens"],
        "gateway_temperature": TEST_GEN_CONFIG["temperature"],
    }
    if gateway == GATEWAY_OPENROUTER:
        metadata.update(
            {
                "openrouter_generation_id": "gen-test123",
                "openrouter_response_model": routed_model,
                "openrouter_reasoning_effort": REQUIRED_REASONING_EFFORT,
                "openrouter_max_tokens": TEST_GEN_CONFIG["max_tokens"],
                "openrouter_temperature": TEST_GEN_CONFIG["temperature"],
            }
        )
    from sparkproof.gateways import get_gateway

    policy = get_gateway(gateway)
    return {
        "prompt": prompt,
        "response": "4",
        "provider": provider,
        "model": model,
        "system": system,
        "gateway": gateway,
        "api_base": policy.api_base,
        "request_url": policy.chat_url,
        "gateway_model": routed_model,
        "openrouter_model": routed_model,
        "request_sha256": request_sha256(body),
        "response_sha256": "b" * 64,
        "metadata": metadata,
    }
