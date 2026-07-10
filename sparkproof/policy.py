"""Pinned teacher models + gateway policy."""

from __future__ import annotations

from typing import Any

from sparkproof.gateways import (
    ALLOWED_GATEWAYS,
    GATEWAY_OPENROUTER,
    GATEWAY_YUNWU,
    OPENROUTER_API_BASE,
    OPENROUTER_CHAT_URL,
    OPENROUTER_MODEL_ANTHROPIC,
    OPENROUTER_MODEL_OPENAI,
    YUNWU_API_BASE,
    YUNWU_CHAT_URL,
    YUNWU_MODEL_ANTHROPIC,
    YUNWU_MODEL_OPENAI,
    allowed_teachers_for_gateway,
    gateway_model_for,
    get_gateway,
    trajectory_gateway_model,
)

GENERATOR_VERSION = "0.3.0"
GATEWAY = GATEWAY_OPENROUTER  # default / legacy alias
DATASET_KIND_TRITON_BLACKWELL = "triton-3.7.1-blackwell"
TRITON_VERSION = "3.7.1"

# Logical teacher models (match SparkDistill `teacher/providers.py`)
ANTHROPIC_TEACHER_MODEL = "claude-fable-5"
OPENAI_TEACHER_MODEL = "gpt-5.6"
REQUIRED_REASONING_EFFORT = "xhigh"

OPENROUTER_MODEL_BY_PROVIDER: dict[str, str] = {
    "anthropic": OPENROUTER_MODEL_ANTHROPIC,
    "openai": OPENROUTER_MODEL_OPENAI,
}

YUNWU_MODEL_BY_PROVIDER: dict[str, str] = {
    "anthropic": YUNWU_MODEL_ANTHROPIC,
    "openai": YUNWU_MODEL_OPENAI,
}

ALLOWED_MODELS: dict[str, frozenset[str]] = {
    "anthropic": frozenset({ANTHROPIC_TEACHER_MODEL}),
    "openai": frozenset({OPENAI_TEACHER_MODEL, "gpt-5.6-sol"}),
}

SUPPORTED_PROVIDERS = frozenset({"anthropic", "openai"})
ALLOWED_OPENROUTER_MODELS = frozenset(OPENROUTER_MODEL_BY_PROVIDER.values())


def openrouter_model_for(provider: str) -> str:
    return gateway_model_for(GATEWAY_OPENROUTER, provider)


def normalize_upstream_model(provider: str, upstream_model: str, *, gateway: str | None = None) -> str:
    """Map gateway/upstream model string to the logical teacher model id."""
    if gateway == GATEWAY_YUNWU:
        # yunwu echoes native slugs (claude-sonnet-5, gpt-5-mini); logical teacher is by provider.
        if provider == "anthropic":
            return ANTHROPIC_TEACHER_MODEL
        if provider == "openai":
            return OPENAI_TEACHER_MODEL

    validate_provider_model(provider, upstream_model)
    if provider == "anthropic":
        return ANTHROPIC_TEACHER_MODEL
    if upstream_model in {"gpt-5.6", "gpt-5.6-sol"}:
        return OPENAI_TEACHER_MODEL
    return upstream_model


def validate_provider_model(provider: str, model: str) -> None:
    if provider not in ALLOWED_MODELS:
        raise ValueError(f"unsupported provider {provider!r}, expected {sorted(ALLOWED_MODELS)}")
    if model not in ALLOWED_MODELS[provider]:
        raise ValueError(
            f"unsupported model {model!r} for provider {provider!r}, "
            f"expected {sorted(ALLOWED_MODELS[provider])}"
        )


def validate_gateway_trajectory(record: dict[str, Any]) -> None:
    """Verify this sample was captured via an approved gateway, not a direct provider API."""
    gateway = record.get("gateway")
    if gateway not in ALLOWED_GATEWAYS:
        raise ValueError(f"gateway must be one of {sorted(ALLOWED_GATEWAYS)!r}, got {gateway!r}")

    policy = get_gateway(gateway)
    if record.get("api_base") != policy.api_base:
        raise ValueError(f"api_base must be {policy.api_base!r}, got {record.get('api_base')!r}")
    if record.get("request_url") != policy.chat_url:
        raise ValueError(f"request_url must be {policy.chat_url!r}, got {record.get('request_url')!r}")

    routed_model = trajectory_gateway_model(record)
    allowed_models = frozenset(policy.models_by_provider.values())
    if routed_model not in allowed_models:
        raise ValueError(
            f"gateway_model {routed_model!r} not allowed for {gateway!r}; "
            f"expected one of {sorted(allowed_models)}"
        )

    expected_slug = policy.models_by_provider.get(record["provider"])
    if routed_model != expected_slug:
        raise ValueError(
            f"gateway_model {routed_model!r} does not match provider "
            f"{record['provider']!r} (expected {expected_slug!r})"
        )

    meta = record.get("metadata") or {}
    response_model = meta.get("gateway_response_model") or meta.get("openrouter_response_model")
    if response_model and response_model != routed_model and gateway != GATEWAY_YUNWU:
        raise ValueError(
            f"response model {response_model!r} does not match requested gateway_model {routed_model!r}"
        )

    if gateway == GATEWAY_OPENROUTER:
        generation_id = meta.get("openrouter_generation_id")
        if generation_id and not str(generation_id).startswith("gen-"):
            raise ValueError(f"openrouter_generation_id does not look like an OpenRouter id: {generation_id!r}")

    reasoning_effort = meta.get("gateway_reasoning_effort") or meta.get("openrouter_reasoning_effort")
    if reasoning_effort is not None and reasoning_effort != REQUIRED_REASONING_EFFORT:
        raise ValueError(
            f"gateway_reasoning_effort must be {REQUIRED_REASONING_EFFORT!r}, got {reasoning_effort!r}"
        )


def validate_openrouter_trajectory(record: dict) -> None:
    """Backward-compatible alias."""
    validate_gateway_trajectory(record)


def allowed_teachers_manifest(gateway: str = GATEWAY_OPENROUTER) -> list[dict[str, str]]:
    return allowed_teachers_for_gateway(gateway)


__all__ = [
    "ALLOWED_GATEWAYS",
    "ALLOWED_MODELS",
    "ALLOWED_OPENROUTER_MODELS",
    "ANTHROPIC_TEACHER_MODEL",
    "DATASET_KIND_TRITON_BLACKWELL",
    "GATEWAY",
    "GATEWAY_OPENROUTER",
    "GENERATOR_VERSION",
    "OPENAI_TEACHER_MODEL",
    "OPENROUTER_API_BASE",
    "OPENROUTER_CHAT_URL",
    "OPENROUTER_MODEL_ANTHROPIC",
    "OPENROUTER_MODEL_OPENAI",
    "REQUIRED_REASONING_EFFORT",
    "SUPPORTED_PROVIDERS",
    "TRITON_VERSION",
    "YUNWU_API_BASE",
    "YUNWU_CHAT_URL",
    "YUNWU_MODEL_ANTHROPIC",
    "YUNWU_MODEL_OPENAI",
    "allowed_teachers_manifest",
    "normalize_upstream_model",
    "openrouter_model_for",
    "trajectory_gateway_model",
    "validate_gateway_trajectory",
    "validate_openrouter_trajectory",
    "validate_provider_model",
]
