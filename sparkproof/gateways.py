"""Pinned teacher API gateways (OpenRouter + yunwu.ai relay)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GatewayPolicy:
    name: str
    api_base: str
    chat_url: str
    api_key_env: str
    models_by_provider: dict[str, str]
    reasoning_in_body: str  # "nested" (OpenRouter) | "top_level" (yunwu)


GATEWAY_OPENROUTER = "openrouter"
GATEWAY_YUNWU = "yunwu"

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_URL = f"{OPENROUTER_API_BASE}/chat/completions"

# OpenRouter slugs (provider/model).
OPENROUTER_MODEL_ANTHROPIC = "anthropic/claude-fable-5"
OPENROUTER_MODEL_OPENAI = "openai/gpt-5.6-sol"

# yunwu docs: https://yunwu.apifox.cn/ — native slugs from 支持模型 (/v1/models).
# Production pins the same logical teachers as OpenRouter (Fable 5 + GPT 5.6 Sol).
YUNWU_DEFAULT_ANTHROPIC = "claude-fable-5"
YUNWU_DEFAULT_OPENAI = "gpt-5.6-sol"
YUNWU_PINNED_SLUGS = frozenset({YUNWU_DEFAULT_ANTHROPIC, YUNWU_DEFAULT_OPENAI})
# yunwu may echo a provider alias without the -sol suffix on responses.
YUNWU_ACCEPTED_RESPONSE_SLUGS = YUNWU_PINNED_SLUGS | frozenset({"gpt-5.6"})


def yunwu_api_base() -> str:
    return os.environ.get("YUNWU_API_BASE", "https://yunwu.ai/v1").rstrip("/")


def yunwu_chat_url() -> str:
    return f"{yunwu_api_base()}/chat/completions"


def yunwu_models_url() -> str:
    return f"{yunwu_api_base()}/models"


def _pinned_yunwu_slug(env_key: str, default: str) -> str:
    value = os.environ.get(env_key, default).strip()
    if value not in YUNWU_PINNED_SLUGS:
        raise ValueError(
            f"{env_key} must be one of {sorted(YUNWU_PINNED_SLUGS)!r} "
            f"(production teachers only), got {value!r}"
        )
    return value


def yunwu_model_anthropic() -> str:
    return _pinned_yunwu_slug("YUNWU_MODEL_ANTHROPIC", YUNWU_DEFAULT_ANTHROPIC)


def yunwu_model_openai() -> str:
    return _pinned_yunwu_slug("YUNWU_MODEL_OPENAI", YUNWU_DEFAULT_OPENAI)


# Back-compat module constants (call lazy helpers when building policy).
YUNWU_API_BASE = yunwu_api_base()
YUNWU_CHAT_URL = yunwu_chat_url()
YUNWU_MODELS_URL = yunwu_models_url()
YUNWU_MODEL_ANTHROPIC = yunwu_model_anthropic()
YUNWU_MODEL_OPENAI = yunwu_model_openai()


def _openrouter_policy() -> GatewayPolicy:
    return GatewayPolicy(
        name=GATEWAY_OPENROUTER,
        api_base=OPENROUTER_API_BASE,
        chat_url=OPENROUTER_CHAT_URL,
        api_key_env="OPENROUTER_API_KEY",
        models_by_provider={
            "anthropic": OPENROUTER_MODEL_ANTHROPIC,
            "openai": OPENROUTER_MODEL_OPENAI,
        },
        reasoning_in_body="nested",
    )


def _yunwu_policy() -> GatewayPolicy:
    base = yunwu_api_base()
    return GatewayPolicy(
        name=GATEWAY_YUNWU,
        api_base=base,
        chat_url=f"{base}/chat/completions",
        api_key_env="YUNWU_API_KEY",
        models_by_provider={
            "anthropic": yunwu_model_anthropic(),
            "openai": yunwu_model_openai(),
        },
        reasoning_in_body="top_level",
    )


def _gateways() -> dict[str, GatewayPolicy]:
    return {
        GATEWAY_OPENROUTER: _openrouter_policy(),
        GATEWAY_YUNWU: _yunwu_policy(),
    }


ALLOWED_GATEWAYS = frozenset({GATEWAY_OPENROUTER, GATEWAY_YUNWU})


def default_gateway() -> str:
    name = os.environ.get("SPARKPROOF_GATEWAY", GATEWAY_OPENROUTER).strip().lower()
    if name not in ALLOWED_GATEWAYS:
        raise ValueError(f"unknown SPARKPROOF_GATEWAY={name!r}; expected one of {sorted(ALLOWED_GATEWAYS)}")
    return name


def get_gateway(name: str) -> GatewayPolicy:
    try:
        return _gateways()[name]
    except KeyError as e:
        raise ValueError(f"unknown gateway {name!r}, expected one of {sorted(ALLOWED_GATEWAYS)}") from e


def gateway_model_for(gateway: str, provider: str) -> str:
    policy = get_gateway(gateway)
    try:
        return policy.models_by_provider[provider]
    except KeyError as e:
        raise ValueError(
            f"unsupported provider {provider!r} for gateway {gateway!r}, "
            f"expected one of {sorted(policy.models_by_provider)}"
        ) from e


def resolve_api_key(gateway: str) -> str:
    policy = get_gateway(gateway)
    value = os.environ.get(policy.api_key_env, "").strip()
    if not value:
        raise KeyError(policy.api_key_env)
    return value


def gateway_timeout_seconds(gateway: str) -> int:
    """Per-request read timeout for teacher chat completions.

  Yunwu at ``xhigh`` reasoning often exceeds 5 minutes; default 15 minutes there.
  Override with ``SPARKPROOF_GATEWAY_TIMEOUT`` (all gateways) or gateway-specific vars.
    """
    if override := os.environ.get("SPARKPROOF_GATEWAY_TIMEOUT", "").strip():
        return max(1, int(override))
    if gateway == GATEWAY_YUNWU:
        return max(1, int(os.environ.get("SPARKPROOF_YUNWU_TIMEOUT", "900")))
    return max(1, int(os.environ.get("SPARKPROOF_OPENROUTER_TIMEOUT", "300")))


def gateway_max_retries() -> int:
    return max(0, int(os.environ.get("SPARKPROOF_GATEWAY_RETRIES", "3")))


def gateway_retry_backoff_seconds() -> float:
    return max(0.0, float(os.environ.get("SPARKPROOF_GATEWAY_RETRY_BACKOFF", "5")))


def trajectory_gateway_model(record: dict[str, Any]) -> str | None:
    return record.get("gateway_model") or record.get("openrouter_model")


def openrouter_response_matches_pinned(actual: str, pinned: str) -> bool:
    """Return whether an OpenRouter response/ledger model matches a pinned slug.

    OpenRouter may echo a dated build id (e.g. openai/gpt-5.6-sol-20260709) for a
    request routed to openai/gpt-5.6-sol. Treat exact matches and dated suffixes as
    equivalent; reject unrelated models.
    """
    if actual == pinned:
        return True
    return actual.startswith(f"{pinned}-")


def allowed_teachers_for_gateway(gateway: str) -> list[dict[str, str]]:
    from sparkproof.policy import (
        ANTHROPIC_TEACHER_MODEL,
        OPENAI_TEACHER_MODEL,
        REQUIRED_REASONING_EFFORT,
    )

    policy = get_gateway(gateway)
    return [
        {
            "provider": "anthropic",
            "model": ANTHROPIC_TEACHER_MODEL,
            "gateway_model": policy.models_by_provider["anthropic"],
            "openrouter_model": policy.models_by_provider["anthropic"],
            "reasoning_effort": REQUIRED_REASONING_EFFORT,
        },
        {
            "provider": "openai",
            "model": OPENAI_TEACHER_MODEL,
            "gateway_model": policy.models_by_provider["openai"],
            "openrouter_model": policy.models_by_provider["openai"],
            "reasoning_effort": REQUIRED_REASONING_EFFORT,
        },
    ]
