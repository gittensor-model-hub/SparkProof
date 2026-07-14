import pytest

from sparkproof.policy import (
    GATEWAY,
    OPENROUTER_API_BASE,
    OPENROUTER_CHAT_URL,
    openrouter_model_for,
    validate_openrouter_trajectory,
    validate_provider_model,
)


def test_allows_fable_5_and_gpt_5_6():
    validate_provider_model("anthropic", "claude-fable-5")
    validate_provider_model("openai", "gpt-5.6")
    validate_provider_model("openai", "gpt-5.6-sol")


def test_rejects_other_models():
    with pytest.raises(ValueError, match="unsupported model"):
        validate_provider_model("openai", "gpt-5")
    with pytest.raises(ValueError, match="unsupported provider"):
        validate_provider_model("deepseek", "deepseek-reasoner")


def test_openrouter_slugs():
    assert openrouter_model_for("anthropic") == "anthropic/claude-fable-5"
    assert openrouter_model_for("openai") == "openai/gpt-5.6-sol"


def test_validate_openrouter_trajectory_accepts_committed_fields():
    record = {
        "provider": "anthropic",
        "model": "claude-fable-5",
        "gateway": GATEWAY,
        "api_base": OPENROUTER_API_BASE,
        "request_url": OPENROUTER_CHAT_URL,
        "gateway_model": "anthropic/claude-fable-5",
        "openrouter_model": "anthropic/claude-fable-5",
        "metadata": {
            "openrouter_generation_id": "gen-abc",
            "openrouter_requested_model": "anthropic/claude-fable-5",
            "openrouter_response_model": "anthropic/claude-fable-5",
            "openrouter_reasoning_effort": "xhigh",
        },
    }
    validate_openrouter_trajectory(record)


def test_validate_openrouter_trajectory_accepts_dated_response_model():
    record = {
        "provider": "openai",
        "model": "gpt-5.6",
        "gateway": GATEWAY,
        "api_base": OPENROUTER_API_BASE,
        "request_url": OPENROUTER_CHAT_URL,
        "gateway_model": "openai/gpt-5.6-sol-20260709",
        "openrouter_model": "openai/gpt-5.6-sol-20260709",
        "metadata": {
            "openrouter_generation_id": "gen-abc",
            "openrouter_requested_model": "openai/gpt-5.6-sol",
            "openrouter_response_model": "openai/gpt-5.6-sol-20260709",
            "openrouter_reasoning_effort": "xhigh",
        },
    }
    validate_openrouter_trajectory(record)


def test_validate_openrouter_rejects_wrong_url():
    record = {
        "provider": "openai",
        "model": "gpt-5.6",
        "gateway": GATEWAY,
        "api_base": OPENROUTER_API_BASE,
        "request_url": "https://api.openai.com/v1/chat/completions",
        "openrouter_model": "openai/gpt-5.6-sol",
        "metadata": {},
    }
    with pytest.raises(ValueError, match="request_url"):
        validate_openrouter_trajectory(record)
