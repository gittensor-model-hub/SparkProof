import os

import pytest

from sparkproof.gateways import GATEWAY_YUNWU, get_gateway
from sparkproof.policy import allowed_teachers_manifest, validate_gateway_trajectory
from sparkproof.teacher_request import build_chat_body, verify_request_sha256
from tests.conftest_helpers import TEST_GEN_CONFIG, make_trajectory


def test_yunwu_chat_body_uses_top_level_reasoning_effort():
    body = build_chat_body(
        gateway=GATEWAY_YUNWU,
        provider="openai",
        prompt="hello",
        system=None,
        max_tokens=2048,
        temperature=0.7,
    )
    assert body["model"] == os.environ.get("YUNWU_MODEL_OPENAI", "gpt-5.6-sol")
    assert body["reasoning_effort"] == "xhigh"
    assert "reasoning" not in body


def test_yunwu_request_sha256_roundtrip():
    record = make_trajectory("anthropic", "claude-fable-5", gateway=GATEWAY_YUNWU)
    verify_request_sha256(record, TEST_GEN_CONFIG)


def test_yunwu_allowed_teachers():
    policy = get_gateway(GATEWAY_YUNWU)
    teachers = allowed_teachers_manifest(GATEWAY_YUNWU)
    assert {t["gateway_model"] for t in teachers} == set(policy.models_by_provider.values())
    record = make_trajectory("openai", "gpt-5.6", gateway=GATEWAY_YUNWU)
    validate_gateway_trajectory(record)
    assert record["api_base"] == policy.api_base
    assert record["request_url"] == policy.chat_url


def test_yunwu_rejects_non_pinned_gateway_slug():
    record = make_trajectory("openai", "gpt-5.6", gateway=GATEWAY_YUNWU)
    record["gateway_model"] = "gpt-5-mini"
    record["openrouter_model"] = "gpt-5-mini"
    with pytest.raises(ValueError, match="gpt-5-mini"):
        validate_gateway_trajectory(record)
