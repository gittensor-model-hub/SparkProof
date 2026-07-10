import pytest

from sparkproof.openrouter_request import build_chat_body, verify_request_sha256
from sparkproof.policy import REQUIRED_REASONING_EFFORT, allowed_teachers_manifest
from tests.conftest_helpers import TEST_GEN_CONFIG, make_trajectory


def test_build_chat_body_pins_xhigh():
    body = build_chat_body(
        provider="openai",
        prompt="hello",
        system=None,
        max_tokens=2048,
        temperature=0.7,
    )
    assert body["model"] == "openai/gpt-5.6-sol"
    assert body["reasoning"] == {"effort": "xhigh"}


def test_verify_request_sha256_accepts_matching_trajectory():
    record = make_trajectory("anthropic", "claude-fable-5")
    verify_request_sha256(record, TEST_GEN_CONFIG)


def test_verify_request_sha256_rejects_tampered_hash():
    record = make_trajectory("openai", "gpt-5.6")
    record["request_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="request_sha256 mismatch"):
        verify_request_sha256(record, TEST_GEN_CONFIG)


def test_allowed_teachers_include_xhigh():
    teachers = allowed_teachers_manifest()
    assert all(t["reasoning_effort"] == REQUIRED_REASONING_EFFORT for t in teachers)
    assert {t["openrouter_model"] for t in teachers} == {
        "anthropic/claude-fable-5",
        "openai/gpt-5.6-sol",
    }
