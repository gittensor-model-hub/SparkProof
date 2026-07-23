import urllib.error
from io import BytesIO

import pytest

import sparkproof.generate.gateway_client as gateway_client
from sparkproof.generate.gateway_client import GatewayTransientError
from sparkproof.gateways import gateway_timeout_seconds


def test_generate_via_gateway_records_openrouter_response_model(monkeypatch):
    def fake_post(*, gateway, api_key, body, timeout=300, max_retries=None):
        assert body["model"] == "openai/gpt-5.6-sol"
        return (
            {
                "id": "gen-xyz",
                "model": "openai/gpt-5.6-sol-20260709",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            },
            {},
        )

    monkeypatch.setattr(gateway_client, "_post_chat", fake_post)

    record = gateway_client.generate_via_gateway(
        gateway="openrouter",
        api_key="k",
        provider="openai",
        prompt="hello",
        system=None,
        max_tokens=16,
        temperature=0.0,
    )

    assert record["gateway_model"] == "openai/gpt-5.6-sol-20260709"
    assert record["openrouter_model"] == "openai/gpt-5.6-sol-20260709"
    assert record["model"] == "gpt-5.6"
    assert record["metadata"]["openrouter_requested_model"] == "openai/gpt-5.6-sol"
    assert record["metadata"]["openrouter_response_model"] == "openai/gpt-5.6-sol-20260709"


def test_generate_via_gateway_records_openrouter_fable_response_model(monkeypatch):
    def fake_post(*, gateway, api_key, body, timeout=300, max_retries=None):
        assert body["model"] == "anthropic/claude-fable-5"
        return (
            {
                "id": "gen-xyz",
                "model": "anthropic/claude-fable-5-20260709",
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {},
            },
            {},
        )

    monkeypatch.setattr(gateway_client, "_post_chat", fake_post)

    record = gateway_client.generate_via_gateway(
        gateway="openrouter",
        api_key="k",
        provider="anthropic",
        prompt="hello",
        system=None,
        max_tokens=16,
        temperature=0.0,
    )

    assert record["gateway_model"] == "anthropic/claude-fable-5-20260709"
    assert record["openrouter_model"] == "anthropic/claude-fable-5-20260709"
    assert record["model"] == "claude-fable-5"
    assert record["metadata"]["openrouter_requested_model"] == "anthropic/claude-fable-5"
    assert record["metadata"]["openrouter_response_model"] == "anthropic/claude-fable-5-20260709"


def test_yunwu_default_timeout_is_longer_than_openrouter(monkeypatch):
    monkeypatch.delenv("SPARKPROOF_GATEWAY_TIMEOUT", raising=False)
    monkeypatch.delenv("SPARKPROOF_YUNWU_TIMEOUT", raising=False)
    monkeypatch.delenv("SPARKPROOF_OPENROUTER_TIMEOUT", raising=False)
    assert gateway_timeout_seconds("yunwu") > gateway_timeout_seconds("openrouter")


def test_post_chat_retries_read_timeout(monkeypatch):
    calls = {"n": 0}

    class FakeResp:
        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}],"model":"claude-fable-5"}'

        @property
        def headers(self):
            return {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=300):
        calls["n"] += 1
        if calls["n"] < 2:
            raise urllib.error.URLError(TimeoutError("The read operation timed out"))
        return FakeResp()

    monkeypatch.setattr(gateway_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(gateway_client.time, "sleep", lambda _: None)

    payload, _ = gateway_client._post_chat(
        gateway="yunwu",
        api_key="k",
        body={"model": "claude-fable-5"},
        timeout=60,
        max_retries=2,
    )
    assert payload["choices"][0]["message"]["content"] == "ok"
    assert calls["n"] == 2


def test_post_chat_raises_gateway_transient_after_retries(monkeypatch):
    def fake_urlopen(req, timeout=300):
        raise urllib.error.URLError(TimeoutError("The read operation timed out"))

    monkeypatch.setattr(gateway_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(gateway_client.time, "sleep", lambda _: None)

    with pytest.raises(GatewayTransientError, match="after 2 attempt"):
        gateway_client._post_chat(
            gateway="yunwu",
            api_key="k",
            body={"model": "claude-fable-5"},
            timeout=30,
            max_retries=1,
        )


def test_post_chat_retries_http_503(monkeypatch):
    calls = {"n": 0}

    class FakeResp:
        def read(self):
            return b'{"choices":[{"message":{"content":"ok"}}],"model":"gpt-5.6-sol"}'

        @property
        def headers(self):
            return {}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(req, timeout=300):
        nonlocal calls
        calls["n"] += 1
        if calls["n"] < 2:
            raise urllib.error.HTTPError(
                req.full_url,
                503,
                "unavailable",
                hdrs=None,
                fp=BytesIO(b"busy"),
            )
        return FakeResp()

    monkeypatch.setattr(gateway_client.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(gateway_client.time, "sleep", lambda _: None)

    payload, _ = gateway_client._post_chat(
        gateway="yunwu",
        api_key="k",
        body={"model": "gpt-5.6-sol"},
        timeout=60,
        max_retries=2,
    )
    assert calls["n"] == 2
    assert payload["model"] == "gpt-5.6-sol"
