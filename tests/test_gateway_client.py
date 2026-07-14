import sparkproof.generate.gateway_client as gateway_client


def test_generate_via_gateway_records_openrouter_response_model(monkeypatch):
    def fake_post(*, gateway, api_key, body, timeout=300):
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
