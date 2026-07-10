import pytest

from sparkproof.manifest import build_manifest, manifest_id
from sparkproof.policy import GATEWAY, REQUIRED_REASONING_EFFORT, allowed_teachers_manifest
from tests.conftest_helpers import TEST_GEN_CONFIG, make_trajectory


def test_build_manifest_pins_allowed_teachers():
    trajectories = [
        make_trajectory("anthropic", "claude-fable-5"),
        make_trajectory("openai", "gpt-5.6", prompt="other"),
    ]
    manifest = build_manifest(
        trajectories,
        prompts_sha256="c" * 64,
        openrouter_generation_config=TEST_GEN_CONFIG,
    ).to_dict()
    assert manifest["sample_count"] == 2
    assert manifest["gateway"] == GATEWAY
    assert manifest["allowed_teachers"] == allowed_teachers_manifest()
    assert manifest["openrouter_generation_config"]["reasoning_effort"] == REQUIRED_REASONING_EFFORT
    assert manifest_id(manifest) == manifest_id(manifest)


def test_rejects_wrong_model():
    with pytest.raises(ValueError):
        build_manifest(
            [make_trajectory("openai", "gpt-5")],
            prompts_sha256="d" * 64,
            openrouter_generation_config=TEST_GEN_CONFIG,
        )


def test_rejects_missing_openrouter_fields():
    bad = make_trajectory("openai", "gpt-5.6")
    bad.pop("gateway")
    with pytest.raises(ValueError, match="gateway"):
        build_manifest(
            [bad],
            prompts_sha256="d" * 64,
            openrouter_generation_config=TEST_GEN_CONFIG,
        )
