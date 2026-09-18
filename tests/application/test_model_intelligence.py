from free_claude_code.application.model_intelligence import (
    intelligence_for_route,
    profile_for_route,
)
from free_claude_code.application.model_registry import (
    CapabilityTier,
    FreeEligibility,
)


def test_ultra_is_tier_one():
    profile = profile_for_route("open_router/nvidia/nemotron-3-ultra-550b-a55b:free")

    assert profile.capability_tier is CapabilityTier.TIER_1
    assert profile.capability_score == 100.0
    assert profile.supports_reasoning is True
    assert profile.free_eligibility is FreeEligibility.UNKNOWN


def test_super_is_below_ultra():
    ultra = profile_for_route("open_router/nvidia/nemotron-3-ultra-550b-a55b:free")
    super_model = profile_for_route(
        "open_router/nvidia/nemotron-3-super-120b-a12b:free"
    )

    assert ultra.capability_tier < super_model.capability_tier
    assert ultra.capability_score > super_model.capability_score


def test_provider_does_not_change_model_capability():
    groq = profile_for_route("groq/openai/gpt-oss-120b")
    cloudflare = profile_for_route("cloudflare/openai/gpt-oss-120b")

    assert groq.capability_tier is cloudflare.capability_tier
    assert groq.capability_score == cloudflare.capability_score


def test_unknown_model_is_conservative():
    profile = profile_for_route("some_provider/completely-unknown-model")

    assert profile.capability_tier is CapabilityTier.TIER_4
    assert profile.capability_score == 0.0
    assert profile.free_eligibility is FreeEligibility.UNKNOWN


def test_intelligence_lookup_returns_known_model():
    intelligence = intelligence_for_route(
        "open_router/nvidia/nemotron-3-super-120b-a12b:free"
    )

    assert intelligence is not None
    assert intelligence.capability_tier is CapabilityTier.TIER_2
