from free_claude_code.application.model_registry import (
    CapabilityTier,
    FreeEligibility,
    ModelProfile,
    ModelRegistry,
)


def test_model_profile_builds_canonical_route_ref():
    profile = ModelProfile(
        provider_id="open_router",
        model_id="nvidia/nemotron-3-ultra-550b-a55b:free",
        capability_tier=CapabilityTier.TIER_1,
        capability_score=100.0,
        free_eligibility=FreeEligibility.VERIFIED_FREE,
    )

    assert profile.route_ref == "open_router/nvidia/nemotron-3-ultra-550b-a55b:free"
    assert profile.executable_for_zero_cost is True


def test_unknown_free_status_is_not_executable():
    profile = ModelProfile(
        provider_id="ollama_cloud",
        model_id="some-model",
        capability_tier=CapabilityTier.TIER_1,
        capability_score=100.0,
    )

    assert profile.free_eligibility is FreeEligibility.UNKNOWN
    assert profile.executable_for_zero_cost is False


def test_registry_replaces_existing_profile():
    registry = ModelRegistry()

    first = ModelProfile(
        provider_id="groq",
        model_id="model",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=50.0,
    )
    second = ModelProfile(
        provider_id="groq",
        model_id="model",
        capability_tier=CapabilityTier.TIER_2,
        capability_score=70.0,
    )

    registry.register(first)
    registry.register(second)

    assert registry.get("groq/model") == second
