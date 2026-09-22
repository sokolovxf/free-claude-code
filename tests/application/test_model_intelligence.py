from free_claude_code.application.model_intelligence import (
    intelligence_for_route,
    profile_for_configured_route,
    profile_for_route,
    synchronize_model_registry,
)
from free_claude_code.application.model_registry import (
    CapabilityTier,
    FreeEligibility,
    ModelRegistry,
)
from free_claude_code.config.settings import Settings


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


def test_parameter_size_infers_tier_for_new_large_models():
    ultra = profile_for_route("nvidia_nim/qwen/Qwen3.8-2.4T-A95B")
    large = profile_for_route("huggingface/Qwen/Qwen3-Coder-480B-A35B-Instruct")
    medium = profile_for_route("nvidia_nim/nvidia/nemotron-4-340b-instruct")
    small = profile_for_route("siliconflow/Qwen/Qwen3.5-27B")

    assert ultra.capability_tier is CapabilityTier.TIER_1
    assert large.capability_tier is CapabilityTier.TIER_1
    assert medium.capability_tier is CapabilityTier.TIER_2
    assert small.capability_tier is CapabilityTier.TIER_3
    assert ultra.capability_score > medium.capability_score > small.capability_score
    assert large.supports_tools is True


def test_intelligence_lookup_returns_known_model():
    intelligence = intelligence_for_route(
        "open_router/nvidia/nemotron-3-super-120b-a12b:free"
    )

    assert intelligence is not None
    assert intelligence.capability_tier is CapabilityTier.TIER_2


def test_profile_for_route_retains_unknown_free_eligibility_by_default():
    profile = profile_for_route("open_router/nvidia/nemotron-3-ultra-550b-a55b:free")

    assert profile.free_eligibility is FreeEligibility.UNKNOWN
    assert profile.verification_source is None
    assert profile.last_verified is None
    assert profile.executable_for_zero_cost is False


def test_configured_route_listed_as_verified_free_becomes_verified_free():
    profile = profile_for_configured_route(
        "open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
        ("open_router/nvidia/nemotron-3-ultra-550b-a55b:free",),
    )

    assert profile.free_eligibility is FreeEligibility.VERIFIED_FREE
    assert profile.verification_source == "FCC_VERIFIED_FREE_MODELS"
    assert profile.last_verified is not None
    assert profile.executable_for_zero_cost is True
    assert profile.capability_tier is CapabilityTier.TIER_1


def test_known_model_not_listed_remains_unknown():
    profile = profile_for_configured_route(
        "open_router/nvidia/nemotron-3-super-120b-a12b:free",
        ("open_router/nvidia/nemotron-3-ultra-550b-a55b:free",),
    )

    assert profile.capability_tier is CapabilityTier.TIER_2
    assert profile.free_eligibility is FreeEligibility.UNKNOWN
    assert profile.verification_source is None


def test_synchronize_registry_populates_configured_model_and_fallbacks():
    settings = Settings(
        MODEL="open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
        MODEL_OPUS="groq/openai/gpt-oss-120b",
        MODEL_FALLBACKS="open_router/nvidia/nemotron-3-super-120b-a12b:free,groq/openai/gpt-oss-120b",
        FCC_VERIFIED_FREE_MODELS=(
            "open_router/nvidia/nemotron-3-ultra-550b-a55b:free,"
            "open_router/nvidia/nemotron-3-super-120b-a12b:free"
        ),
    )

    registry = ModelRegistry()
    synchronize_model_registry(registry, settings)

    primary = registry.get("open_router/nvidia/nemotron-3-ultra-550b-a55b:free")
    fallback = registry.get("open_router/nvidia/nemotron-3-super-120b-a12b:free")
    override = registry.get("groq/openai/gpt-oss-120b")

    assert primary is not None
    assert primary.free_eligibility is FreeEligibility.VERIFIED_FREE
    assert primary.capability_tier is CapabilityTier.TIER_1

    assert fallback is not None
    assert fallback.free_eligibility is FreeEligibility.VERIFIED_FREE
    assert fallback.capability_tier is CapabilityTier.TIER_2

    assert override is not None
    assert override.capability_tier is CapabilityTier.TIER_2
    assert override.free_eligibility is FreeEligibility.UNKNOWN


def test_ollama_ultra_route_is_tier_one():
    # The two Nemotron 3 Ultra-class routes must share a TIER_1 classification
    # even though one omits the "550b" size qualifier present in the other.
    openrouter = profile_for_route(
        "open_router/nvidia/nemotron-3-ultra-550b-a55b:free"
    )
    ollama = profile_for_route("ollama_cloud/nemotron-3-ultra")

    assert openrouter.capability_tier is CapabilityTier.TIER_1
    assert ollama.capability_tier is CapabilityTier.TIER_1
    assert ollama.capability_score == openrouter.capability_score


def test_capability_tier_is_independent_of_configured_position():
    # Ollama Ultra appears after Super in some configured ordering, but that
    # position must not downgrade it below the Super class.
    super_route = profile_for_route(
        "open_router/nvidia/nemotron-3-super-120b-a12b:free"
    )
    ollama_ultra = profile_for_route("ollama_cloud/nemotron-3-ultra")
    ollama_super = profile_for_route("ollama_cloud/nemotron-3-super")

    assert ollama_ultra.capability_tier < super_route.capability_tier
    assert ollama_ultra.capability_tier < ollama_super.capability_tier
    assert ollama_super.capability_tier is CapabilityTier.TIER_2


def test_same_family_provider_routes_remain_separate_registry_entries():
    settings = Settings(
        MODEL="open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
        MODEL_FALLBACKS="ollama_cloud/nemotron-3-ultra",
        FCC_VERIFIED_FREE_MODELS=(
            "open_router/nvidia/nemotron-3-ultra-550b-a55b:free"
        ),
    )

    registry = ModelRegistry()
    synchronize_model_registry(registry, settings)

    openrouter = registry.get(
        "open_router/nvidia/nemotron-3-ultra-550b-a55b:free"
    )
    ollama = registry.get("ollama_cloud/nemotron-3-ultra")

    # Both routes exist independently, never collapsed into one executable route.
    assert openrouter is not None
    assert ollama is not None
    assert openrouter.route_ref != ollama.route_ref
    assert openrouter.free_eligibility is FreeEligibility.VERIFIED_FREE
    assert ollama.free_eligibility is FreeEligibility.UNKNOWN
    assert openrouter.capability_tier is ollama.capability_tier is CapabilityTier.TIER_1


def test_synchronize_registry_mirrors_active_settings_and_prunes_removed_route():
    settings = Settings(
        MODEL="open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
        MODEL_FALLBACKS="ollama_cloud/nemotron-3-ultra,groq/openai/gpt-oss-120b",
    )
    registry = ModelRegistry()
    synchronize_model_registry(registry, settings)
    assert registry.get("groq/openai/gpt-oss-120b") is not None

    # Removing a route from the active configuration removes it as a candidate.
    settings.model_fallbacks = ("ollama_cloud/nemotron-3-ultra",)
    synchronize_model_registry(registry, settings)

    refs = {profile.route_ref for profile in registry.all_profiles()}
    assert "groq/openai/gpt-oss-120b" not in refs
    assert "ollama_cloud/nemotron-3-ultra" in refs
    assert "open_router/nvidia/nemotron-3-ultra-550b-a55b:free" in refs


def test_synchronize_registry_discovers_new_route_without_editing_smart_router():
    registry = ModelRegistry()
    synchronize_model_registry(
        registry,
        Settings(MODEL="open_router/nvidia/nemotron-3-ultra-550b-a55b:free"),
    )
    assert registry.get("ollama_cloud/nemotron-3-super") is None

    # Adding a new configured fallback makes it visible without any Smart Router
    # or hard-coded pool change.
    synchronize_model_registry(
        registry,
        Settings(
            MODEL="open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
            MODEL_FALLBACKS="ollama_cloud/nemotron-3-super",
        ),
    )
    added = registry.get("ollama_cloud/nemotron-3-super")
    assert added is not None
    assert added.capability_tier is CapabilityTier.TIER_2


def test_synchronize_registry_does_not_invent_a_second_pool():
    registry = ModelRegistry()
    synchronize_model_registry(
        registry,
        Settings(MODEL="open_router/nvidia/nemotron-3-ultra-550b-a55b:free"),
    )

    refs = {profile.route_ref for profile in registry.all_profiles()}
    assert refs == {"open_router/nvidia/nemotron-3-ultra-550b-a55b:free"}


def test_unknown_pool_route_gets_safe_conservative_profile():
    # A route with an explicit size marker gets a coarse inferred tier, while
    # a route with no static intelligence or size evidence stays conservative.
    sized = profile_for_configured_route("ollama_cloud/gemma4:31b", ())
    unknown = profile_for_configured_route("open_router/openrouter/free", ())

    assert sized.capability_tier is CapabilityTier.TIER_3
    assert sized.capability_score > 0.0
    assert sized.free_eligibility is FreeEligibility.UNKNOWN
    assert unknown.capability_tier is CapabilityTier.TIER_4
    assert unknown.capability_score == 0.0
    assert unknown.free_eligibility is FreeEligibility.UNKNOWN
