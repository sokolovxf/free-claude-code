from datetime import UTC, datetime, timedelta

from free_claude_code.application.model_registry import (
    CapabilityTier,
    FreeEligibility,
    ModelProfile,
    ModelRegistry,
)
from free_claude_code.application.route_health import RouteHealthStore
from free_claude_code.application.routing import ProviderModelTarget
from free_claude_code.application.smart_router import (
    RouteRequirements,
    SmartRouter,
)


def target(ref: str) -> ProviderModelTarget:
    provider, model = ref.split("/", 1)
    return ProviderModelTarget(
        provider_id=provider,
        provider_model=model,
        provider_model_ref=ref,
    )


def test_router_prefers_higher_capability_tier():
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="groq",
                model_id="120b",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="open_router",
                model_id="550b",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=100.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )

    health = RouteHealthStore()
    router = SmartRouter(registry, health)

    selected = router.select(
        (
            target("groq/120b"),
            target("open_router/550b"),
        )
    )

    assert selected is not None
    assert selected.target.provider_model_ref == "open_router/550b"


def test_router_does_not_execute_unknown_free_status():
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="ollama_cloud",
                model_id="unknown",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=100.0,
                free_eligibility=FreeEligibility.UNKNOWN,
            ),
        )
    )

    router = SmartRouter(registry, RouteHealthStore())

    assert router.select((target("ollama_cloud/unknown"),)) is None


def test_router_skips_blocked_route():
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="groq",
                model_id="model",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )

    health = RouteHealthStore()
    health.get("groq/model").mark_blocked("not eligible")

    router = SmartRouter(registry, health)

    assert router.select((target("groq/model"),)) is None


def test_router_skips_active_backoff():
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="groq",
                model_id="model",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )

    health = RouteHealthStore()
    health.get("groq/model").mark_failure(
        failure_kind="timeout",
        retry_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    router = SmartRouter(registry, health)

    assert router.select((target("groq/model"),)) is None


def test_router_can_require_reasoning():
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="groq",
                model_id="no-reasoning",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
                supports_reasoning=False,
            ),
            ModelProfile(
                provider_id="open_router",
                model_id="reasoning",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=75.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
                supports_reasoning=True,
            ),
        )
    )

    router = SmartRouter(registry, RouteHealthStore())

    selected = router.select(
        (
            target("groq/no-reasoning"),
            target("open_router/reasoning"),
        ),
        requirements=RouteRequirements(requires_reasoning=True),
    )

    assert selected is not None
    assert selected.target.provider_model_ref == "open_router/reasoning"


def test_capability_tier_outranks_provider_diversity():
    # open_router is crowded (two candidates) so its diversity score is
    # worse, yet its TIER_1 candidate must still beat the TIER_2 candidates
    # from the unique groq provider.
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="groq",
                model_id="120b",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="open_router",
                model_id="120b",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="open_router",
                model_id="550b",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=100.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )

    router = SmartRouter(registry, RouteHealthStore())

    ranked = router.rank(
        (
            target("groq/120b"),
            target("open_router/120b"),
            target("open_router/550b"),
        )
    )

    assert [route.target.provider_model_ref for route in ranked] == [
        "open_router/550b",
        "groq/120b",
        "open_router/120b",
    ]


def test_provider_diversity_breaks_capability_score_ties():
    # Within the same tier and capability score, the distinct provider wins.
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="groq",
                model_id="model-a",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="groq",
                model_id="model-b",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="cloudflare",
                model_id="model-c",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )

    router = SmartRouter(registry, RouteHealthStore())

    ranked = router.rank(
        (
            target("groq/model-a"),
            target("groq/model-b"),
            target("cloudflare/model-c"),
        )
    )

    assert ranked[0].target.provider_id == "cloudflare"


def test_health_excludes_otherwise_verified_free_route():
    # A verified-free route in active backoff is not executable even though
    # its eligibility is VERIFIED_FREE.
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="groq",
                model_id="model",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=100.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )

    health = RouteHealthStore()
    health.get("groq/model").mark_failure(
        failure_kind="quota_exhausted",
        quarantine_until=datetime.now(UTC) + timedelta(minutes=30),
    )

    router = SmartRouter(registry, health)

    assert router.select((target("groq/model"),)) is None


def test_rank_is_pure_and_does_not_mutate_health_or_registry():
    # Smart Router ranking is local decision logic: it must not make provider
    # calls or mutate route health / registry state.
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="groq",
                model_id="model-a",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="groq",
                model_id="model-b",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="cloudflare",
                model_id="model-c",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )
    health = RouteHealthStore()
    router = SmartRouter(registry, health)
    targets = (
        target("groq/model-a"),
        target("groq/model-b"),
        target("cloudflare/model-c"),
    )

    before_registry = registry.all_profiles()
    ranked = router.rank(targets)

    # Deterministic local result without side effects.
    assert [r.target.provider_model_ref for r in ranked] == [
        "cloudflare/model-c",
        "groq/model-a",
        "groq/model-b",
    ]
    assert registry.all_profiles() == before_registry
    assert all(h.state.value == "unknown" for h in health.all())


def test_router_can_exclude_provider():
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="groq",
                model_id="model-a",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="cloudflare",
                model_id="model-b",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=75.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )

    router = SmartRouter(registry, RouteHealthStore())

    selected = router.select(
        (
            target("groq/model-a"),
            target("cloudflare/model-b"),
        ),
        excluded_providers=frozenset({"groq"}),
    )

    assert selected is not None
    assert selected.target.provider_id == "cloudflare"
