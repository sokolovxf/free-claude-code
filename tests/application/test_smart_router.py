from datetime import UTC, datetime, timedelta

from free_claude_code.application.model_registry import (
    CapabilityTier,
    FreeEligibility,
    ModelProfile,
    ModelRegistry,
)
from free_claude_code.application.route_health import RouteHealthStore, RouteState
from free_claude_code.application.routing import ProviderModelTarget
from free_claude_code.application.smart_router import (
    ExclusionReason,
    RouteRequirements,
    RoutingPolicy,
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


def test_router_executes_configured_unknown_cost_route_when_healthy():
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

    selected = router.select((target("ollama_cloud/unknown"),))

    assert selected is not None
    assert selected.target.provider_model_ref == "ollama_cloud/unknown"


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


def test_router_skips_routes_in_exhausted_shared_quota_bucket():
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="open_router",
                model_id="model-a",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=100.0,
                free_eligibility=FreeEligibility.UNKNOWN,
                quota_bucket="openrouter_free_tier_daily",
            ),
            ModelProfile(
                provider_id="open_router",
                model_id="model-b",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.UNKNOWN,
                quota_bucket="openrouter_free_tier_daily",
            ),
        )
    )
    health = RouteHealthStore()
    health.get_quota_bucket("openrouter_free_tier_daily").mark_quarantined()
    router = SmartRouter(registry, health)

    assert router.select((target("open_router/model-a"), target("open_router/model-b"))) is None


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


def test_router_prefers_measured_fast_route_within_same_tier():
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="nvidia_nim",
                model_id="super-120b",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=90.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="groq",
                model_id="gpt-oss-120b",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=88.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )
    health = RouteHealthStore()
    health.get("nvidia_nim/super-120b").mark_success(latency_ms=1_200.0)
    health.get("groq/gpt-oss-120b").mark_success(latency_ms=700.0)

    ranked = SmartRouter(registry, health).rank(
        (target("nvidia_nim/super-120b"), target("groq/gpt-oss-120b"))
    )

    assert [route.target.provider_model_ref for route in ranked] == [
        "groq/gpt-oss-120b",
        "nvidia_nim/super-120b",
    ]


def test_router_prefers_proven_route_over_unknown_peer_within_same_tier():
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="nvidia_nim",
                model_id="super-120b",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=90.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="groq",
                model_id="gpt-oss-120b",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=88.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )
    health = RouteHealthStore()
    health.get("groq/gpt-oss-120b").mark_success(latency_ms=700.0)

    ranked = SmartRouter(registry, health).rank(
        (target("nvidia_nim/super-120b"), target("groq/gpt-oss-120b"))
    )

    assert ranked[0].target.provider_model_ref == "groq/gpt-oss-120b"


def test_fastest_policy_can_choose_lower_tier_measured_route():
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="nvidia_nim",
                model_id="550b",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=100.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="groq",
                model_id="120b",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=88.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )
    health = RouteHealthStore()
    health.get("nvidia_nim/550b").mark_success(latency_ms=1_500.0)
    health.get("groq/120b").mark_success(latency_ms=400.0)

    router = SmartRouter(registry, health, policy=RoutingPolicy.FASTEST)

    selected = router.select((target("nvidia_nim/550b"), target("groq/120b")))

    assert selected is not None
    assert selected.target.provider_model_ref == "groq/120b"


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

def test_explain_returns_exclusion_reasons_for_all_targets():
    """explain() must return one explanation per input target with machine-readable reasons."""
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
                provider_id="open_router",
                model_id="model-b",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=100.0,
                free_eligibility=FreeEligibility.UNKNOWN,
            ),
        )
    )
    health = RouteHealthStore()
    router = SmartRouter(registry, health)

    explanations = router.explain(
        (
            target("groq/model-a"),
            target("open_router/model-b"),
            target("unknown/provider"),
        )
    )

    assert len(explanations) == 3
    # groq/model-a: configured and healthy -> executable
    assert explanations[0].executable is True
    assert explanations[0].reason is None
    assert explanations[0].rank == 2
    # open_router/model-b: configured and healthy -> executable
    assert explanations[1].executable is True
    assert explanations[1].reason is None
    assert explanations[1].rank == 1
    # unknown/provider: not in registry -> NOT_REGISTERED
    assert explanations[2].executable is False
    assert explanations[2].reason == ExclusionReason.NOT_REGISTERED
    assert explanations[2].rank is None


def test_explain_preserves_rank_order_matching_rank():
    """Explain's executable ranks must match rank() output exactly."""
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
                provider_id="open_router",
                model_id="model-b",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=100.0,
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
        target("open_router/model-b"),
        target("cloudflare/model-c"),
    )

    ranked = router.rank(targets)
    explained = router.explain(targets)

    # rank() returns TIER_1 first; TIER_2 ties resolved by stable input order
    ranked_refs = [r.target.provider_model_ref for r in ranked]
    assert ranked_refs == ["open_router/model-b", "groq/model-a", "cloudflare/model-c"]

    # explain() returns in input order; check rank mapping matches
    rank_by_ref = {e.target.provider_model_ref: e.rank for e in explained if e.executable}
    assert rank_by_ref == {
        "open_router/model-b": 1,
        "groq/model-a": 2,
        "cloudflare/model-c": 3,
    }


def test_explain_respects_excluded_providers():
    """explain() must honor excluded_providers parameter."""
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
                provider_id="open_router",
                model_id="model-b",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=100.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )
    health = RouteHealthStore()
    router = SmartRouter(registry, health)

    explained = router.explain(
        (target("groq/model-a"), target("open_router/model-b")),
        excluded_providers=frozenset({"groq"}),
    )

    # groq excluded -> PROVIDER_EXCLUDED
    assert next(e for e in explained if e.target.provider_model_ref == "groq/model-a").reason == ExclusionReason.PROVIDER_EXCLUDED
    # open_router not excluded -> executable rank 1
    assert next(e for e in explained if e.target.provider_model_ref == "open_router/model-b").reason is None
    assert next(e for e in explained if e.target.provider_model_ref == "open_router/model-b").rank == 1


def test_explain_respects_reasoning_requirement():
    """explain() must honor requires_reasoning filter."""
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
    health = RouteHealthStore()
    router = SmartRouter(registry, health)

    # No reasoning required -> both executable
    explained = router.explain(
        (target("groq/no-reasoning"), target("open_router/reasoning")),
        requirements=RouteRequirements(),
    )
    assert all(e.executable for e in explained)

    # Reasoning required -> groq excluded
    explained = router.explain(
        (target("groq/no-reasoning"), target("open_router/reasoning")),
        requirements=RouteRequirements(requires_reasoning=True),
    )
    groq_exp = next(e for e in explained if e.target.provider_model_ref == "groq/no-reasoning")
    or_exp = next(e for e in explained if e.target.provider_model_ref == "open_router/reasoning")
    assert groq_exp.reason == ExclusionReason.REASONING_REQUIRED
    assert or_exp.executable is True
    assert or_exp.rank == 1


def test_explain_respects_capability_requirements():
    """explain() must honor minimum_tier and minimum_capability_score."""
    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="groq",
                model_id="tier2-score80",
                capability_tier=CapabilityTier.TIER_2,
                capability_score=80.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="open_router",
                model_id="tier1-score90",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=90.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )
    health = RouteHealthStore()
    router = SmartRouter(registry, health)

    # Require TIER_1 -> groq excluded
    explained = router.explain(
        (target("groq/tier2-score80"), target("open_router/tier1-score90")),
        requirements=RouteRequirements(minimum_tier=CapabilityTier.TIER_1),
    )
    groq_exp = next(e for e in explained if e.target.provider_model_ref == "groq/tier2-score80")
    or_exp = next(e for e in explained if e.target.provider_model_ref == "open_router/tier1-score90")
    assert groq_exp.reason == ExclusionReason.BELOW_MINIMUM_TIER
    assert or_exp.executable is True

    # Require score >= 85 -> groq excluded
    explained = router.explain(
        (target("groq/tier2-score80"), target("open_router/tier1-score90")),
        requirements=RouteRequirements(minimum_capability_score=85.0),
    )
    groq_exp = next(e for e in explained if e.target.provider_model_ref == "groq/tier2-score80")
    or_exp = next(e for e in explained if e.target.provider_model_ref == "open_router/tier1-score90")
    assert groq_exp.reason == ExclusionReason.BELOW_MINIMUM_CAPABILITY_SCORE
    assert or_exp.executable is True


def test_explain_reflects_health_exclusions():
    """explain() must reflect health state exclusions (BLOCKED, BACKOFF, QUARANTINED)."""
    from datetime import UTC, datetime, timedelta

    registry = ModelRegistry(
        (
            ModelProfile(
                provider_id="groq",
                model_id="available",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=100.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="groq",
                model_id="blocked",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=100.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="groq",
                model_id="backoff",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=100.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
            ModelProfile(
                provider_id="groq",
                model_id="quarantined",
                capability_tier=CapabilityTier.TIER_1,
                capability_score=100.0,
                free_eligibility=FreeEligibility.VERIFIED_FREE,
            ),
        )
    )
    health = RouteHealthStore()
    router = SmartRouter(registry, health)

    # Mark states
    health.get("groq/blocked").mark_blocked("billing")
    health.get("groq/backoff").mark_failure(
        failure_kind="timeout", retry_at=datetime.now(UTC) + timedelta(minutes=5)
    )
    health.get("groq/quarantined").mark_quarantined(until=datetime.now(UTC) + timedelta(minutes=30))

    explained = router.explain(
        (
            target("groq/available"),
            target("groq/blocked"),
            target("groq/backoff"),
            target("groq/quarantined"),
        )
    )

    available = next(e for e in explained if e.target.provider_model_ref == "groq/available")
    blocked = next(e for e in explained if e.target.provider_model_ref == "groq/blocked")
    backoff = next(e for e in explained if e.target.provider_model_ref == "groq/backoff")
    quarantined = next(e for e in explained if e.target.provider_model_ref == "groq/quarantined")

    assert available.executable is True
    assert available.rank == 1
    assert blocked.reason == ExclusionReason.BLOCKED
    assert backoff.reason == ExclusionReason.BACKOFF
    assert quarantined.reason == ExclusionReason.QUARANTINED


def test_explain_is_pure_no_mutation():
    """explain() must not mutate registry or health state (pure diagnostic).

    Note: RouteHealthStore.get() auto-creates UNKNOWN entries on first access,
    which is expected read-side behavior, not a mutation.
    """
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
    router = SmartRouter(registry, health)

    before_registry = registry.all_profiles()

    router.explain((target("groq/model"),))

    assert registry.all_profiles() == before_registry
    # Health entries may be auto-created by get(), but their state must remain UNKNOWN
    for h in health.all():
        assert h.state is RouteState.UNKNOWN
