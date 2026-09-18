"""Capability-first, hard-$0 route selection for Smart Router v2."""

from dataclasses import dataclass
from datetime import UTC, datetime

from free_claude_code.application.routing import ProviderModelTarget

from .model_registry import (
    CapabilityTier,
    FreeEligibility,
    ModelProfile,
    ModelRegistry,
)
from .route_health import RouteHealth, RouteHealthStore, RouteState


@dataclass(frozen=True, slots=True)
class RouteRequirements:
    """Requirements extracted from an incoming request."""

    minimum_tier: CapabilityTier = CapabilityTier.TIER_4
    minimum_capability_score: float = 0.0
    requires_reasoning: bool = False
    requires_tools: bool = False


@dataclass(frozen=True, slots=True)
class RankedRoute:
    """A candidate plus the evidence used to rank it."""

    target: ProviderModelTarget
    profile: ModelProfile
    health: RouteHealth
    tier_score: int
    capability_score: float
    provider_diversity_score: int


class SmartRouter:
    """Select routes using capability, free eligibility, health and diversity."""

    def __init__(
        self,
        registry: ModelRegistry,
        health: RouteHealthStore,
    ) -> None:
        self.registry = registry
        self.health = health

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def rank(
        self,
        targets: tuple[ProviderModelTarget, ...],
        *,
        requirements: RouteRequirements | None = None,
        excluded_providers: frozenset[str] = frozenset(),
    ) -> tuple[RankedRoute, ...]:
        """Rank executable candidates without making a provider call."""
        requirements = requirements or RouteRequirements()
        now = self._now()

        candidates: list[RankedRoute] = []

        for target in targets:
            profile = self.registry.get(target.provider_model_ref)

            if profile is None:
                continue

            if profile.free_eligibility is not FreeEligibility.VERIFIED_FREE:
                continue

            if profile.capability_tier > requirements.minimum_tier:
                continue

            if profile.capability_score < requirements.minimum_capability_score:
                continue

            if (
                requirements.requires_reasoning
                and profile.supports_reasoning is not True
            ):
                continue

            if requirements.requires_tools and profile.supports_tools is not True:
                continue

            if target.provider_id in excluded_providers:
                continue

            route_health = self.health.get(target.provider_model_ref)

            if route_health.state is RouteState.BLOCKED:
                continue

            if not route_health.is_usable(now):
                continue

            candidates.append(
                RankedRoute(
                    target=target,
                    profile=profile,
                    health=route_health,
                    tier_score=int(profile.capability_tier),
                    capability_score=profile.capability_score,
                    provider_diversity_score=0,
                )
            )

        return self._apply_provider_diversity(candidates)

    def select(
        self,
        targets: tuple[ProviderModelTarget, ...],
        *,
        requirements: RouteRequirements | None = None,
        excluded_providers: frozenset[str] = frozenset(),
    ) -> RankedRoute | None:
        """Return the highest-ranked currently executable route."""
        ranked = self.rank(
            targets,
            requirements=requirements,
            excluded_providers=excluded_providers,
        )

        return ranked[0] if ranked else None

    @staticmethod
    def _apply_provider_diversity(
        candidates: list[RankedRoute],
    ) -> tuple[RankedRoute, ...]:
        """Prefer distinct providers when capability is otherwise comparable."""
        provider_counts: dict[str, int] = {}

        for candidate in candidates:
            provider_counts[candidate.target.provider_id] = (
                provider_counts.get(candidate.target.provider_id, 0) + 1
            )

        scored = [
            RankedRoute(
                target=candidate.target,
                profile=candidate.profile,
                health=candidate.health,
                tier_score=candidate.tier_score,
                capability_score=candidate.capability_score,
                provider_diversity_score=provider_counts[candidate.target.provider_id],
            )
            for candidate in candidates
        ]

        scored.sort(
            key=lambda candidate: (
                candidate.tier_score,
                -candidate.capability_score,
                candidate.provider_diversity_score,
                candidate.health.failure_count,
                candidate.health.observed_latency_ms
                if candidate.health.observed_latency_ms is not None
                else float("inf"),
            )
        )

        return tuple(scored)
