"""Capability-first, hard-$0 route selection for Smart Router v2."""

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

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


class ExclusionReason(StrEnum):
    """Read-only reason a candidate route is not currently executable.

    Values mirror the exact filter rules SmartRouter.rank() applies, in the
    same order. ``UNKNOWN_HEALTH`` is reserved for schema stability: an
    UNKNOWN-health route is treated as usable by SmartRouter and so never
    yields an exclusion today.
    """

    NOT_REGISTERED = "NOT_REGISTERED"
    NOT_VERIFIED_FREE = "NOT_VERIFIED_FREE"
    BELOW_MINIMUM_TIER = "BELOW_MINIMUM_TIER"
    BELOW_MINIMUM_CAPABILITY_SCORE = "BELOW_MINIMUM_CAPABILITY_SCORE"
    REASONING_REQUIRED = "REASONING_REQUIRED"
    TOOLS_REQUIRED = "TOOLS_REQUIRED"
    PROVIDER_EXCLUDED = "PROVIDER_EXCLUDED"
    BLOCKED = "BLOCKED"
    BACKOFF = "BACKOFF"
    QUARANTINED = "QUARANTINED"
    UNKNOWN_HEALTH = "UNKNOWN_HEALTH"


@dataclass(frozen=True, slots=True)
class RouteExplanation:
    """Read-only explanation of one candidate against current requirements.

    ``reason`` is None exactly when the route is executable; ``rank`` is the
    1-based position among executable candidates (None when not executable).
    This is a pure diagnostic view — it never mutates registry or health state.
    """

    target: ProviderModelTarget
    profile: ModelProfile | None
    health: RouteHealth
    reason: ExclusionReason | None
    rank: int | None

    @property
    def executable(self) -> bool:
        return self.reason is None


@dataclass(frozen=True, slots=True)
class _Evaluation:
    """Internal per-target evaluation shared by rank() and explain()."""

    target: ProviderModelTarget
    profile: ModelProfile | None
    health: RouteHealth
    reason: ExclusionReason | None


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
            evaluation = self._evaluate(
                target,
                requirements=requirements,
                excluded_providers=excluded_providers,
                now=now,
            )
            if evaluation.reason is not None:
                continue

            profile = evaluation.profile
            candidates.append(
                RankedRoute(
                    target=target,
                    profile=profile,
                    health=evaluation.health,
                    tier_score=int(profile.capability_tier),
                    capability_score=profile.capability_score,
                    provider_diversity_score=0,
                )
            )

        return self._apply_provider_diversity(candidates)

    def explain(
        self,
        targets: tuple[ProviderModelTarget, ...],
        *,
        requirements: RouteRequirements | None = None,
        excluded_providers: frozenset[str] = frozenset(),
    ) -> tuple[RouteExplanation, ...]:
        """Explain every candidate without changing routing behavior.

        Read-only diagnostics: returns one :class:`RouteExplanation` per
        target, using the exact same filter rules as :meth:`rank`. Executable
        routes carry a rank matching the provider-diversity ordering that
        :meth:`rank` returns; excluded routes carry an exclusion reason.
        """
        requirements = requirements or RouteRequirements()
        now = self._now()

        evaluations = tuple(
            self._evaluate(
                target,
                requirements=requirements,
                excluded_providers=excluded_providers,
                now=now,
            )
            for target in targets
        )

        ranked = self._apply_provider_diversity(
            [
                RankedRoute(
                    target=evaluation.target,
                    profile=evaluation.profile,
                    health=evaluation.health,
                    tier_score=int(evaluation.profile.capability_tier),
                    capability_score=evaluation.profile.capability_score,
                    provider_diversity_score=0,
                )
                for evaluation in evaluations
                if evaluation.reason is None
            ]
        )
        rank_by_ref = {
            route.target.provider_model_ref: index + 1
            for index, route in enumerate(ranked)
        }

        return tuple(
            RouteExplanation(
                target=evaluation.target,
                profile=evaluation.profile,
                health=evaluation.health,
                reason=evaluation.reason,
                rank=rank_by_ref.get(evaluation.target.provider_model_ref),
            )
            for evaluation in evaluations
        )

    def _evaluate(
        self,
        target: ProviderModelTarget,
        *,
        requirements: RouteRequirements,
        excluded_providers: frozenset[str],
        now: datetime,
    ) -> _Evaluation:
        """Classify one target as executable or excluded with a reason.

        This mirrors the exact filter order of :meth:`rank` so that the
        read-only explanation and the live ranking always agree.
        """
        profile = self.registry.get(target.provider_model_ref)

        if profile is None:
            return _Evaluation(
                target,
                None,
                self.health.get(target.provider_model_ref),
                ExclusionReason.NOT_REGISTERED,
            )

        if profile.free_eligibility is not FreeEligibility.VERIFIED_FREE:
            return _Evaluation(
                target,
                profile,
                self.health.get(target.provider_model_ref),
                ExclusionReason.NOT_VERIFIED_FREE,
            )

        if profile.capability_tier > requirements.minimum_tier:
            return _Evaluation(
                target,
                profile,
                self.health.get(target.provider_model_ref),
                ExclusionReason.BELOW_MINIMUM_TIER,
            )

        if profile.capability_score < requirements.minimum_capability_score:
            return _Evaluation(
                target,
                profile,
                self.health.get(target.provider_model_ref),
                ExclusionReason.BELOW_MINIMUM_CAPABILITY_SCORE,
            )

        if (
            requirements.requires_reasoning
            and profile.supports_reasoning is not True
        ):
            return _Evaluation(
                target,
                profile,
                self.health.get(target.provider_model_ref),
                ExclusionReason.REASONING_REQUIRED,
            )

        if requirements.requires_tools and profile.supports_tools is not True:
            return _Evaluation(
                target,
                profile,
                self.health.get(target.provider_model_ref),
                ExclusionReason.TOOLS_REQUIRED,
            )

        if target.provider_id in excluded_providers:
            return _Evaluation(
                target,
                profile,
                self.health.get(target.provider_model_ref),
                ExclusionReason.PROVIDER_EXCLUDED,
            )

        route_health = self.health.get(target.provider_model_ref)

        if route_health.state is RouteState.BLOCKED:
            return _Evaluation(
                target,
                profile,
                route_health,
                ExclusionReason.BLOCKED,
            )

        if not route_health.is_usable(now):
            reason = (
                ExclusionReason.BACKOFF
                if route_health.state is RouteState.BACKOFF
                else ExclusionReason.QUARANTINED
            )
            return _Evaluation(target, profile, route_health, reason)

        return _Evaluation(target, profile, route_health, None)

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
