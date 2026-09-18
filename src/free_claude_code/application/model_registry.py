"""Static model intelligence for Smart Router v2.

This module deliberately contains no live provider calls. It describes what
FCC knows about a route before health/quota observations are considered.
"""

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class CapabilityTier(IntEnum):
    """Relative intelligence tiers used by Smart Router v2."""

    TIER_4 = 4
    TIER_3 = 3
    TIER_2 = 2
    TIER_1 = 1


class FreeEligibility(StrEnum):
    """Whether FCC is allowed to execute a route under hard-$0 policy."""

    VERIFIED_FREE = "verified_free"
    UNKNOWN = "unknown"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Static intelligence and eligibility metadata for one route."""

    provider_id: str
    model_id: str
    capability_tier: CapabilityTier
    capability_score: float
    free_eligibility: FreeEligibility = FreeEligibility.UNKNOWN
    verification_source: str | None = None
    last_verified: str | None = None
    supports_reasoning: bool | None = None
    supports_tools: bool | None = None
    context_window_tokens: int | None = None

    @property
    def route_ref(self) -> str:
        """Return the canonical FCC provider/model reference."""
        return f"{self.provider_id}/{self.model_id}"

    @property
    def executable_for_zero_cost(self) -> bool:
        """Return whether this route may execute under hard-$0 policy."""
        return self.free_eligibility is FreeEligibility.VERIFIED_FREE


class ModelRegistry:
    """In-memory registry of static route intelligence.

    The registry is intentionally independent from Settings and route health.
    Settings supplies the candidate inventory; this registry supplies the
    intelligence needed to rank that inventory.
    """

    def __init__(
        self,
        profiles: tuple[ModelProfile, ...] = (),
    ) -> None:
        self._profiles = {profile.route_ref: profile for profile in profiles}

    def get(self, route_ref: str) -> ModelProfile | None:
        """Return metadata for a route, if known."""
        return self._profiles.get(route_ref)

    def register(self, profile: ModelProfile) -> None:
        """Add or replace a route profile."""
        self._profiles[profile.route_ref] = profile

    def all_profiles(self) -> tuple[ModelProfile, ...]:
        """Return all registered profiles."""
        return tuple(self._profiles.values())

    def profiles_for(
        self,
        route_refs: tuple[str, ...],
    ) -> tuple[ModelProfile, ...]:
        """Return known profiles for the supplied route references."""
        return tuple(
            profile
            for route_ref in route_refs
            if (profile := self.get(route_ref)) is not None
        )

    def executable_profiles(
        self,
        route_refs: tuple[str, ...],
    ) -> tuple[ModelProfile, ...]:
        """Return routes explicitly verified as free."""
        return tuple(
            profile
            for profile in self.profiles_for(route_refs)
            if profile.executable_for_zero_cost
        )
