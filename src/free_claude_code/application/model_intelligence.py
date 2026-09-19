"""Static model intelligence for Smart Router v2.

This module contains model capability metadata only.
Live availability, quota, and health belong to route_health.py.

Free eligibility remains conservative: a route is executable under the
hard-$0 policy only when explicitly verified as free.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from free_claude_code.config.model_refs import configured_chat_model_refs

from .model_registry import CapabilityTier, FreeEligibility, ModelProfile, ModelRegistry


@dataclass(frozen=True, slots=True)
class ModelIntelligence:
    """Static intelligence metadata for a model family.

    ``family`` is the canonical, human-readable name of the model family
    (for example ``"nemotron-3-ultra"``). Matching is token-based so that
    every provider route exposing that family — regardless of provider
    prefix, vendor sub-path, ``:free``/``:30b`` style suffix, or size
    qualifier in the route — is classified the same way.
    """

    family: str
    capability_tier: CapabilityTier
    capability_score: float
    supports_reasoning: bool | None = None
    supports_tools: bool | None = None
    context_window_tokens: int | None = None


MODEL_INTELLIGENCE: tuple[ModelIntelligence, ...] = (
    ModelIntelligence(
        family="nemotron-3-ultra",
        capability_tier=CapabilityTier.TIER_1,
        capability_score=100.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="nemotron-3-super",
        capability_tier=CapabilityTier.TIER_2,
        capability_score=90.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="gpt-oss",
        capability_tier=CapabilityTier.TIER_2,
        capability_score=88.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="deepseek-v4-flash",
        capability_tier=CapabilityTier.TIER_2,
        capability_score=87.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="glm-5-3",
        capability_tier=CapabilityTier.TIER_2,
        capability_score=86.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="nemotron-3-nano",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=70.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="gemma-4-31b",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=68.0,
        supports_reasoning=False,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="qwen3.8-27b",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=67.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="qwen3-30b",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=66.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="gemma-4-26b",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=64.0,
        supports_reasoning=False,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="nex-n2.5-pro",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=63.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="nemotron-3.5-lightning",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=62.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="laguna-s-2.1",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=60.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="laguna-xs-2.1",
        capability_tier=CapabilityTier.TIER_4,
        capability_score=50.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="nex-n2.5-mini",
        capability_tier=CapabilityTier.TIER_4,
        capability_score=48.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="north-mini-code",
        capability_tier=CapabilityTier.TIER_4,
        capability_score=47.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="inkling",
        capability_tier=CapabilityTier.TIER_4,
        capability_score=45.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="ling-3.0-flash-fin",
        capability_tier=CapabilityTier.TIER_4,
        capability_score=44.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="compound",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=65.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        family="compound-mini",
        capability_tier=CapabilityTier.TIER_4,
        capability_score=46.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
)

_TOKEN_PATTERN = re.compile(r"[^0-9a-z]+")


def _tokens(value: str) -> tuple[str, ...]:
    """Split a model/family string into significant lowercase tokens."""
    return tuple(
        part
        for part in _TOKEN_PATTERN.split(value.casefold())
        if part
    )


def _is_contiguous_subsequence(
    sub: tuple[str, ...],
    seq: tuple[str, ...],
) -> bool:
    """Return whether ``sub`` appears contiguously within ``seq``."""
    width = len(sub)
    if not width:
        return True
    if width > len(seq):
        return False
    return any(
        tuple(seq[index : index + width]) == sub
        for index in range(len(seq) - width + 1)
    )


def intelligence_for_route(route_ref: str) -> ModelIntelligence | None:
    """Return static intelligence for a known model family route.

    Matching is token-based and position-independent with respect to the
    configured fallback order: the provider/model identifier is tokenized and
    the most specific (longest-token) catalog family whose tokens appear
    contiguously is chosen. This lets different provider routes for the same
    family (for example ``open_router/nvidia/nemotron-3-ultra-550b-a55b:free``
    and ``ollama_cloud/nemotron-3-ultra``) share a classification while an
    unknown model stays unmatched rather than being guessed.
    """
    _, separator, model_id = route_ref.partition("/")
    if not separator:
        return None

    model_tokens = _tokens(model_id)

    best: ModelIntelligence | None = None
    best_tokens = -1
    for entry in MODEL_INTELLIGENCE:
        family_tokens = _tokens(entry.family)
        if len(family_tokens) <= best_tokens:
            continue
        if _is_contiguous_subsequence(family_tokens, model_tokens):
            best = entry
            best_tokens = len(family_tokens)

    return best


def profile_for_route(route_ref: str) -> ModelProfile:
    """Build a conservative ModelProfile for one route."""
    provider_id, separator, model_id = route_ref.partition("/")

    if not separator or not provider_id or not model_id:
        raise ValueError(f"Invalid provider/model route: {route_ref!r}")

    intelligence = intelligence_for_route(route_ref)

    if intelligence is None:
        return ModelProfile(
            provider_id=provider_id,
            model_id=model_id,
            capability_tier=CapabilityTier.TIER_4,
            capability_score=0.0,
        )

    return ModelProfile(
        provider_id=provider_id,
        model_id=model_id,
        capability_tier=intelligence.capability_tier,
        capability_score=intelligence.capability_score,
        supports_reasoning=intelligence.supports_reasoning,
        supports_tools=intelligence.supports_tools,
        context_window_tokens=intelligence.context_window_tokens,
    )


def profile_for_configured_route(
    route_ref: str,
    verified_free_models: Sequence[str],
    *,
    now: datetime | None = None,
) -> ModelProfile:
    """Build a profile for one configured route with explicit free eligibility.

    Uses :func:`profile_for_route` for static capability metadata. Free
    eligibility is derived only from the operator's explicit
    ``FCC_VERIFIED_FREE_MODELS`` allow-list; an exact match is marked
    ``VERIFIED_FREE`` and anything else stays ``UNKNOWN``.
    """
    profile = profile_for_route(route_ref)

    if route_ref in verified_free_models:
        return replace(
            profile,
            free_eligibility=FreeEligibility.VERIFIED_FREE,
            verification_source="FCC_VERIFIED_FREE_MODELS",
            last_verified=(now or datetime.now(UTC)).isoformat(),
        )

    return profile


def synchronize_model_registry(registry: ModelRegistry, settings: object) -> None:
    """Rebuild the shared registry to mirror the active Settings configuration.

    Configuration remains the source of truth for which routes exist. Every
    configured ``MODEL`` / ``MODEL_*`` override / ``MODEL_FALLBACKS`` route is
    discovered through :func:`configured_chat_model_refs` and registered with
    its static capability metadata plus explicit free eligibility. Rebuilding
    (rather than only appending) means a route removed from configuration is
    no longer present as a candidate, and a newly added route appears without
    any Smart Router changes.
    """
    verified = tuple(settings.verified_free_models or ())  # type: ignore[attr-defined]

    profiles = []
    for ref in configured_chat_model_refs(settings):  # type: ignore[arg-type]
        profile = profile_for_configured_route(ref.model_ref, verified)
        existing = registry.get(ref.model_ref)
        if existing is not None:
            profile = replace(
                profile,
                capability_tier=existing.capability_tier,
                capability_score=existing.capability_score,
                supports_reasoning=existing.supports_reasoning,
                supports_tools=existing.supports_tools,
                context_window_tokens=existing.context_window_tokens,
            )
        profiles.append(profile)

    registry.replace_all(profiles)
