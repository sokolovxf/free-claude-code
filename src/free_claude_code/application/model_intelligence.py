"""Static model intelligence for Smart Router v2.

This module contains model capability metadata only.
Live availability, quota, and health belong to route_health.py.

Free eligibility remains conservative: a route is executable under the
hard-$0 policy only when explicitly verified as free.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model_registry import CapabilityTier, ModelProfile


@dataclass(frozen=True, slots=True)
class ModelIntelligence:
    """Static intelligence metadata for a model family."""

    match: str
    capability_tier: CapabilityTier
    capability_score: float
    supports_reasoning: bool | None = None
    supports_tools: bool | None = None
    context_window_tokens: int | None = None


MODEL_INTELLIGENCE: tuple[ModelIntelligence, ...] = (
    ModelIntelligence(
        match="nemotron-3-ultra-550b",
        capability_tier=CapabilityTier.TIER_1,
        capability_score=100.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="nemotron-3-super-120b",
        capability_tier=CapabilityTier.TIER_2,
        capability_score=90.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="gpt-oss:120b",
        capability_tier=CapabilityTier.TIER_2,
        capability_score=88.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="gpt-oss-120b",
        capability_tier=CapabilityTier.TIER_2,
        capability_score=88.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="deepseek-v4-flash",
        capability_tier=CapabilityTier.TIER_2,
        capability_score=87.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="glm-5-3",
        capability_tier=CapabilityTier.TIER_2,
        capability_score=86.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="nemotron-3-nano",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=70.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="gemma-4-31b",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=68.0,
        supports_reasoning=False,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="qwen3.8-27b",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=67.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="qwen3-30b",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=66.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="gemma-4-26b",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=64.0,
        supports_reasoning=False,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="nex-n2.5-pro",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=63.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="nemotron-3.5-lightning",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=62.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="laguna-s-2.1",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=60.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="laguna-xs-2.1",
        capability_tier=CapabilityTier.TIER_4,
        capability_score=50.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="nex-n2.5-mini",
        capability_tier=CapabilityTier.TIER_4,
        capability_score=48.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="north-mini-code",
        capability_tier=CapabilityTier.TIER_4,
        capability_score=47.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="inkling",
        capability_tier=CapabilityTier.TIER_4,
        capability_score=45.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="ling-3.0-flash-fin",
        capability_tier=CapabilityTier.TIER_4,
        capability_score=44.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="compound",
        capability_tier=CapabilityTier.TIER_3,
        capability_score=65.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
    ModelIntelligence(
        match="compound-mini",
        capability_tier=CapabilityTier.TIER_4,
        capability_score=46.0,
        supports_reasoning=True,
        supports_tools=True,
    ),
)


def intelligence_for_route(route_ref: str) -> ModelIntelligence | None:
    """Return static intelligence for a known model route."""
    normalized = route_ref.casefold()

    return next(
        (entry for entry in MODEL_INTELLIGENCE if entry.match.casefold() in normalized),
        None,
    )


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
