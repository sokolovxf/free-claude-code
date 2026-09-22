"""Read-only health/routing status snapshot for observability.

This module builds a JSON snapshot of the current local routing and health
state purely from in-memory state. It performs no provider/network calls and
mutates no health or registry state. Registry synchronization belongs to the
request/runtime path; this module only observes the shared state.
It is observability only — never a second routing engine.
"""

from free_claude_code.application.model_registry import ModelRegistry
from free_claude_code.application.route_health import RouteHealthStore
from free_claude_code.application.routing import ProviderModelTarget
from free_claude_code.application.smart_router import RouteExplanation, SmartRouter
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.core.json_types import JsonObject

_ROUTE_STATES = ("available", "backoff", "quarantined", "blocked", "unknown")


def build_router_status(
    *,
    registry: ModelRegistry,
    health: RouteHealthStore,
    router: SmartRouter,
    settings: object,
) -> JsonObject:
    """Return the current read-only routing and health snapshot.

    Configured targets are read from settings and rendered against the current
    shared registry and health state. Synchronization belongs to the request
    path and is intentionally not performed here.
    """
    targets = tuple(
        ProviderModelTarget(
            provider_id=ref.provider_id,
            provider_model=ref.model_id,
            provider_model_ref=ref.model_ref,
        )
        for ref in configured_chat_model_refs(settings)
    )

    explanations = router.explain(targets)
    routes = [
        _route_payload(
            explanation,
            quota_health=(
                health.get_quota_bucket(explanation.profile.quota_bucket)
                if explanation.profile and explanation.profile.quota_bucket
                else None
            ),
        )
        for explanation in explanations
    ]

    # SmartRouter.select() is the authoritative selection API. The status
    # surface must not infer selection from configuration order or from its
    # diagnostic representation.
    selected_route = router.select(targets)
    selected_ref = (
        selected_route.target.provider_model_ref if selected_route is not None else None
    )
    selected_payload = next(
        (route for route in routes if route["provider_model_ref"] == selected_ref),
        None,
    )
    # Health/quota state can change between the diagnostic pass and the
    # authoritative select call (background probes run concurrently). Never
    # advertise a route that the snapshot itself marks non-executable.
    if selected_payload is None or not selected_payload["executable"]:
        selected_payload = next(
            (route for route in routes if route["executable"]),
            None,
        )
        selected_ref = (
            selected_payload["provider_model_ref"] if selected_payload else None
        )

    return {
        "routes": routes,
        "selected": selected_ref,
        "selected_rank": selected_payload["rank"] if selected_payload else None,
        "last_successful": _last_successful(routes),
        "summary": _summary(routes),
    }


def _route_payload(explanation: RouteExplanation, *, quota_health) -> JsonObject:
    profile = explanation.profile
    return {
        "provider_model_ref": explanation.target.provider_model_ref,
        "provider": explanation.target.provider_id,
        "model": explanation.target.provider_model,
        "quota_bucket": profile.quota_bucket if profile else None,
        "quota": {
            "bucket": profile.quota_bucket if profile else None,
            "state": quota_health.state.value if quota_health else "unknown",
            "reset_at": (
                quota_health.quarantine_until.isoformat()
                if quota_health and quota_health.quarantine_until
                else None
            ),
        },
        "capability": {
            "tier": int(profile.capability_tier) if profile else None,
            "tier_name": profile.capability_tier.name if profile else None,
            "capability_score": profile.capability_score if profile else None,
            "supports_reasoning": profile.supports_reasoning if profile else None,
            "supports_tools": profile.supports_tools if profile else None,
            "context_window_tokens": (
                profile.context_window_tokens if profile else None
            ),
            "known": profile is not None,
        },
        "free": {
            "eligibility": profile.free_eligibility.value if profile else None,
            "verification_source": profile.verification_source if profile else None,
            "executable_for_zero_cost": bool(
                profile and profile.executable_for_zero_cost
            ),
        },
        "health": _health_payload(explanation.health),
        "executable": explanation.executable,
        "exclusion_reason": (
            explanation.reason.value if explanation.reason else None
        ),
        "rank": explanation.rank,
    }


def _health_payload(health) -> JsonObject:
    """Serialize health using the store's canonical ``to_dict`` shape."""
    payload = dict(health.to_dict())
    payload.pop("route_ref", None)  # provider_model_ref is the top-level key.
    return payload


def _summary(routes: list[JsonObject]) -> JsonObject:
    by_state = dict.fromkeys(_ROUTE_STATES, 0)
    for route in routes:
        state = route["health"]["state"]
        if state in by_state:
            by_state[state] += 1
    return {
        "total": len(routes),
        "executable": sum(1 for route in routes if route["executable"]),
        "by_state": by_state,
    }


def _last_successful(routes: list[JsonObject]) -> JsonObject | None:
    """Return the most recently completed route, if health has observed one."""
    candidates = [
        route
        for route in routes
        if route["health"].get("last_success_at") is not None
    ]
    if not candidates:
        return None
    route = max(candidates, key=lambda item: item["health"]["last_success_at"])
    return {
        "provider_model_ref": route["provider_model_ref"],
        "at": route["health"]["last_success_at"],
        "latency_ms": route["health"].get("observed_latency_ms"),
    }
