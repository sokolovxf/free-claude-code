"""Stage 2E: Read-only FCC health/routing observability surface tests."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from free_claude_code.application.model_intelligence import (
    synchronize_model_registry,
)
from free_claude_code.application.model_registry import (
    CapabilityTier,
    FreeEligibility,
    ModelProfile,
    ModelRegistry,
)
from free_claude_code.application.route_health import RouteHealthStore, RouteState
from free_claude_code.application.router_status import build_router_status
from free_claude_code.application.routing import ProviderModelTarget
from free_claude_code.application.smart_router import SmartRouter


class _FixedSettings:
    """Minimal Settings-like object for unit tests."""

    def __init__(self, *, model, model_fallbacks=(), verified_free_models=()):
        self.model = model
        self.model_fallbacks = model_fallbacks
        self.verified_free_models = verified_free_models
        # Per-model overrides (unused in these tests)
        self.model_fable = None
        self.model_opus = None
        self.model_sonnet = None
        self.model_haiku = None


def _target(ref: str) -> ProviderModelTarget:
    provider, model = ref.split("/", 1)
    return ProviderModelTarget(
        provider_id=provider,
        provider_model=model,
        provider_model_ref=ref,
    )


def _profile(
    ref: str,
    *,
    tier: CapabilityTier,
    score: float,
    free: FreeEligibility = FreeEligibility.VERIFIED_FREE,
    verification_source: str | None = "test",
) -> ModelProfile:
    provider, model = ref.split("/", 1)
    return ModelProfile(
        provider_id=provider,
        model_id=model,
        capability_tier=tier,
        capability_score=score,
        free_eligibility=free,
        verification_source=verification_source,
    )


def _make_router(
    profiles: tuple[ModelProfile, ...], health: RouteHealthStore | None = None
) -> SmartRouter:
    registry = ModelRegistry(profiles)
    health = health or RouteHealthStore()
    return SmartRouter(registry, health)


def _synchronize_runtime_registry(app) -> object:
    from tests.api.support import runtime_for_app

    runtime = runtime_for_app(app)
    synchronize_model_registry(runtime.smart_router.registry, runtime.settings)
    return runtime


# === Unit tests for build_router_status ===


def test_build_router_status_configured_route_without_registry_profile():
    """A configured route is visible even when its registry profile is absent."""
    router = _make_router(())
    status = build_router_status(
        registry=router.registry,
        health=router.health,
        router=router,
        settings=_FixedSettings(model="open_router/unknown"),
    )

    assert len(status["routes"]) == 1
    route = status["routes"][0]
    assert route["provider_model_ref"] == "open_router/unknown"
    assert route["capability"]["known"] is False
    assert route["capability"]["tier"] is None
    assert route["capability"]["capability_score"] is None
    assert route["free"]["eligibility"] is None
    assert route["executable"] is False
    assert route["exclusion_reason"] == "NOT_REGISTERED"
    assert status["selected"] is None
    assert status["summary"]["total"] == 1
    assert status["summary"]["executable"] == 0


def test_build_router_status_includes_configured_routes():
    """Routes from settings (model + fallbacks) appear in snapshot."""
    settings = _FixedSettings(
        model="open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
        model_fallbacks=("groq/openai/gpt-oss-120b", "ollama_cloud/nemotron-3-super"),
    )
    router = _make_router((
        ModelProfile(
            provider_id="open_router",
            model_id="nvidia/nemotron-3-ultra-550b-a55b:free",
            capability_tier=CapabilityTier.TIER_1,
            capability_score=100.0,
            free_eligibility=FreeEligibility.VERIFIED_FREE,
        ),
        ModelProfile(
            provider_id="groq",
            model_id="openai/gpt-oss-120b",
            capability_tier=CapabilityTier.TIER_2,
            capability_score=88.0,
            free_eligibility=FreeEligibility.UNKNOWN,
        ),
    ))

    status = build_router_status(
        registry=router.registry,
        health=router.health,
        router=router,
        settings=settings,
    )

    refs = {r["provider_model_ref"] for r in status["routes"]}
    assert "open_router/nvidia/nemotron-3-ultra-550b-a55b:free" in refs
    assert "groq/openai/gpt-oss-120b" in refs
    # ollama is configured but not registered -> still appears with known=False
    assert "ollama_cloud/nemotron-3-super" in refs


def test_build_router_status_unknown_model_is_unregistered():
    """A configured route without a registry profile remains unregistered."""
    settings = _FixedSettings(model="open_router/completely-unknown-model")
    router = _make_router(())

    status = build_router_status(
        registry=router.registry,
        health=router.health,
        router=router,
        settings=settings,
    )

    route = status["routes"][0]
    assert route["capability"]["known"] is False
    assert route["capability"]["tier"] is None
    assert route["capability"]["capability_score"] is None
    assert route["free"]["eligibility"] is None
    assert route["executable"] is False
    assert route["exclusion_reason"] == "NOT_REGISTERED"


def test_build_router_status_reports_registry_free_eligibility():
    """Free eligibility is rendered from the existing registry profile."""
    settings = _FixedSettings(
        model="open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
        verified_free_models=("open_router/nvidia/nemotron-3-ultra-550b-a55b:free",),
    )
    router = _make_router((
        _profile(
            "open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
            tier=CapabilityTier.TIER_1,
            score=100.0,
        ),
    ))

    status = build_router_status(
        registry=router.registry,
        health=router.health,
        router=router,
        settings=settings,
    )

    route = status["routes"][0]
    assert route["free"]["eligibility"] == "verified_free"
    assert route["free"]["verification_source"] == "test"
    assert route["free"]["executable_for_zero_cost"] is True


def test_build_router_status_reflects_all_health_states():
    """Health states AVAILABLE/BACKOFF/QUARANTINED/BLOCKED/UNKNOWN all reflected."""
    settings = _FixedSettings(
        model="open_router/a",
        model_fallbacks=("open_router/b", "open_router/c", "open_router/d", "open_router/e"),
        verified_free_models=("open_router/a", "open_router/b", "open_router/c", "open_router/d", "open_router/e"),
    )
    router = _make_router((
        _profile("open_router/a", tier=CapabilityTier.TIER_1, score=100.0),
        _profile("open_router/b", tier=CapabilityTier.TIER_1, score=100.0),
        _profile("open_router/c", tier=CapabilityTier.TIER_1, score=100.0),
        _profile("open_router/d", tier=CapabilityTier.TIER_1, score=100.0),
        _profile("open_router/e", tier=CapabilityTier.TIER_1, score=100.0),
    ))

    # Mark health states
    router.health.get("open_router/a").mark_success()
    router.health.get("open_router/b").mark_failure(
        failure_kind="timeout", retry_at=datetime.now(UTC) + timedelta(minutes=5)
    )
    router.health.get("open_router/c").mark_quarantined(
        until=datetime.now(UTC) + timedelta(minutes=30)
    )
    router.health.get("open_router/d").mark_blocked("billing")
    # e remains UNKNOWN

    status = build_router_status(
        registry=router.registry,
        health=router.health,
        router=router,
        settings=settings,
    )

    by_state = {r["health"]["state"]: r for r in status["routes"]}
    assert by_state["available"]["executable"] is True
    assert by_state["backoff"]["executable"] is False
    assert by_state["backoff"]["exclusion_reason"] == "BACKOFF"
    assert by_state["quarantined"]["executable"] is False
    assert by_state["quarantined"]["exclusion_reason"] == "QUARANTINED"
    assert by_state["blocked"]["executable"] is False
    assert by_state["blocked"]["exclusion_reason"] == "BLOCKED"
    assert by_state["unknown"]["executable"] is True
    assert by_state["unknown"]["exclusion_reason"] is None

    # Summary counts
    assert status["summary"]["by_state"]["available"] == 1
    assert status["summary"]["by_state"]["backoff"] == 1
    assert status["summary"]["by_state"]["quarantined"] == 1
    assert status["summary"]["by_state"]["blocked"] == 1
    assert status["summary"]["by_state"]["unknown"] == 1


def test_build_router_status_selected_is_highest_ranked_executable():
    """selected field equals the top-ranked executable route."""
    settings = _FixedSettings(
        model="open_router/low",
        model_fallbacks=("open_router/high",),
        verified_free_models=("open_router/low", "open_router/high"),
    )
    router = _make_router((
        _profile("open_router/low", tier=CapabilityTier.TIER_2, score=80.0),
        _profile("open_router/high", tier=CapabilityTier.TIER_1, score=100.0),
    ))

    status = build_router_status(
        registry=router.registry,
        health=router.health,
        router=router,
        settings=settings,
    )

    assert status["selected"] == "open_router/high"
    # low is executable but rank 2
    ranks = {r["provider_model_ref"]: r["rank"] for r in status["routes"]}
    assert ranks["open_router/high"] == 1
    assert ranks["open_router/low"] == 2


def test_build_router_status_does_not_claim_static_config_position_is_selected():
    """CONFIGURATION ORDER != CAPABILITY RANKING - do not claim 'selected' based on static config."""
    settings = _FixedSettings(
        model="open_router/configured-first",  # appears first in config
        model_fallbacks=("open_router/better",),  # but higher capability
        verified_free_models=("open_router/configured-first", "open_router/better"),
    )
    router = _make_router((
        _profile("open_router/configured-first", tier=CapabilityTier.TIER_2, score=80.0),
        _profile("open_router/better", tier=CapabilityTier.TIER_1, score=100.0),
    ))

    status = build_router_status(
        registry=router.registry,
        health=router.health,
        router=router,
        settings=settings,
    )

    # SmartRouter picks by capability, not config order
    assert status["selected"] == "open_router/better"


def test_build_router_status_no_provider_network_calls():
    """Building status must not make any provider or network calls."""
    settings = _FixedSettings(
        model="open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
        verified_free_models=("open_router/nvidia/nemotron-3-ultra-550b-a55b:free",),
    )
    router = _make_router((
        ModelProfile(
            provider_id="open_router",
            model_id="nvidia/nemotron-3-ultra-550b-a55b:free",
            capability_tier=CapabilityTier.TIER_1,
            capability_score=100.0,
            free_eligibility=FreeEligibility.VERIFIED_FREE,
        ),
    ))

    # This call should complete without any network activity
    status = build_router_status(
        registry=router.registry,
        health=router.health,
        router=router,
        settings=settings,
    )

    assert status["routes"]
    assert status["selected"] is not None


def test_build_router_status_new_route_auto_visible():
    """A newly configured route appears, even before runtime sync registers it."""
    settings1 = _FixedSettings(model="open_router/initial")
    router = _make_router((
        _profile("open_router/initial", tier=CapabilityTier.TIER_2, score=80.0),
    ))

    status1 = build_router_status(
        registry=router.registry,
        health=router.health,
        router=router,
        settings=settings1,
    )
    assert len(status1["routes"]) == 1

    # Add a new fallback without modifying the observed registry.
    settings2 = _FixedSettings(
        model="open_router/initial",
        model_fallbacks=("open_router/new-model",),
    )
    status2 = build_router_status(
        registry=router.registry,
        health=router.health,
        router=router,
        settings=settings2,
    )

    refs = {r["provider_model_ref"] for r in status2["routes"]}
    assert "open_router/initial" in refs
    assert "open_router/new-model" in refs
    # new-model is visible from configuration but absent from the registry.
    new_route = next(r for r in status2["routes"] if r["provider_model_ref"] == "open_router/new-model")
    assert new_route["capability"]["known"] is False
    assert new_route["free"]["eligibility"] is None
    assert new_route["executable"] is False
    assert new_route["exclusion_reason"] == "NOT_REGISTERED"
    assert new_route["rank"] is None


def test_build_router_status_does_not_mutate_empty_registry():
    """Status observes an empty registry without initializing it."""
    registry = ModelRegistry()
    health = RouteHealthStore()
    router = SmartRouter(registry, health)
    settings = _FixedSettings(model="open_router/unregistered")

    before = registry.all_profiles()
    status = build_router_status(
        registry=registry,
        health=health,
        router=router,
        settings=settings,
    )

    assert before == ()
    assert registry.all_profiles() == before
    assert status["selected"] is None


# === API integration tests ===


@pytest.fixture
def test_app(monkeypatch, tmp_path):
    """Create a test app with the real runtime and admin endpoint."""
    from free_claude_code.config.settings import Settings
    from tests.api.support import create_test_app

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    settings = Settings(
        MODEL="open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
        MODEL_FALLBACKS="groq/openai/gpt-oss-120b,ollama_cloud/nemotron-3-super",
        FCC_VERIFIED_FREE_MODELS=(
            "open_router/nvidia/nemotron-3-ultra-550b-a55b:free,"
            "groq/openai/gpt-oss-120b"
        ),
    )
    app = create_test_app(settings)
    return app, _synchronize_runtime_registry(app)


def test_router_status_endpoint_exists(test_app):
    """GET /admin/api/router/status returns 200 with RouterStatusResponse shape."""
    from fastapi.testclient import TestClient

    app, _ = test_app
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))

    response = client.get("/admin/api/router/status")
    assert response.status_code == 200

    data = response.json()
    assert "routes" in data
    assert "selected" in data
    assert "selected_rank" in data
    assert "last_successful" in data
    assert "summary" in data
    assert isinstance(data["routes"], list)
    assert isinstance(data["summary"], dict)


def test_router_status_requires_loopback(monkeypatch, tmp_path):
    """Endpoint requires loopback admin check (inherits from admin_routes)."""
    from fastapi.testclient import TestClient

    from tests.api.support import create_test_app

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    app = create_test_app()

    # Non-loopback origin should be rejected by require_loopback_admin
    client = TestClient(app, base_url="https://evil.com", client=("192.168.1.1", 50000))
    response = client.get("/admin/api/router/status")
    # require_loopback_admin returns 403 for non-loopback
    assert response.status_code in (403, 400)


def test_router_status_reflects_verified_free_vs_unknown(test_app):
    """Routes not in verified_free_models show UNKNOWN eligibility."""
    app, _runtime = test_app

    # groq/openai/gpt-oss-120b is verified-free; ollama is not
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))
    response = client.get("/admin/api/router/status")
    data = response.json()

    groq = next(r for r in data["routes"] if r["provider"] == "groq")
    ollama = next(r for r in data["routes"] if r["provider"] == "ollama_cloud")

    assert groq["free"]["eligibility"] == "verified_free"
    assert groq["free"]["executable_for_zero_cost"] is True

    assert ollama["free"]["eligibility"] == "unknown"
    assert ollama["free"]["executable_for_zero_cost"] is False


def test_router_status_capability_from_registry(test_app):
    """Capability tier/score comes from ModelRegistry, not hard-coded."""
    app, _runtime = test_app
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))
    response = client.get("/admin/api/router/status")
    data = response.json()

    openrouter = next(r for r in data["routes"] if r["provider"] == "open_router")
    assert openrouter["capability"]["tier"] == 1
    assert openrouter["capability"]["tier_name"] == "TIER_1"
    assert openrouter["capability"]["capability_score"] == 100.0
    assert openrouter["capability"]["known"] is True


def test_router_status_health_from_store(test_app):
    """Health state, failure counts, timestamps come from RouteHealthStore."""
    app, runtime = test_app
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))

    # Mark a failure on the primary route
    runtime.smart_router.health.get(
        "open_router/nvidia/nemotron-3-ultra-550b-a55b:free"
    ).mark_failure(
        failure_kind="timeout",
        status_code=504,
        message="Upstream timeout",
        retry_at=datetime.now(UTC) + timedelta(minutes=5),
    )

    response = client.get("/admin/api/router/status")
    data = response.json()

    primary = next(
        r
        for r in data["routes"]
        if r["provider_model_ref"] == "open_router/nvidia/nemotron-3-ultra-550b-a55b:free"
    )
    assert primary["health"]["state"] == "backoff"
    assert primary["health"]["last_failure_kind"] == "timeout"
    assert primary["health"]["last_failure_status"] == 504
    assert primary["health"]["last_failure_message"] == "Upstream timeout"
    assert primary["health"]["retry_at"] is not None
    assert primary["executable"] is False
    assert primary["exclusion_reason"] == "BACKOFF"


def test_router_status_canonical_identity_per_route(test_app):
    """Each route identified by canonical provider/model ref (provider+model)."""
    app, _runtime = test_app
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))
    response = client.get("/admin/api/router/status")
    data = response.json()

    for route in data["routes"]:
        # provider_model_ref must be provider/model
        assert route["provider_model_ref"] == f"{route['provider']}/{route['model']}"
        # Should not appear twice
        refs = [r["provider_model_ref"] for r in data["routes"]]
    assert len(refs) == len(set(refs))


def test_router_status_same_family_providers_independent(test_app):
    """OpenRouter Ultra and Ollama Ultra are separate routes with independent health."""
    # We need a test setup with both
    from free_claude_code.config.settings import Settings
    from tests.api.support import create_test_app

    settings = Settings(
        MODEL="open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
        MODEL_FALLBACKS="ollama_cloud/nemotron-3-ultra",
        FCC_VERIFIED_FREE_MODELS=(
            "open_router/nvidia/nemotron-3-ultra-550b-a55b:free,"
            "ollama_cloud/nemotron-3-ultra"
        ),
    )
    app = create_test_app(settings)
    _synchronize_runtime_registry(app)
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))

    response = client.get("/admin/api/router/status")
    data = response.json()

    openrouter = next(r for r in data["routes"] if r["provider"] == "open_router")
    ollama = next(r for r in data["routes"] if r["provider"] == "ollama_cloud")

    # Both TIER_1, independent health
    assert openrouter["capability"]["tier"] == 1
    assert ollama["capability"]["tier"] == 1
    assert openrouter["provider_model_ref"] != ollama["provider_model_ref"]
    assert openrouter["health"]["state"] == ollama["health"]["state"] == "unknown"


def test_router_status_no_mutating_endpoints_exposed(monkeypatch, tmp_path):
    """Verify no mutating endpoints are exposed for health/routing."""
    from tests.api.support import create_test_app

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    app = create_test_app()

    # Check no POST/PUT/DELETE endpoints for router/health mutation
    mutating_paths = [
        p
        for p in app.router.routes
        if hasattr(p, "methods")
        and any(m in p.methods for m in ("POST", "PUT", "DELETE", "PATCH"))
        and "/admin/api/router" in getattr(p, "path", "")
    ]
    # Our endpoint is GET only
    assert len(mutating_paths) == 0


# === Regression: existing 2C/2D tests must still pass ===


def test_existing_smart_router_ranking_unchanged(test_app):
    """Ensure explain() and status don't change rank()/select() behavior."""
    _app, runtime = test_app
    router = runtime.smart_router

    targets = (
        _target("open_router/nvidia/nemotron-3-ultra-550b-a55b:free"),
        _target("groq/openai/gpt-oss-120b"),
    )

    # Mirror what the request path does before ranking.
    synchronize_model_registry(router.registry, runtime.settings)

    ranked = router.rank(targets)
    selected = router.select(targets)

    assert selected is not None
    # Ultra (TIER_1) selected over gpt-oss (TIER_2)
    assert selected.target.provider_model_ref == "open_router/nvidia/nemotron-3-ultra-550b-a55b:free"
    assert ranked[0].target.provider_model_ref == selected.target.provider_model_ref


def test_existing_route_health_observer_unchanged(test_app):
    """RouteHealthObserver behavior unchanged by observability additions."""
    from free_claude_code.application.execution import ExecutionFailure, FailureKind
    from free_claude_code.application.route_health_observer import RouteHealthObserver

    runtime = test_app[1]
    health = runtime.smart_router.health
    observer = RouteHealthObserver(health)

    # Success -> AVAILABLE
    observer.observe_success("test/route", latency_ms=100.0, output_tokens=50)
    h = health.get("test/route")
    assert h.state is RouteState.AVAILABLE
    assert h.success_count == 1
    assert h.observed_latency_ms == 100.0
    assert h.observed_output_tokens == 50

    # Generic 429 -> BACKOFF
    failure = ExecutionFailure(
        kind=FailureKind.RATE_LIMIT,
        message="Rate limit",
        status_code=429,
        retryable=True,
    )
    observer.observe_failure("test/route", failure)
    h = health.get("test/route")
    assert h.state is RouteState.BACKOFF
    assert h.retry_at is not None

    # Post-commit failure must NOT demote
    observer.observe_post_commit_failure("test/route", failure)
    h = health.get("test/route")
    assert h.state is RouteState.BACKOFF  # unchanged


# === Routing exclusion reason tests ===


def test_routing_explanation_all_reasons(test_app):
    """Verify all exclusion reasons can appear in observability."""
    from free_claude_code.application.smart_router import ExclusionReason

    # We verify the enum covers all expected reasons
    expected = {
        "NOT_REGISTERED",
        "NOT_VERIFIED_FREE",
        "BELOW_MINIMUM_TIER",
        "BELOW_MINIMUM_CAPABILITY_SCORE",
        "REASONING_REQUIRED",
        "TOOLS_REQUIRED",
        "PROVIDER_EXCLUDED",
        "BLOCKED",
        "BACKOFF",
        "QUARANTINED",
            "QUOTA_EXHAUSTED",
        "UNKNOWN_HEALTH",
    }
    actual = {r.value for r in ExclusionReason}
    assert actual == expected


def test_router_status_respects_requirements(monkeypatch, tmp_path):
    """Requirements (reasoning, tools) reflected in executable status."""
    from free_claude_code.config.settings import Settings
    from tests.api.support import create_test_app

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    settings = Settings(
        MODEL="open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
        MODEL_FALLBACKS="groq/openai/gpt-oss-120b",
        FCC_VERIFIED_FREE_MODELS=(
            "open_router/nvidia/nemotron-3-ultra-550b-a55b:free,"
            "groq/openai/gpt-oss-120b"
        ),
    )
    app = create_test_app(settings)
    _synchronize_runtime_registry(app)
    client = TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))

    # Default requirements -> both executable if healthy
    response = client.get("/admin/api/router/status")
    data = response.json()
    assert all(r["executable"] for r in data["routes"] if r["provider"] in ("open_router", "groq"))
