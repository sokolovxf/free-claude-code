from datetime import UTC, datetime, timedelta

from free_claude_code.application.route_health import (
    RouteHealth,
    RouteHealthStore,
    RouteState,
)


def test_new_route_is_unknown():
    health = RouteHealth(route_ref="groq/model")

    assert health.state is RouteState.UNKNOWN
    assert health.is_usable() is True


def test_success_restores_route_to_available():
    health = RouteHealth(
        route_ref="groq/model",
        state=RouteState.BACKOFF,
        retry_at=datetime.now(UTC) + timedelta(minutes=10),
        consecutive_failures=3,
    )

    health.mark_success(latency_ms=120.0, output_tokens=42)

    assert health.state is RouteState.AVAILABLE
    assert health.retry_at is None
    assert health.consecutive_failures == 0
    assert health.success_count == 1
    assert health.observed_output_tokens == 42
    assert health.observed_latency_ms == 120.0


def test_temporary_failure_enters_backoff_without_quarantine():
    retry_at = datetime.now(UTC) + timedelta(seconds=30)
    health = RouteHealth(route_ref="groq/model")

    health.mark_failure(
        failure_kind="rate_limit",
        status_code=429,
        retry_at=retry_at,
    )

    assert health.state is RouteState.BACKOFF
    assert health.is_usable() is False
    assert health.quarantine_until is None


def test_confirmed_quota_exhaustion_is_quarantined():
    until = datetime.now(UTC) + timedelta(hours=1)
    health = RouteHealth(route_ref="groq/model")

    health.mark_quarantined(until=until)

    assert health.state is RouteState.QUARANTINED
    assert health.is_usable() is False


def test_blocked_route_is_not_usable():
    health = RouteHealth(route_ref="paid/model")

    health.mark_blocked("paid route")

    assert health.state is RouteState.BLOCKED
    assert health.is_usable() is False


def test_health_store_round_trip(tmp_path):
    path = tmp_path / "route-health.json"

    store = RouteHealthStore(path)
    health = store.get("groq/model")
    health.mark_success(latency_ms=80.0, output_tokens=10)
    store.save()

    restored = RouteHealthStore(path)
    restored.load()

    loaded = restored.get("groq/model")

    assert loaded.state is RouteState.AVAILABLE
    assert loaded.success_count == 1
    assert loaded.observed_output_tokens == 10
    assert loaded.observed_latency_ms == 80.0


def test_corrupt_health_file_is_ignored(tmp_path):
    path = tmp_path / "route-health.json"
    path.write_text("{not-json", encoding="utf-8")

    store = RouteHealthStore(path)
    store.load()

    assert store.all() == ()
