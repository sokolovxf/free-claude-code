from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from free_claude_code.application.route_health import RouteHealthStore, RouteState
from free_claude_code.application.route_health_observer import RouteHealthObserver
from free_claude_code.application.route_probe import RouteProbeService
from free_claude_code.application.routing import ProviderModelTarget
from free_claude_code.config.settings import Settings


class ProbeProvider:
    def __init__(self, chunk: str = "event: content_block_delta\ndata: {}\n\n"):
        self.chunk = chunk
        self.calls = 0

    def stream_messages(self, *args, **kwargs) -> AsyncIterator[str]:
        del args, kwargs
        self.calls += 1

        async def stream() -> AsyncIterator[str]:
            yield self.chunk

        return stream()


class ProbeLease:
    def __init__(self, provider: ProbeProvider):
        self.provider = provider

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def resolve_provider(self, _provider_id: str):
        return self.provider

    def model_info(self, _provider_id: str, _model_id: str):
        return None


class ProbeManager:
    def __init__(self, provider: ProbeProvider):
        self.provider = provider
        self.settings = Settings(model="groq/model")

    def current_settings(self):
        return self.settings

    async def wait_for_catalog(self):
        return None

    async def acquire(self):
        return ProbeLease(self.provider)


def _target() -> ProviderModelTarget:
    return ProviderModelTarget(
        provider_id="groq",
        provider_model="model",
        provider_model_ref="groq/model",
    )


@pytest.mark.asyncio
async def test_probe_promotes_route_after_three_valid_streams() -> None:
    provider = ProbeProvider()
    manager = ProbeManager(provider)
    health = RouteHealthStore()
    health.get("groq/model").mark_failure(
        failure_kind="upstream",
        retry_at=datetime.now(UTC) - timedelta(seconds=1),
        requires_probe=True,
    )
    observer = RouteHealthObserver(health)
    service = RouteProbeService(
        manager,
        health,
        observer,
        required_successes=3,
        interval_seconds=0.01,
    )

    target = _target()
    await service._probe(target, manager.settings)
    health.get("groq/model").retry_at = datetime.now(UTC) - timedelta(seconds=1)
    await service._probe(target, manager.settings)
    health.get("groq/model").retry_at = datetime.now(UTC) - timedelta(seconds=1)
    await service._probe(target, manager.settings)

    route = health.get("groq/model")
    assert provider.calls == 3
    assert route.state is RouteState.AVAILABLE
    assert route.probe_required is False


@pytest.mark.asyncio
async def test_probe_only_selects_expired_probe_required_backoff() -> None:
    manager = ProbeManager(ProbeProvider())
    health = RouteHealthStore()
    observer = RouteHealthObserver(health)
    service = RouteProbeService(manager, health, observer)
    now = datetime.now(UTC)
    health.get("groq/model").mark_failure(
        failure_kind="upstream",
        retry_at=now + timedelta(seconds=30),
        requires_probe=True,
    )

    assert service._due_targets(manager.settings, now) == ()


def test_probe_selects_never_tested_unknown_route() -> None:
    manager = ProbeManager(ProbeProvider())
    health = RouteHealthStore()
    observer = RouteHealthObserver(health)
    service = RouteProbeService(manager, health, observer)

    assert service._due_targets(manager.settings, datetime.now(UTC)) == (_target(),)


def test_probe_429_preserves_quota_marker_for_quarantine() -> None:
    manager = ProbeManager(ProbeProvider())
    health = RouteHealthStore()
    observer = RouteHealthObserver(health)
    service = RouteProbeService(manager, health, observer)

    class QuotaError(Exception):
        status_code = 429

    failure = service._failure(
        QuotaError("monthly usage limit reached"),
        _target(),
    )
    observer.observe_probe_failure("groq/model", failure)

    assert health.get("groq/model").state is RouteState.QUARANTINED
