"""Stage 2D live request-path health observation tests."""

from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from free_claude_code.application.execution import ProviderExecutor
from free_claude_code.application.model_registry import (
    CapabilityTier,
    FreeEligibility,
    ModelProfile,
    ModelRegistry,
)
from free_claude_code.application.route_health import RouteHealthStore, RouteState
from free_claude_code.application.route_health_observer import (
    HealthClassification,
    RouteHealthObserver,
    classify_failure,
)
from free_claude_code.application.routing import (
    ProviderModelTarget,
    ResolvedModelRoute,
    RoutedMessagesRequest,
)
from free_claude_code.application.smart_router import SmartRouter
from free_claude_code.config.reasoning import ReasoningPreference
from free_claude_code.core.anthropic.models import Message, MessagesRequest
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.reasoning import ReasoningPolicy

FIXED = datetime(2026, 1, 1, tzinfo=UTC)


def _failure(
    kind: FailureKind, status: int, message: str, retryable: bool = True
) -> ExecutionFailure:
    return ExecutionFailure(
        kind=kind, status_code=status, message=message, retryable=retryable
    )


def _observer(store: RouteHealthStore) -> RouteHealthObserver:
    return RouteHealthObserver(store, now=lambda: FIXED)


# ---------------------------------------------------------------------------
# classify_failure unit mapping
# ---------------------------------------------------------------------------


def test_classify_generic_429_is_backoff():
    assert classify_failure(_failure(FailureKind.RATE_LIMIT, 429, "rate limit")) is (
        HealthClassification.BACKOFF
    )


def test_classify_429_with_rate_limit_wording_stays_backoff():
    # "rate limit reached, retry shortly" must never become quarantine.
    assert classify_failure(
        _failure(FailureKind.RATE_LIMIT, 429, "Provider rate limit reached. Retry.")
    ) is HealthClassification.BACKOFF


def test_classify_429_with_confirmed_quota_is_quarantined():
    assert classify_failure(
        _failure(FailureKind.RATE_LIMIT, 429, "You have exceeded your free quota.")
    ) is HealthClassification.QUARANTINED
    assert classify_failure(
        _failure(FailureKind.RATE_LIMIT, 429, "Daily limit exhausted.")
    ) is HealthClassification.QUARANTINED
    assert classify_failure(
        _failure(FailureKind.RATE_LIMIT, 429, "Out of credits.")
    ) is HealthClassification.QUARANTINED


def test_classify_401_and_403_are_blocked():
    assert classify_failure(
        _failure(FailureKind.AUTHENTICATION, 401, "invalid api key")
    ) is HealthClassification.BLOCKED
    assert classify_failure(
        _failure(FailureKind.PERMISSION, 403, "permission denied")
    ) is HealthClassification.BLOCKED
    assert classify_failure(
        _failure(FailureKind.PERMISSION, 402, "payment required")
    ) is HealthClassification.BLOCKED


def test_classify_transient_is_backoff():
    for kind, status in (
        (FailureKind.TIMEOUT, 504),
        (FailureKind.UPSTREAM, 503),
        (FailureKind.OVERLOADED, 529),
        (FailureKind.UNAVAILABLE, 502),
    ):
        assert classify_failure(_failure(kind, status, "transient")) is (
            HealthClassification.BACKOFF
        )


def test_classify_request_scoped_is_ignored():
    assert classify_failure(
        _failure(FailureKind.INVALID_REQUEST, 400, "bad request")
    ) is HealthClassification.IGNORE


def test_classify_413_capacity_rejection_is_backoff():
    """Provider payload/TPM limits must remove the route from the next ranking."""
    assert classify_failure(
        _failure(
            FailureKind.INVALID_REQUEST,
            413,
            "Provider rejected the request as too large.",
            retryable=False,
        )
    ) is HealthClassification.BACKOFF
    assert classify_failure(
        _failure(FailureKind.CONTEXT_WINDOW_EXCEEDED, 400, "context too long")
    ) is HealthClassification.IGNORE


# ---------------------------------------------------------------------------
# observer state transitions
# ---------------------------------------------------------------------------


def test_success_marks_route_available(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)

    observer.observe_success("groq/model", latency_ms=120.0, output_tokens=10)

    health = store.get("groq/model")
    assert health.state is RouteState.AVAILABLE
    assert health.is_usable() is True
    assert health.success_count == 1
    assert health.observed_latency_ms == 120.0
    assert health.observed_output_tokens == 10


def test_429_backoff(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)

    observer.observe_failure("groq/model", _failure(FailureKind.RATE_LIMIT, 429, "rate limit"))

    health = store.get("groq/model")
    assert health.state is RouteState.BACKOFF
    assert health.retry_at == FIXED + timedelta(seconds=60)
    # Expired backoff enters background-probe probation instead of user traffic.
    assert health.is_usable(FIXED) is False
    assert health.is_usable(FIXED + timedelta(seconds=61)) is False


def test_429_quota_exhaustion_quarantined(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)

    observer.observe_failure(
        "open_router/model",
        _failure(FailureKind.RATE_LIMIT, 429, "free quota exhausted"),
    )

    health = store.get("open_router/model")
    assert health.state is RouteState.QUARANTINED
    assert health.quarantine_until == FIXED + timedelta(seconds=1800)
    assert health.is_usable(FIXED) is False
    bucket = store.get_quota_bucket("openrouter_free_tier_daily")
    assert bucket.state is RouteState.QUARANTINED
    assert bucket.is_usable(FIXED + timedelta(seconds=1801)) is True


def test_quota_reset_timestamp_controls_shared_bucket(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)
    reset_at = FIXED + timedelta(hours=2)
    failure = ExecutionFailure(
        FailureKind.RATE_LIMIT,
        429,
        "free quota exhausted",
        True,
        reset_at=reset_at,
    )

    observer.observe_failure("open_router/model", failure)

    bucket = store.get_quota_bucket("openrouter_free_tier_daily")
    assert bucket.quarantine_until == reset_at
    assert bucket.is_usable(FIXED + timedelta(hours=1)) is False
    assert bucket.is_usable(reset_at + timedelta(seconds=1)) is True


def test_401_blocked(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)

    observer.observe_failure(
        "groq/model", _failure(FailureKind.AUTHENTICATION, 401, "bad key")
    )

    health = store.get("groq/model")
    assert health.state is RouteState.BLOCKED
    assert health.is_usable() is False


def test_403_billing_blocked(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)

    observer.observe_failure(
        "groq/model", _failure(FailureKind.PERMISSION, 403, "forbidden")
    )
    observer.observe_failure(
        "other/model", _failure(FailureKind.PERMISSION, 402, "billing disabled")
    )

    assert store.get("groq/model").state is RouteState.BLOCKED
    assert store.get("other/model").state is RouteState.BLOCKED


def test_transient_failure_backoff(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)

    observer.observe_failure("groq/model", _failure(FailureKind.TIMEOUT, 504, "timeout"))

    assert store.get("groq/model").state is RouteState.BACKOFF


def test_request_scoped_failure_does_not_demote(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)
    observer.observe_success("groq/model")

    observer.observe_failure(
        "groq/model", _failure(FailureKind.INVALID_REQUEST, 400, "client error")
    )

    # The route already proved AVAILABLE; a client error must not demote it.
    health = store.get("groq/model")
    assert health.state is RouteState.AVAILABLE
    assert health.is_usable() is True


def test_413_capacity_rejection_backs_off_route(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)

    observer.observe_failure(
        "groq/openai/gpt-oss-120b",
        _failure(
            FailureKind.INVALID_REQUEST,
            413,
            "Provider rejected the request as too large.",
            retryable=False,
        ),
    )

    health = store.get("groq/openai/gpt-oss-120b")
    assert health.state is RouteState.BACKOFF
    assert health.retry_at == FIXED + timedelta(seconds=60)
    assert health.is_usable(FIXED) is False


def test_success_after_backoff_recovers(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)
    observer.observe_failure("groq/model", _failure(FailureKind.RATE_LIMIT, 429, "rate limit"))

    observer.observe_success("groq/model")

    health = store.get("groq/model")
    assert health.state is RouteState.AVAILABLE
    assert health.retry_at is None
    assert health.consecutive_failures == 0


def test_post_commit_failure_does_not_demote(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)
    observer.observe_success("groq/model")

    observer.observe_post_commit_failure(
        "groq/model", _failure(FailureKind.UPSTREAM, 502, "late failure")
    )

    health = store.get("groq/model")
    assert health.state is RouteState.AVAILABLE
    assert health.is_usable() is True


def test_route_health_is_independent(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)

    observer.observe_failure(
        "open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
        _failure(FailureKind.RATE_LIMIT, 429, "free quota exhausted"),
    )

    other = store.get("ollama_cloud/nemotron-3-ultra")
    assert other.state is RouteState.UNKNOWN
    assert other.is_usable() is True


def test_new_route_automatically_tracked(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)

    observer.observe_success("brand_new_provider/future-model")

    health = store.get("brand_new_provider/future-model")
    assert health.state is RouteState.AVAILABLE


def test_observer_persists_transitions(tmp_path):
    path = tmp_path / "rh.json"
    observer = _observer(RouteHealthStore(path))

    observer.observe_failure(
        "open_router/nvidia/nemotron-3-ultra-550b-a55b:free",
        _failure(FailureKind.RATE_LIMIT, 429, "free quota exhausted"),
    )
    observer.observe_success("ollama_cloud/nemotron-3-ultra")

    restored = RouteHealthStore(path)
    restored.load()

    assert restored.get("open_router/nvidia/nemotron-3-ultra-550b-a55b:free").state is (
        RouteState.QUARANTINED
    )
    assert restored.get("ollama_cloud/nemotron-3-ultra").state is RouteState.AVAILABLE


# ---------------------------------------------------------------------------
# executor integration
# ---------------------------------------------------------------------------


def _target(provider_id: str, provider_model: str) -> ProviderModelTarget:
    return ProviderModelTarget(
        provider_id=provider_id,
        provider_model=provider_model,
        provider_model_ref=f"{provider_id}/{provider_model}",
    )


def _routed_request(*fallbacks: ProviderModelTarget) -> RoutedMessagesRequest:
    return RoutedMessagesRequest(
        request=MessagesRequest(
            model="provider-model",
            messages=[Message(role="user", content="hello")],
        ),
        resolved=ResolvedModelRoute(
            original_model="gateway-model",
            primary=_target("provider", "provider-model"),
            fallbacks=fallbacks,
            reasoning_preference=ReasoningPreference.CLIENT,
        ),
        reasoning=ReasoningPolicy.on(),
    )


class _StepsProvider:
    """Minimal streaming provider that replays a script of chunks/errors."""

    def __init__(self, steps: list[object]) -> None:
        self._steps = list(steps)
        self.calls = 0
        self.closed = 0

    async def stream_messages(
        self,
        request: MessagesRequest,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
        response_model: str | None = None,
        reasoning: ReasoningPolicy = ReasoningPolicy.on(),
        request_headers: Mapping[str, str] | None = None,
        model_info: object | None = None,
    ) -> AsyncIterator[str]:
        self.calls += 1
        try:
            for step in self._steps:
                if isinstance(step, BaseException):
                    raise step
                yield step
        finally:
            self.closed += 1


def _executor(providers: dict[str, object], observer) -> ProviderExecutor:
    return ProviderExecutor(
        AsyncMock(side_effect=lambda name: providers[name]),
        token_counter=lambda _m, _s, _t: 17,
        progress_timeout_seconds=60.0,
        route_health_observer=observer,
    )


@pytest.mark.asyncio
async def test_first_non_empty_chunk_marks_available(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)
    provider = _StepsProvider(["event: content\ndata: x\n\n"])
    executor = _executor({"provider": provider}, observer)

    output = [
        chunk
        async for chunk in executor.stream_messages(
            _routed_request(), raw_log_payload={}, request_id="req1"
        )
    ]

    assert output == ["event: content\ndata: x\n\n"]
    health = store.get("provider/provider-model")
    assert health.state is RouteState.AVAILABLE
    assert health.observed_latency_ms is not None
    assert health.observed_latency_ms >= 0.0


@pytest.mark.asyncio
async def test_empty_only_stream_does_not_mark_available(tmp_path):
    # Heartbeat / empty chunks are not meaningful content and must not count.
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)
    provider = _StepsProvider(["", "", ""])
    executor = _executor({"provider": provider}, observer)

    output = [
        chunk
        async for chunk in executor.stream_messages(
            _routed_request(), raw_log_payload={}, request_id="req2"
        )
    ]

    assert output == []
    assert store.get("provider/provider-model").state is RouteState.UNKNOWN


@pytest.mark.asyncio
async def test_post_first_chunk_failure_does_not_retry_another_provider(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)
    primary = _StepsProvider(["event: content\ndata: x\n\n", _failure(FailureKind.UPSTREAM, 502, "late")])
    fallback = _StepsProvider(["event: fallback\ndata: y\n\n"])
    executor = _executor({"provider": primary, "fallback": fallback}, observer)

    with pytest.raises(ExecutionFailure):
        output = [
            chunk
            async for chunk in executor.stream_messages(
                _routed_request(_target("fallback", "fallback-model")),
                raw_log_payload={},
                request_id="req3",
            )
        ]
        # The committed chunk was delivered before the later failure.
        assert output == ["event: content\ndata: x\n\n"]

    # The active stream failed AFTER delivering content: route stays AVAILABLE
    # and no second provider is opened.
    assert store.get("provider/provider-model").state is RouteState.AVAILABLE
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_pre_commit_429_marks_backoff_and_falls_back(tmp_path):
    store = RouteHealthStore(tmp_path / "rh.json")
    observer = _observer(store)
    primary = _StepsProvider([_failure(FailureKind.RATE_LIMIT, 429, "rate limit")])
    fallback = _StepsProvider(["event: fallback\ndata: y\n\n"])
    executor = _executor({"provider": primary, "fallback": fallback}, observer)

    output = [
        chunk
        async for chunk in executor.stream_messages(
            _routed_request(_target("fallback", "fallback-model")),
            raw_log_payload={},
            request_id="req4",
        )
    ]

    assert output == ["event: fallback\ndata: y\n\n"]
    assert store.get("provider/provider-model").state is RouteState.BACKOFF
    assert fallback.calls == 1


# ---------------------------------------------------------------------------
# smart router integration lifecycle
# ---------------------------------------------------------------------------

_ULTRA_A = "open_router/ultra-a"
_ULTRA_B = "ollama_cloud/ultra-b"
_SUPER_A = "groq/super-a"


def _free_profile(route_ref: str, tier: CapabilityTier, score: float) -> ModelProfile:
    provider, model = route_ref.split("/", 1)
    return ModelProfile(
        provider_id=provider,
        model_id=model,
        capability_tier=tier,
        capability_score=score,
        free_eligibility=FreeEligibility.VERIFIED_FREE,
    )


def _targets() -> tuple[ProviderModelTarget, ...]:
    return (
        _target("open_router", "ultra-a"),
        _target("ollama_cloud", "ultra-b"),
        _target("groq", "super-a"),
    )


def _select(router: SmartRouter, targets: tuple[ProviderModelTarget, ...]):
    ranked = router.select(targets)
    return ranked.target.provider_model_ref if ranked is not None else None


def test_smart_router_reflects_observed_health_lifecycle():
    registry = ModelRegistry(
        (
            _free_profile(_ULTRA_A, CapabilityTier.TIER_1, 100.0),
            _free_profile(_ULTRA_B, CapabilityTier.TIER_1, 100.0),
            _free_profile(_SUPER_A, CapabilityTier.TIER_2, 90.0),
        )
    )
    store = RouteHealthStore()
    observer = RouteHealthObserver(store)
    router = SmartRouter(registry, store)
    targets = _targets()

    # All verified-free and healthy: capability-first picks Ultra A.
    assert _select(router, targets) == _ULTRA_A

    # Ultra A confirms quota exhaustion -> QUARANTINED; excluded next request.
    observer.observe_failure(
        _ULTRA_A, _failure(FailureKind.RATE_LIMIT, 429, "free quota exhausted")
    )
    assert store.get(_ULTRA_A).state is RouteState.QUARANTINED
    assert _select(router, targets) == _ULTRA_B

    # Ultra B hits a temporary 429 -> BACKOFF; both Ultra A and B excluded.
    observer.observe_failure(
        _ULTRA_B, _failure(FailureKind.RATE_LIMIT, 429, "rate limit")
    )
    assert store.get(_ULTRA_B).state is RouteState.BACKOFF
    assert _select(router, targets) == _SUPER_A

    # Ultra B succeeds and recovers; Ultra A remains quarantined.
    observer.observe_success(_ULTRA_B)
    assert store.get(_ULTRA_B).state is RouteState.AVAILABLE
    assert _select(router, targets) == _ULTRA_B
