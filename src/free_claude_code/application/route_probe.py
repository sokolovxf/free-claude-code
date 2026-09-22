"""Bounded background canaries for routes recovering from transient failures."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from time import monotonic

from loguru import logger

from free_claude_code.application.execution import _is_meaningful_chunk
from free_claude_code.application.model_registry import quota_bucket_for_route
from free_claude_code.application.route_health import RouteHealthStore, RouteState
from free_claude_code.application.route_health_observer import RouteHealthObserver
from free_claude_code.application.routing import ProviderModelTarget
from free_claude_code.config.model_refs import configured_chat_model_refs
from free_claude_code.core.anthropic import Message, MessagesRequest, ThinkingConfig
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.reasoning import ReasoningPolicy
from free_claude_code.core.trace import close_stream_input


class RouteProbeService:
    """Rehabilitate failed routes with bounded, low-volume background work.

    Routes in transient backoff are rehabilitated, and never-tested routes get
    a bounded startup canary. A route must pass several small streamed canaries
    before it becomes available again. Quarantined and blocked routes remain
    untouched.
    """

    def __init__(
        self,
        provider_manager: object,
        health: RouteHealthStore,
        observer: RouteHealthObserver,
        *,
        initial_delay_seconds: float = 15.0,
        interval_seconds: float = 30.0,
        probe_timeout_seconds: float = 12.0,
        required_successes: int = 3,
        max_probes_per_cycle: int = 2,
    ) -> None:
        if initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds must be >= 0")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0")
        if probe_timeout_seconds <= 0:
            raise ValueError("probe_timeout_seconds must be > 0")
        if required_successes <= 0:
            raise ValueError("required_successes must be > 0")
        if max_probes_per_cycle <= 0:
            raise ValueError("max_probes_per_cycle must be > 0")
        self._provider_manager = provider_manager
        self._health = health
        self._observer = observer
        self._initial_delay = float(initial_delay_seconds)
        self._interval = float(interval_seconds)
        self._timeout = float(probe_timeout_seconds)
        self._required_successes = required_successes
        self._max_probes_per_cycle = max_probes_per_cycle

    async def run(self) -> None:
        """Run until cancelled by the application lifecycle."""
        await asyncio.sleep(self._initial_delay)
        await self._provider_manager.wait_for_catalog()
        while True:
            await self._run_cycle()
            await asyncio.sleep(self._interval)

    async def _run_cycle(self) -> None:
        settings = self._provider_manager.current_settings()
        now = datetime.now(UTC)
        targets = self._due_targets(settings, now)
        for target in targets[: self._max_probes_per_cycle]:
            try:
                await self._probe(target, settings)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "FCC background route probe failed unexpectedly: {}",
                    target.provider_model_ref,
                )

    def _due_targets(self, settings: object, now: datetime) -> tuple[ProviderModelTarget, ...]:
        targets: list[ProviderModelTarget] = []
        for ref in configured_chat_model_refs(settings):  # type: ignore[arg-type]
            health = self._health.get(ref.model_ref)
            recovering = health.state is RouteState.BACKOFF and health.probe_required
            startup_unknown = (
                health.state is RouteState.UNKNOWN
                and health.last_probe_at is None
                and health.success_count == 0
                and health.failure_count == 0
            )
            if not recovering and not startup_unknown:
                continue
            if health.retry_at is not None and now < health.retry_at:
                continue
            bucket = quota_bucket_for_route(ref.model_ref)
            if bucket is not None and not self._health.get_quota_bucket(bucket).is_usable(now):
                continue
            targets.append(
                ProviderModelTarget(
                    provider_id=ref.provider_id,
                    provider_model=ref.model_id,
                    provider_model_ref=ref.model_ref,
                )
            )
        return tuple(targets)

    async def _probe(self, target: ProviderModelTarget, settings: object) -> None:
        request_id = f"probe_{uuid.uuid4().hex}"
        started = monotonic()
        route_model = target.provider_model
        request = MessagesRequest(
            model=route_model,
            messages=[Message(role="user", content="ping")],
            max_tokens=4,
            stream=True,
            thinking=ThinkingConfig(enabled=False),
        )
        provider_stream = None
        preserved_error: BaseException | None = None
        try:
            async with await self._provider_manager.acquire() as lease:
                provider = await lease.resolve_provider(target.provider_id)
                provider_stream = provider.stream_messages(
                    request,
                    input_tokens=1,
                    request_id=request_id,
                    response_model=route_model,
                    reasoning=ReasoningPolicy.off(),
                    model_info=lease.model_info(target.provider_id, route_model),
                )
                async with asyncio.timeout(self._timeout):
                    while True:
                        chunk = await anext(provider_stream)
                        if _is_meaningful_chunk(chunk):
                            break
                    latency_ms = max(0.0, (monotonic() - started) * 1000.0)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            preserved_error = error
            failure = self._failure(error, target)
            self._observer.observe_probe_failure(target.provider_model_ref, failure)
            return
        finally:
            if provider_stream is not None:
                await close_stream_input(
                    provider_stream,
                    owner="route_probe",
                    source="background",
                    preserved_error=preserved_error,
                )

        promoted = self._observer.observe_probe_success(
            target.provider_model_ref,
            latency_ms=latency_ms,
            required_successes=self._required_successes,
            next_probe_at=datetime.now(UTC) + timedelta(seconds=self._interval),
        )
        logger.info(
            "FCC PROBE route={} result={} latency_ms={:.0f}",
            target.provider_model_ref,
            "available" if promoted else "warming",
            latency_ms,
        )

    @staticmethod
    def _failure(
        error: Exception,
        target: ProviderModelTarget,
    ) -> ExecutionFailure:
        if isinstance(error, ExecutionFailure):
            return error
        status = _status_code(error)
        if isinstance(error, TimeoutError):
            return ExecutionFailure(
                kind=FailureKind.TIMEOUT,
                status_code=504,
                message=f"Background probe timed out for {target.provider_model_ref}.",
                retryable=True,
            )
        if status == 429:
            return ExecutionFailure(
                kind=FailureKind.RATE_LIMIT,
                status_code=429,
                message=f"Background probe was rate limited: {error}",
                retryable=True,
            )
        if status in (401, 402, 403):
            return ExecutionFailure(
                kind=(
                    FailureKind.AUTHENTICATION
                    if status == 401
                    else FailureKind.PERMISSION
                ),
                status_code=status,
                message=(
                    "Background probe was rejected by provider credentials."
                    if status != 402
                    else "Background probe was rejected because provider billing or credits are required."
                ),
                retryable=False,
            )
        effective_status = status if status is not None else 502
        return ExecutionFailure(
            kind=(
                FailureKind.INVALID_REQUEST
                if 400 <= effective_status < 500
                else FailureKind.UPSTREAM
            ),
            status_code=effective_status,
            message=f"Background probe failed: {error}",
            retryable=effective_status >= 500,
        )


def _status_code(error: BaseException) -> int | None:
    """Extract a provider status without importing provider SDK policy here."""
    for candidate in (
        getattr(error, "status_code", None),
        getattr(error, "status", None),
        getattr(getattr(error, "response", None), "status_code", None),
        getattr(error, "code", None),
    ):
        if isinstance(candidate, int):
            return candidate
    return None
