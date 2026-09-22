"""Translate provider execution outcomes into RouteHealthStore updates.

Stage 2D wires live request-path observations back into the shared
:class:`~free_claude_code.application.route_health.RouteHealthStore` that
:class:`~free_claude_code.application.smart_router.SmartRouter` reads. It is a
thin, protocol-neutral adapter: execution semantics (an
:class:`~free_claude_code.core.failures.ExecutionFailure`) are reduced to a
health classification, which is then applied with the store's existing
``mark_*`` APIs. Health never affects the active request; it only shapes the
next routing decision.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from loguru import logger

from free_claude_code.core.failures import ExecutionFailure, FailureKind

from .model_registry import quota_bucket_for_route
from .route_health import RouteHealthStore, RouteState


class HealthClassification(StrEnum):
    """Health outcome categories reduced from one execution failure."""

    IGNORE = "ignore"  # request-scoped; not evidence of provider availability
    BACKOFF = "backoff"  # temporary, retryable
    QUARANTINED = "quarantined"  # confirmed quota / free-allocation exhaustion
    BLOCKED = "blocked"  # authentication / billing / permission


# Strong, explicit quota/credit-exhaustion signals. These are deliberately
# narrow: the most dangerous misclassification is calling a *temporary* rate
# limit a *permanent* quota block, so only unmistakable quota language upgrades
# a 429 from BACKOFF to QUARANTINED.
_QUOTA_EXHAUSTION_MARKERS = frozenset(
    {
        "quota",  # "quota exceeded", "quota exhausted", "free quota"
        "daily limit",
        "monthly limit",
        "daily usage limit",
        "monthly usage limit",
        "usage limit",
        "free-models-per-day",
        "free models per day",
        "credits exhausted",
        "insufficient credits",
        "insufficient_quota",
        "insufficient user quota",
        "out of credits",
        "allocation exhausted",
    }
)


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _has_quota_exhaustion_marker(message: str) -> bool:
    text = message.casefold()
    return any(marker in text for marker in _QUOTA_EXHAUSTION_MARKERS)


def classify_failure(failure: ExecutionFailure) -> HealthClassification:
    """Reduce one canonical execution failure to a health classification.

    - A generic 429 (``RATE_LIMIT``) is temporary -> BACKOFF; only a strong
      explicit quota/credit signal upgrades it to QUARANTINED.
    - Authentication / billing / permission -> BLOCKED (never transient).
    - Overload / timeout / upstream / unavailable -> BACKOFF.
    - Ordinary request-scoped failures (invalid request, context window
      exceeded) do not reflect provider availability and are IGNORED so a
      healthy route is not demoted for a client-side error.
    - A 413 request-size rejection is different: it is a route capacity signal
      (for example, a provider TPM or payload limit), so that route enters
      temporary BACKOFF and SmartRouter does not select it for every turn.
    """
    kind = failure.kind

    if kind is FailureKind.RATE_LIMIT:
        if _has_quota_exhaustion_marker(failure.message):
            return HealthClassification.QUARANTINED
        return HealthClassification.BACKOFF

    if kind in (FailureKind.AUTHENTICATION, FailureKind.PERMISSION):
        return HealthClassification.BLOCKED

    if kind in (
        FailureKind.OVERLOADED,
        FailureKind.TIMEOUT,
        FailureKind.UPSTREAM,
        FailureKind.UNAVAILABLE,
    ):
        return HealthClassification.BACKOFF

    if failure.status_code == 413 and kind in (
        FailureKind.INVALID_REQUEST,
        FailureKind.CONTEXT_WINDOW_EXCEEDED,
    ):
        return HealthClassification.BACKOFF

    # FailureKind.INVALID_REQUEST and FailureKind.CONTEXT_WINDOW_EXCEEDED.
    return HealthClassification.IGNORE


class RouteHealthObserver:
    """Translate execution outcomes into updates on a shared health store.

    Wraps the process-lifetime :class:`RouteHealthStore` so observations flow
    into exactly the store SmartRouter reads. In-memory state is updated
    immediately (no I/O, so an active stream is never blocked); a meaningful
    transition is persisted once with an atomic write.
    """

    def __init__(
        self,
        store: RouteHealthStore,
        *,
        backoff_seconds: float = 60.0,
        quarantine_seconds: float = 1800.0,
        now: Callable[[], datetime] = _utc_now,
    ) -> None:
        self._store = store
        self._backoff_seconds = float(backoff_seconds)
        self._quarantine_seconds = float(quarantine_seconds)
        self._now = now

    def observe_success(
        self,
        route_ref: str,
        *,
        latency_ms: float | None = None,
        output_tokens: int = 0,
    ) -> None:
        """Record that a route served meaningful content and is AVAILABLE.

        For streaming this is triggered by the first non-empty chunk; the route
        is proven functional without waiting for the stream to finish.
        """
        self._store.get(route_ref).mark_success(
            latency_ms=latency_ms,
            output_tokens=output_tokens,
        )
        self._finish(route_ref, "success", RouteState.AVAILABLE)

    def observe_failure(self, route_ref: str, failure: ExecutionFailure) -> None:
        """Classify and record a pre-commit execution failure on the route."""
        classification = classify_failure(failure)
        if classification is HealthClassification.IGNORE:
            logger.info(
                "FCC HEALTH route={} event=ignore failure_kind={} (request-scoped)",
                route_ref,
                failure.kind.value,
            )
            return
        self._apply(route_ref, classification, failure)

    def observe_probe_failure(self, route_ref: str, failure: ExecutionFailure) -> None:
        """Record a canary failure, including probe-specific request errors."""
        classification = classify_failure(failure)
        if classification is HealthClassification.IGNORE:
            now = self._now()
            health = self._store.get(route_ref)
            health.mark_failure(
                failure_kind="probe_failure",
                status_code=failure.status_code,
                message=failure.message or failure.kind.value,
                retry_at=now + timedelta(seconds=self._backoff_seconds),
                requires_probe=True,
            )
            self._finish(route_ref, "probe_failure", RouteState.BACKOFF)
            return
        self._apply(route_ref, classification, failure)

    def quota_bucket_for_route(self, route_ref: str) -> str | None:
        """Return the shared quota bucket associated with a route, if any."""
        return quota_bucket_for_route(route_ref)

    def is_route_usable(self, route_ref: str) -> bool:
        """Return whether route and shared quota health allow another attempt."""
        route = self._store.get(route_ref)
        if not route.is_usable(self._now()):
            return False
        bucket = quota_bucket_for_route(route_ref)
        return bucket is None or self._store.get_quota_bucket(bucket).is_usable(
            self._now()
        )

    def observe_post_commit_failure(
        self, route_ref: str, failure: ExecutionFailure
    ) -> None:
        """Record a stream failure after content was already delivered.

        The route already demonstrated successful service, so it stays
        AVAILABLE and is never demoted. Diagnostics only; the active stream is
        left for the caller to finalize.
        """
        logger.info(
            "FCC HEALTH route={} event=post_commit_failure state=AVAILABLE "
            "failure_kind={} status_code={}",
            route_ref,
            failure.kind.value,
            failure.status_code,
        )

    def observe_probe_success(
        self,
        route_ref: str,
        *,
        latency_ms: float | None,
        required_successes: int,
        next_probe_at: datetime,
    ) -> bool:
        """Record a background canary and promote after repeated passes."""
        promoted = self._store.get(route_ref).mark_probe_success(
            latency_ms=latency_ms,
            required_successes=required_successes,
            next_probe_at=next_probe_at,
        )
        self._finish(
            route_ref,
            "probe_promoted" if promoted else "probe_success",
            RouteState.AVAILABLE if promoted else RouteState.BACKOFF,
        )
        return promoted

    def _apply(
        self,
        route_ref: str,
        classification: HealthClassification,
        failure: ExecutionFailure,
    ) -> None:
        health = self._store.get(route_ref)
        now = self._now()
        message = failure.message or failure.kind.value

        if classification is HealthClassification.BACKOFF:
            health.mark_failure(
                failure_kind=failure.kind.value,
                status_code=failure.status_code,
                message=message,
                retry_at=now + timedelta(seconds=self._backoff_seconds),
                requires_probe=True,
            )
            self._finish(route_ref, "transient_failure", RouteState.BACKOFF)
        elif classification is HealthClassification.QUARANTINED:
            bucket = quota_bucket_for_route(route_ref)
            health.mark_quarantined(
                until=failure.reset_at
                or now + timedelta(seconds=self._quarantine_seconds),
                reason=message,
            )
            if bucket is not None:
                self._store.get_quota_bucket(bucket).mark_quarantined(
                    until=failure.reset_at
                    or now + timedelta(seconds=self._quarantine_seconds),
                    reason=message,
                )
            self._finish(route_ref, "quota_exhausted", RouteState.QUARANTINED)
        elif classification is HealthClassification.BLOCKED:
            health.mark_blocked(reason=message)
            self._finish(route_ref, "authorization_failure", RouteState.BLOCKED)

    def _finish(self, route_ref: str, event: str, state: RouteState) -> None:
        logger.info(
            "FCC HEALTH route={} event={} state={}",
            route_ref,
            event,
            state.value,
        )
        self._persist()

    def _persist(self) -> None:
        try:
            self._store.save()
        except Exception:  # never let a persistence failure raise into a request
            logger.warning("FCC HEALTH failed to persist route health", exc_info=True)
