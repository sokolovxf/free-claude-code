"""Persistent dynamic route health for Smart Router v2."""

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path


class RouteState(StrEnum):
    """Current operational state of a provider/model route."""

    AVAILABLE = "available"
    BACKOFF = "backoff"
    QUARANTINED = "quarantined"
    BLOCKED = "blocked"
    UNKNOWN = "unknown"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)

    return parsed


@dataclass(slots=True)
class RouteHealth:
    """Observed operational state for one provider/model route."""

    route_ref: str
    state: RouteState = RouteState.UNKNOWN
    last_success_at: datetime | None = None
    last_failure_at: datetime | None = None
    retry_at: datetime | None = None
    quarantine_until: datetime | None = None
    last_failure_kind: str | None = None
    last_failure_status: int | None = None
    last_failure_message: str | None = None
    consecutive_failures: int = 0
    success_count: int = 0
    failure_count: int = 0
    observed_input_tokens: int = 0
    observed_output_tokens: int = 0
    observed_latency_ms: float | None = None
    probe_required: bool = False
    probe_successes: int = 0
    last_probe_at: datetime | None = None
    updated_at: datetime = field(default_factory=_utc_now)

    def is_usable(self, now: datetime | None = None) -> bool:
        """Return whether this route is currently usable."""
        now = now or _utc_now()

        if self.state is RouteState.BLOCKED:
            return False

        if self.state is RouteState.QUARANTINED and (
            self.quarantine_until is None or now < self.quarantine_until
        ):
            return False

        if (
            self.state is RouteState.BACKOFF
            and self.retry_at is not None
            and now < self.retry_at
        ):
            return False

        if self.state is RouteState.BACKOFF and self.probe_required:
            return False

        return self.state in {
            RouteState.AVAILABLE,
            RouteState.UNKNOWN,
            RouteState.BACKOFF,
            RouteState.QUARANTINED,
        }

    def mark_success(
        self,
        *,
        latency_ms: float | None = None,
        output_tokens: int = 0,
    ) -> None:
        """Record a successful route execution."""
        now = _utc_now()

        self.state = RouteState.AVAILABLE
        self.last_success_at = now
        self.retry_at = None
        self.quarantine_until = None
        self.last_failure_kind = None
        self.last_failure_status = None
        self.last_failure_message = None
        self.consecutive_failures = 0
        self.probe_required = False
        self.probe_successes = 0
        self.success_count += 1
        self.observed_output_tokens += max(output_tokens, 0)
        self.observed_latency_ms = latency_ms
        self.updated_at = now

    def mark_failure(
        self,
        *,
        failure_kind: str,
        status_code: int | None = None,
        message: str | None = None,
        retry_at: datetime | None = None,
        quarantine_until: datetime | None = None,
        block: bool = False,
        requires_probe: bool = False,
    ) -> None:
        """Record a route failure without assuming that it is quota exhaustion."""
        now = _utc_now()

        if block:
            state = RouteState.BLOCKED
        elif quarantine_until is not None:
            state = RouteState.QUARANTINED
        elif retry_at is not None:
            state = RouteState.BACKOFF
        else:
            state = RouteState.UNKNOWN

        self.state = state
        self.last_failure_at = now
        self.retry_at = retry_at
        self.quarantine_until = quarantine_until
        self.last_failure_kind = failure_kind
        self.last_failure_status = status_code
        self.last_failure_message = message
        self.probe_required = requires_probe
        self.probe_successes = 0
        self.consecutive_failures += 1
        self.failure_count += 1
        self.updated_at = now

    def mark_quarantined(
        self,
        *,
        until: datetime | None = None,
        reason: str = "confirmed quota exhaustion",
    ) -> None:
        """Mark a route unavailable because quota exhaustion was confirmed."""
        now = _utc_now()

        self.state = RouteState.QUARANTINED
        self.quarantine_until = until
        self.retry_at = None
        self.last_failure_at = now
        self.last_failure_kind = "quota_exhausted"
        self.last_failure_message = reason
        self.probe_required = False
        self.probe_successes = 0
        self.consecutive_failures += 1
        self.failure_count += 1
        self.updated_at = now

    def mark_blocked(self, reason: str) -> None:
        """Permanently block a route from automatic execution."""
        now = _utc_now()

        self.state = RouteState.BLOCKED
        self.retry_at = None
        self.quarantine_until = None
        self.last_failure_at = now
        self.last_failure_kind = "blocked"
        self.last_failure_message = reason
        self.probe_required = False
        self.probe_successes = 0
        self.updated_at = now

    def mark_probe_success(
        self,
        *,
        latency_ms: float | None,
        required_successes: int,
        next_probe_at: datetime,
    ) -> bool:
        """Record a valid canary and promote only after repeated successes."""
        if required_successes <= 0:
            raise ValueError("required_successes must be > 0")
        now = _utc_now()
        self.last_probe_at = now
        self.probe_successes += 1
        if self.probe_successes >= required_successes:
            self.mark_success(latency_ms=latency_ms)
            return True
        self.state = RouteState.BACKOFF
        self.retry_at = next_probe_at
        self.quarantine_until = None
        self.probe_required = True
        self.updated_at = now
        return False

    def to_dict(self) -> dict[str, object]:
        """Serialize health state."""
        return {
            "route_ref": self.route_ref,
            "state": self.state.value,
            "last_success_at": (
                self.last_success_at.isoformat() if self.last_success_at else None
            ),
            "last_failure_at": (
                self.last_failure_at.isoformat() if self.last_failure_at else None
            ),
            "retry_at": self.retry_at.isoformat() if self.retry_at else None,
            "quarantine_until": (
                self.quarantine_until.isoformat() if self.quarantine_until else None
            ),
            "last_failure_kind": self.last_failure_kind,
            "last_failure_status": self.last_failure_status,
            "last_failure_message": self.last_failure_message,
            "consecutive_failures": self.consecutive_failures,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "observed_input_tokens": self.observed_input_tokens,
            "observed_output_tokens": self.observed_output_tokens,
            "observed_latency_ms": self.observed_latency_ms,
            "probe_required": self.probe_required,
            "probe_successes": self.probe_successes,
            "last_probe_at": (
                self.last_probe_at.isoformat() if self.last_probe_at else None
            ),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> RouteHealth:
        """Deserialize health state defensively."""
        state = RouteState(str(data.get("state", RouteState.UNKNOWN)))
        stored_probe_required = data.get("probe_required")
        return cls(
            route_ref=str(data["route_ref"]),
            state=state,
            last_success_at=_parse_datetime(
                data.get("last_success_at")  # type: ignore[arg-type]
            ),
            last_failure_at=_parse_datetime(
                data.get("last_failure_at")  # type: ignore[arg-type]
            ),
            retry_at=_parse_datetime(data.get("retry_at")),  # type: ignore[arg-type]
            quarantine_until=_parse_datetime(
                data.get("quarantine_until")  # type: ignore[arg-type]
            ),
            last_failure_kind=(
                str(data["last_failure_kind"])
                if data.get("last_failure_kind") is not None
                else None
            ),
            last_failure_status=(
                int(data["last_failure_status"])
                if data.get("last_failure_status") is not None
                else None
            ),
            last_failure_message=(
                str(data["last_failure_message"])
                if data.get("last_failure_message") is not None
                else None
            ),
            consecutive_failures=int(data.get("consecutive_failures", 0)),
            success_count=int(data.get("success_count", 0)),
            failure_count=int(data.get("failure_count", 0)),
            observed_input_tokens=int(data.get("observed_input_tokens", 0)),
            observed_output_tokens=int(data.get("observed_output_tokens", 0)),
            observed_latency_ms=(
                float(data["observed_latency_ms"])
                if data.get("observed_latency_ms") is not None
                else None
            ),
            # Older health files predate background rehabilitation. Treat old
            # transient-backoff records as probe-required on upgrade.
            probe_required=(
                bool(stored_probe_required)
                if stored_probe_required is not None
                else state is RouteState.BACKOFF
            ),
            probe_successes=int(data.get("probe_successes", 0)),
            last_probe_at=_parse_datetime(
                data.get("last_probe_at")  # type: ignore[arg-type]
            ),
            updated_at=(
                _parse_datetime(data.get("updated_at"))  # type: ignore[arg-type]
                or _utc_now()
            ),
        )


class RouteHealthStore:
    """Versioned persistent collection of route health observations."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (Path.home() / ".fcc" / "route-health.json")
        self._routes: dict[str, RouteHealth] = {}
        self._quota_buckets: dict[str, RouteHealth] = {}

    def get(self, route_ref: str) -> RouteHealth:
        """Return existing health or create an UNKNOWN record."""
        route = self._routes.get(route_ref)
        if route is None:
            route = RouteHealth(route_ref=route_ref)
            self._routes[route_ref] = route
        return route

    def get_quota_bucket(self, bucket: str) -> RouteHealth:
        """Return the shared health record for a provider quota bucket."""
        quota = self._quota_buckets.get(bucket)
        if quota is None:
            quota = RouteHealth(route_ref=f"quota/{bucket}")
            self._quota_buckets[bucket] = quota
        return quota

    def all(self) -> tuple[RouteHealth, ...]:
        """Return all stored route states."""
        return tuple(self._routes.values())

    def load(self) -> None:
        """Load persisted state if it exists."""
        if not self.path.exists():
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except OSError, ValueError, TypeError:
            return

        if not isinstance(payload, dict):
            return

        if payload.get("schema_version") != self.SCHEMA_VERSION:
            return

        routes = payload.get("routes")
        if not isinstance(routes, list):
            return

        loaded: dict[str, RouteHealth] = {}

        for item in routes:
            if not isinstance(item, dict):
                continue

            try:
                health = RouteHealth.from_dict(item)
            except KeyError, TypeError, ValueError:
                continue

            loaded[health.route_ref] = health

        self._routes = loaded

        quota_buckets = payload.get("quota_buckets", [])
        if isinstance(quota_buckets, list):
            loaded_buckets: dict[str, RouteHealth] = {}
            for item in quota_buckets:
                if not isinstance(item, dict):
                    continue
                try:
                    health = RouteHealth.from_dict(item)
                except KeyError, TypeError, ValueError:
                    continue
                if health.route_ref.startswith("quota/"):
                    loaded_buckets[health.route_ref.removeprefix("quota/")] = health
            self._quota_buckets = loaded_buckets

    def save(self) -> None:
        """Atomically persist current state."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "routes": [route.to_dict() for route in self._routes.values()],
            "quota_buckets": [
                bucket.to_dict() for bucket in self._quota_buckets.values()
            ],
        }

        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")

            os.replace(temporary_path, self.path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_path)
