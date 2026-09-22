"""One setup, execution, and cleanup path for installed native harnesses."""

import json
import os
import re
import secrets
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal
from urllib.request import Request

from free_claude_code.application.model_catalog import ModelCatalog
from free_claude_code.config.loader import get_settings
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.config.settings import Settings
from free_claude_code.harnesses.launch import NativeCheck, PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from ..local_http import open_local_request
from .catalog_http import fetch_proxy_model_catalog
from .common import preflight_proxy, resolve_client_binary, run_client_process

# Kept as a small seam for launcher tests and downstream embedders. The actual
# implementation lives in cli.local_http so FCC-local calls never use machine
# proxy settings.
urlopen = open_local_request


@dataclass(frozen=True, slots=True)
class LaunchContext:
    binary_path: str
    settings: Settings = field(repr=False)
    proxy_root_url: str
    auth_token: str = field(repr=False)
    base_env: Mapping[str, str] = field(repr=False)
    catalog: ModelCatalog | None
    launch_id: str = field(default_factory=lambda: secrets.token_hex(16))

    def require_catalog(self) -> ModelCatalog:
        if self.catalog is None:
            raise ValueError("harness configuration requires a model catalog")
        return self.catalog


@dataclass(frozen=True, slots=True)
class HarnessSpec:
    binary_name: str
    display_name: str
    install_hint: str
    configure: Callable[[LaunchContext, list[str], LaunchResources], PreparedLaunch]
    catalog_view: Literal["messages", "responses"] | None = None
    compatibility_check: NativeCheck | None = None
    show_router_banner: bool = False


class LaunchError(Exception):
    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def version_at_least(
    output: str, pattern: re.Pattern[str], minimum: tuple[int, int, int]
) -> bool:
    match = pattern.search(output)
    return match is not None and tuple(map(int, match.groups())) >= minimum


def _check_native(
    binary_path: str,
    check: NativeCheck,
    env: Mapping[str, str],
    *,
    exit_code: int,
    install_hint: str = "",
) -> None:
    try:
        result = subprocess.run(
            [binary_path, *check.args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=check.timeout_seconds,
            env=dict(env),
        )
        if result.returncode == 0 and check.accepts(result.stdout):
            return
    except OSError, subprocess.TimeoutExpired:
        pass
    message = check.failure_message
    if install_hint:
        message += f"\n{install_hint}"
    raise LaunchError(message, exit_code)


def launch_harness(spec: HarnessSpec, argv: Sequence[str] | None = None) -> None:
    """Prepare the FCC connection and leave command semantics to the harness."""

    args = list(sys.argv[1:] if argv is None else argv)
    base_env = dict(os.environ)
    auth_token = base_env.get("ANTHROPIC_AUTH_TOKEN", "")
    stage = "load settings"
    try:
        binary_path = resolve_client_binary(
            binary_name=spec.binary_name,
            display_name=spec.display_name,
            install_hint=spec.install_hint,
        )
        if spec.compatibility_check:
            _check_native(
                binary_path,
                spec.compatibility_check,
                base_env,
                exit_code=126,
                install_hint=spec.install_hint,
            )
        settings = get_settings()
        auth_token = settings.proxy_auth_token.strip()
        if not auth_token:
            raise LaunchError("Free Claude Code proxy authentication token is empty.")
        proxy_root_url = local_proxy_root_url(settings)
        if error := preflight_proxy(proxy_root_url):
            raise LaunchError(
                f"Free Claude Code proxy is not reachable at {proxy_root_url}: {error}\n"
                "Start it in another terminal with: fcc-server"
            )
        catalog = None
        if spec.catalog_view is not None:
            stage = "prepare model catalog"
            catalog = fetch_proxy_model_catalog(
                proxy_root_url, auth_token, view=spec.catalog_view
            )
        context = LaunchContext(
            binary_path, settings, proxy_root_url, auth_token, base_env, catalog
        )
        with ExitStack() as stack:
            stage = "prepare configuration"
            prepared = spec.configure(context, args, LaunchResources(stack))
            if prepared.activation_check:
                _check_native(
                    binary_path, prepared.activation_check, prepared.env, exit_code=1
                )
            if spec.show_router_banner and not _is_help_request(args):
                _print_router_banner(proxy_root_url, auth_token)
            stage = "start process"
            run_client_process(
                command=prepared.command,
                env=prepared.env,
                binary_name=spec.binary_name,
                display_name=spec.display_name,
                install_hint=spec.install_hint,
            )
    except (LaunchError, OSError, ValueError) as exc:
        message = (
            str(exc)
            if isinstance(exc, LaunchError)
            else f"Could not {stage} for {spec.display_name}: {exc}"
        )
        if auth_token:
            message = message.replace(auth_token, "[redacted]")
        print(message, file=sys.stderr)
        raise SystemExit(exc.exit_code if isinstance(exc, LaunchError) else 1) from None


def _print_router_banner(proxy_root_url: str, auth_token: str) -> None:
    """Print a best-effort FCC route snapshot before the native client starts."""
    try:
        request = Request(
            f"{proxy_root_url.rstrip('/')}/admin/api/router/status",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        with urlopen(request, timeout=0.35) as response:
            payload = json.load(response)
        routes = payload.get("routes", [])
        summary = payload.get("summary", {})
        selected = payload.get("selected") or "none"
        selected_rank = payload.get("selected_rank")
        usable = summary.get("executable", 0)
        total = summary.get("total", len(routes))
        selected_payload = next(
            (route for route in routes if route.get("provider_model_ref") == selected),
            None,
        )
        health = (selected_payload or {}).get("health", {})
        capability = (selected_payload or {}).get("capability", {})
        quota = (selected_payload or {}).get("quota", {})
        state = health.get("state", "unknown").upper()
        quota_state = quota.get("state", "unknown").upper()
        reset_at = quota.get("reset_at") or health.get("quarantine_until") or ""
        if quota_state == "QUARANTINED" and reset_at:
            try:
                if datetime.fromisoformat(reset_at) <= datetime.now(UTC):
                    quota_state = "RESET_DUE"
            except ValueError:
                pass
        tier = capability.get("tier_name") or "UNKNOWN_TIER"
        position = f"#{selected_rank}/{total}" if selected_rank else "unranked"
        suffix = f" · {position} · tier {tier} · quota {quota_state}"
        if reset_at:
            suffix += f" · reset {reset_at}"
        last_successful = payload.get("last_successful")
        if last_successful and last_successful.get("provider_model_ref") != selected:
            suffix += f" · last completed {last_successful['provider_model_ref']}"
        print(
            f"FCC ROUTE  {selected} · {usable}/{total} usable · {state}{suffix}",
            file=sys.stderr,
            flush=True,
        )
        _print_route_preview(routes, selected, usable, total)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        # Observability must never prevent a Claude session from launching.
        return


def _is_help_request(args: Sequence[str]) -> bool:
    """Keep informational help fast and free of optional observability calls."""
    return any(arg in {"-h", "--help"} for arg in args)


def _print_route_preview(
    routes: list[object], selected: str, usable: object, total: object
) -> None:
    """Print a compact numbered view of the configured fallback pool."""
    # Preserve configured fallback order. It is the operator's "model number"
    # and keeps throttled/HTTP-failed entries visible instead of moving every
    # currently executable route ahead of them.
    ordered = [route for route in routes[:10] if isinstance(route, dict)]
    if selected and not any(
        isinstance(route, dict) and route.get("provider_model_ref") == selected
        for route in ordered
    ):
        selected_route = next(
            (
                route
                for route in routes
                if isinstance(route, dict)
                and route.get("provider_model_ref") == selected
            ),
            None,
        )
        if selected_route is not None:
            ordered.append(selected_route)
    if not ordered:
        return

    print(
        f"FCC ROUTES  showing {len(ordered)}/{total} · usable {usable}",
        file=sys.stderr,
    )
    for index, route in enumerate(ordered, 1):
        if not isinstance(route, dict):
            continue
        ref = str(route.get("provider_model_ref") or route.get("model") or "?")
        marker = " < NOW" if ref == selected else ""
        print(
            f"  {index:02d} {_route_preview_state(route):<12} {ref}{marker}",
            file=sys.stderr,
        )


def _route_preview_state(route: dict[str, object]) -> str:
    """Reduce one route's health/quota evidence to a glanceable label."""
    quota = route.get("quota")
    quota_state = quota.get("state") if isinstance(quota, dict) else None
    reset_at = quota.get("reset_at") if isinstance(quota, dict) else None
    if quota_state == "quarantined":
        if isinstance(reset_at, str):
            try:
                if datetime.fromisoformat(reset_at) <= datetime.now(UTC):
                    return "RESET DUE"
            except ValueError:
                pass
        return "EXHAUSTED"
    health = route.get("health")
    health_state = health.get("state") if isinstance(health, dict) else None
    if isinstance(health, dict) and health.get("last_failure_status"):
        return str(health["last_failure_status"])
    if health_state == "backoff":
        return "THROTTLED"
    if route.get("executable"):
        return "READY"
    reason = route.get("exclusion_reason")
    return str(reason or "UNKNOWN")
