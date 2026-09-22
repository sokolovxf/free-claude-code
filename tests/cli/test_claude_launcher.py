"""Claude owns permission options and prompt interpretation."""

import pytest

from tests.cli.conftest import LaunchCapture
from tests.cli.test_launcher_workflow import launch


def test_router_banner_reports_selected_route_and_capacity(monkeypatch, capsys) -> None:
    from free_claude_code.cli.launchers import runner

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b""

    payload = {
        "selected": "groq/openai/gpt-oss-120b",
        "summary": {"executable": 13, "total": 35},
        "routes": [
            {
                "provider_model_ref": "groq/openai/gpt-oss-120b",
                "capability": {"tier_name": "TIER_2"},
                "health": {"state": "available"},
                "quota": {"state": "unknown", "reset_at": None},
            }
        ],
    }

    monkeypatch.setattr(runner, "urlopen", lambda *_args, **_kwargs: Response())
    monkeypatch.setattr(runner.json, "load", lambda _response: payload)

    runner._print_router_banner("http://127.0.0.1:8083", "freecc")

    output = capsys.readouterr().err
    assert "FCC ROUTE  groq/openai/gpt-oss-120b" in output
    assert "13/35 usable" in output


def test_router_preview_preserves_route_order_and_shows_http_failure_codes(capsys) -> None:
    from free_claude_code.cli.launchers import runner

    runner._print_route_preview(
        [
            {
                "provider_model_ref": "route-two",
                "rank": 2,
                "executable": False,
                "health": {"state": "backoff", "last_failure_status": 509},
                "quota": {"state": "unknown"},
            },
            {
                "provider_model_ref": "route-one",
                "rank": 1,
                "executable": True,
                "health": {"state": "available"},
                "quota": {"state": "unknown"},
            },
        ],
        "route-one",
        1,
        2,
    )

    output = capsys.readouterr().err
    assert output.index("01 509") < output.index("02 READY")
    assert "route-one < NOW" in output


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["--permission-mode", "auto", "fix tests"],
        ["--permission-mode=acceptEdits", "fix tests"],
        ["--dangerously-skip-permissions", "fix tests"],
        ["--permission-mode"],
        ["--", "--permission-mode=auto"],
        ["explain --permission-mode auto"],
    ],
)
def test_claude_owns_permission_selection(
    args: list[str], launch_capture: LaunchCapture
) -> None:
    launch("claude", args)
    assert launch_capture.commands == [["claude", *args]]
    env = launch_capture.environments[0]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8182"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "launcher-test-token"
    assert env["CLAUDE_CODE_USE_GATEWAY"] == "1"
    assert "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY" not in env
