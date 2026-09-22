from __future__ import annotations

from typer.testing import CliRunner

import adarena.cli.app as cli_app


runner = CliRunner()


def test_cli_sem_subcomando_abre_tui(
    monkeypatch,
):
    called = {
        "value": False,
    }

    def fake_run_tui():
        called["value"] = True

    monkeypatch.setattr(
        cli_app,
        "run_tui",
        fake_run_tui,
    )

    result = runner.invoke(
        cli_app.app,
        [],
    )

    assert result.exit_code == 0
    assert called["value"] is True


def test_cli_version_nao_abre_tui(
    monkeypatch,
):
    called = {
        "value": False,
    }

    def fake_run_tui():
        called["value"] = True

    monkeypatch.setattr(
        cli_app,
        "run_tui",
        fake_run_tui,
    )

    result = runner.invoke(
        cli_app.app,
        [
            "version",
        ],
    )

    assert result.exit_code == 0
    assert called["value"] is False
    assert "ADArena" in result.stdout
