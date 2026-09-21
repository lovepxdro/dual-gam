from __future__ import annotations

import json

from typer.testing import CliRunner

from adarena.cli.app import app


runner = CliRunner()


def _make_run(
    root,
    *,
    output_name,
    run_id,
    mode,
    created_at,
):
    run_dir = (
        root
        / output_name
        / "experiments"
        / run_id
    )

    run_dir.mkdir(
        parents=True
    )

    for name in (
        "checkpoints",
        "metrics",
        "plots",
        "logs",
    ):
        (
            run_dir
            / name
        ).mkdir()

    payload = {
        "run_id": run_id,
        "created_at": created_at,
        "mode": mode,
        "seed": 42,
        "metadata": {
            "purpose": "test-run",
        },
        "components": {
            "attacker": None,
            "defender": (
                "adarena.binary_mlp_defender"
            ),
            "protocol": (
                "adarena.network_observation"
            ),
        },
    }

    (
        run_dir
        / "config_execucao.json"
    ).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    (
        run_dir
        / "metrics"
        / "result.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    return run_dir


def test_cli_runs_lista_execucoes(
    tmp_path,
):
    _make_run(
        tmp_path,
        output_name="observe",
        run_id="run_example",
        mode="observe",
        created_at="2026-09-21T15:00:00",
    )

    result = runner.invoke(
        app,
        [
            "runs",
            "--root",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "run_example" in result.stdout
    assert "observe" in result.stdout
    assert "test-run" in result.stdout


def test_cli_runs_show(
    tmp_path,
):
    _make_run(
        tmp_path,
        output_name="observe",
        run_id="run_example",
        mode="observe",
        created_at="2026-09-21T15:00:00",
    )

    result = runner.invoke(
        app,
        [
            "runs",
            "--root",
            str(tmp_path),
            "show",
            "run_example",
        ],
    )

    assert result.exit_code == 0
    assert "run_example" in result.stdout
    assert "binary_mlp_defender" in result.stdout
    assert "metrics" in result.stdout


def test_cli_runs_show_inexistente(
    tmp_path,
):
    result = runner.invoke(
        app,
        [
            "runs",
            "--root",
            str(tmp_path),
            "show",
            "run_missing",
        ],
    )

    assert result.exit_code == 2
    assert "Run inválido" in result.stdout
