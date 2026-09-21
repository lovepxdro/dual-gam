from __future__ import annotations

from types import SimpleNamespace

import pytest

import adarena.application as application

from adarena.application import (
    PreflightError,
)
from adarena.core.config import (
    ExperimentMode,
)


def test_preflight_train_rejeita_dataset_ausente(
    tmp_path,
):
    config = SimpleNamespace(
        mode=ExperimentMode.TRAIN,
        attack_dataset=SimpleNamespace(
            source=str(
                tmp_path
                / "dataset.parquet"
            )
        ),
        benign_dataset=None,
    )

    with pytest.raises(
        PreflightError,
        match="dataset de ataque",
    ):
        application._preflight(
            config
        )


def test_preflight_simulate_exige_dry_run():
    config = SimpleNamespace(
        mode=ExperimentMode.SIMULATE,
        network=SimpleNamespace(
            dry_run=False,
        ),
    )

    with pytest.raises(
        PreflightError,
        match="dry_run = true",
    ):
        application._preflight(
            config
        )


def test_cli_train_delega_para_execute_config(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    import adarena.cli.app as cli_app
    from adarena.cli.app import app

    from typer.testing import CliRunner

    runner = CliRunner()

    path = tmp_path / "train.toml"
    path.write_text(
        'mode = "train"\n',
        encoding="utf-8",
    )

    observed = {}

    def fake_execute(
        config,
        *,
        expected_mode,
    ):
        observed["mode"] = expected_mode

        return SimpleNamespace(
            run_id="run_train",
            run_dir=tmp_path / "run_train",
            final_metrics={
                "score": 0.75,
            },
        )

    monkeypatch.setattr(
        cli_app,
        "execute_config",
        fake_execute,
    )

    result = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(path),
        ],
    )

    assert result.exit_code == 0
    assert "run_train" in result.stdout
    assert "train" in result.stdout

    assert (
        observed["mode"]
        == ExperimentMode.TRAIN
    )


def test_cli_simulate_delega_para_execute_config(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    import adarena.cli.app as cli_app
    from adarena.cli.app import app

    from typer.testing import CliRunner

    runner = CliRunner()

    path = tmp_path / "simulate.toml"
    path.write_text(
        'mode = "simulate"\n',
        encoding="utf-8",
    )

    observed = {}

    def fake_execute(
        config,
        *,
        expected_mode,
    ):
        observed["mode"] = expected_mode

        return SimpleNamespace(
            run_id="run_simulate",
            run_dir=(
                tmp_path
                / "run_simulate"
            ),
            final_metrics={
                "dry_run": True,
            },
        )

    monkeypatch.setattr(
        cli_app,
        "execute_config",
        fake_execute,
    )

    result = runner.invoke(
        app,
        [
            "simulate",
            "-c",
            str(path),
        ],
    )

    assert result.exit_code == 0
    assert "run_simulate" in result.stdout
    assert "simulate" in result.stdout

    assert (
        observed["mode"]
        == ExperimentMode.SIMULATE
    )
