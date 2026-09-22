from __future__ import annotations

from types import SimpleNamespace

import pytest

import adarena.application as application

from adarena.application import (
    ModeMismatchError,
    PreflightError,
)
from adarena.core.config import (
    ExperimentMode,
)


def test_execute_config_rejeita_mode_diferente(
    monkeypatch,
):
    config = SimpleNamespace(
        mode=ExperimentMode.TRAIN,
    )

    monkeypatch.setattr(
        application,
        "validate_config_file",
        lambda path: config,
    )

    with pytest.raises(
        ModeMismatchError,
        match="observe",
    ):
        application.execute_config(
            "config.toml",
            expected_mode=(
                ExperimentMode.OBSERVE
            ),
        )


def test_execute_config_valida_preflight_e_executa(
    monkeypatch,
):
    calls = []

    class FakeConfig:
        mode = ExperimentMode.OBSERVE

        def validate(
            self,
            registry,
        ):
            calls.append(
                ("validate", registry)
            )

    fake_config = FakeConfig()
    fake_registry = object()
    fake_result = object()

    class FakeRunner:
        def __init__(
            self,
            *,
            config,
            registry,
        ):
            assert config is fake_config
            assert registry is fake_registry
            calls.append(
                ("runner", registry)
            )

        def run(self):
            calls.append(
                ("run", None)
            )
            return fake_result

    monkeypatch.setattr(
        application,
        "load_experiment_config",
        lambda path: fake_config,
    )
    monkeypatch.setattr(
        application,
        "create_default_registry",
        lambda: fake_registry,
    )
    monkeypatch.setattr(
        application,
        "_preflight",
        lambda config: calls.append(
            ("preflight", None)
        ),
    )
    monkeypatch.setattr(
        application,
        "ExperimentRunner",
        FakeRunner,
    )

    result = application.execute_config(
        "config.toml",
        expected_mode=(
            ExperimentMode.OBSERVE
        ),
    )

    assert result is fake_result

    assert [
        name
        for name, _
        in calls
    ] == [
        "validate",
        "preflight",
        "runner",
        "run",
    ]


def test_preflight_observe_rejeita_checkpoint_ausente(
    tmp_path,
):
    config = SimpleNamespace(
        mode=ExperimentMode.OBSERVE,
        defender=SimpleNamespace(
            source=str(
                tmp_path
                / "defender.pth"
            )
        ),
        network=SimpleNamespace(),
    )

    with pytest.raises(
        PreflightError,
        match="checkpoint",
    ):
        application._preflight(
            config
        )
