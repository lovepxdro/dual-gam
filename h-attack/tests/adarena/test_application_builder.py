from __future__ import annotations

from types import SimpleNamespace

import pytest

import adarena.application as application

from adarena.application import ModeMismatchError
from adarena.core.config import ExperimentMode


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
            expected_mode=ExperimentMode.OBSERVE,
        )


def test_validate_experiment_config_em_memoria():
    registry = object()

    class FakeConfig:
        received_registry = None

        def validate(
            self,
            received_registry,
        ):
            self.received_registry = received_registry

    config = FakeConfig()

    result = application.validate_experiment_config(
        config,
        registry=registry,
    )

    assert result is config
    assert config.received_registry is registry


def test_execute_experiment_config_em_memoria(
    monkeypatch,
):
    config = SimpleNamespace(
        mode=ExperimentMode.TRAIN,
    )

    registry = object()
    expected = object()

    monkeypatch.setattr(
        application,
        "create_default_registry",
        lambda: registry,
    )

    monkeypatch.setattr(
        application,
        "validate_experiment_config",
        lambda config, registry=None: config,
    )

    monkeypatch.setattr(
        application,
        "_preflight",
        lambda config: None,
    )

    class FakeRunner:
        def __init__(
            self,
            *,
            config,
            registry,
        ):
            self.config = config
            self.registry = registry

        def run(self):
            assert self.config is config
            assert self.registry is registry
            return expected

    monkeypatch.setattr(
        application,
        "ExperimentRunner",
        FakeRunner,
    )

    result = application.execute_experiment_config(
        config
    )

    assert result is expected
