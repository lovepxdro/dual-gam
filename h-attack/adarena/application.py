from __future__ import annotations

from pathlib import Path

from adarena.builtin import create_default_registry
from adarena.config_io import load_experiment_config
from adarena.core.config import (
    ExperimentConfig,
    ExperimentMode,
)
from adarena.core.experiment import (
    ExperimentResult,
    ExperimentRunner,
)
from adarena.core.registry import ComponentRegistry


class ModeMismatchError(ValueError):
    """O comando escolhido não corresponde ao mode do experimento."""


class PreflightError(ValueError):
    """Recurso necessário à execução não está disponível."""


def validate_experiment_config(
    config: ExperimentConfig,
    *,
    registry: ComponentRegistry | None = None,
) -> ExperimentConfig:
    """Valida uma configuração já construída em memória."""
    active_registry = registry or create_default_registry()
    config.validate(active_registry)
    return config


def validate_config_file(
    path: str | Path,
) -> ExperimentConfig:
    """Carrega e valida semanticamente uma configuração TOML."""
    config = load_experiment_config(path)
    return validate_experiment_config(config)


def execute_experiment_config(
    config: ExperimentConfig,
    *,
    expected_mode: ExperimentMode | None = None,
) -> ExperimentResult:
    """Executa uma configuração já construída em memória."""
    registry = create_default_registry()

    validate_experiment_config(
        config,
        registry=registry,
    )

    return _execute_validated_config(
        config,
        expected_mode=expected_mode,
        registry=registry,
    )


def execute_config(
    path: str | Path,
    *,
    expected_mode: ExperimentMode | None = None,
) -> ExperimentResult:
    """Executa um experimento descrito por TOML."""
    config = validate_config_file(path)

    return _execute_validated_config(
        config,
        expected_mode=expected_mode,
    )


def _execute_validated_config(
    config: ExperimentConfig,
    *,
    expected_mode: ExperimentMode | None,
    registry: ComponentRegistry | None = None,
) -> ExperimentResult:
    if (
        expected_mode is not None
        and config.mode != expected_mode
    ):
        raise ModeMismatchError(
            "o comando exige "
            f'mode = "{expected_mode.value}", '
            "mas o experimento define "
            f'mode = "{config.mode.value}"'
        )

    _preflight(config)

    active_registry = registry or create_default_registry()

    runner = ExperimentRunner(
        config=config,
        registry=active_registry,
    )

    return runner.run()


def _preflight(
    config: ExperimentConfig,
) -> None:
    """Verifica recursos externos antes de criar o run."""
    if config.mode == ExperimentMode.TRAIN:
        _preflight_train(config)
    elif config.mode == ExperimentMode.SIMULATE:
        _preflight_simulate(config)
    elif config.mode == ExperimentMode.OBSERVE:
        _preflight_observe(config)


def _preflight_train(
    config: ExperimentConfig,
) -> None:
    if config.attack_dataset is None:
        raise PreflightError("TRAIN exige attack_dataset")

    _require_file(
        config.attack_dataset.source,
        "dataset de ataque",
    )

    if config.benign_dataset is not None:
        _require_file(
            config.benign_dataset.source,
            "dataset benigno",
        )


def _preflight_simulate(
    config: ExperimentConfig,
) -> None:
    """SIMULATE permanece exclusivamente em dry-run."""
    network = config.network

    if network is None:
        raise PreflightError(
            "SIMULATE exige configuração de rede"
        )

    if not network.dry_run:
        raise PreflightError(
            "SIMULATE exige network.dry_run = true"
        )

    if config.attack_dataset is None:
        raise PreflightError(
            "SIMULATE exige attack_dataset"
        )

    _require_file(
        config.attack_dataset.source,
        "dataset de ataque",
    )

    if config.attacker is None:
        raise PreflightError(
            "SIMULATE exige attacker"
        )

    _require_file(
        config.attacker.source,
        "checkpoint do Attacker",
    )

    _require_file(
        config.defender.source,
        "checkpoint do Defender",
    )

    _require_directory(
        network.preprocessor_source,
        "preprocessador",
    )

    if network.observe:
        _require_capture_interface(config)


def _preflight_observe(
    config: ExperimentConfig,
) -> None:
    network = config.network

    if network is None:
        raise PreflightError(
            "OBSERVE exige configuração de rede"
        )

    _require_file(
        config.defender.source,
        "checkpoint do Defender",
    )

    _require_directory(
        network.preprocessor_source,
        "preprocessador",
    )

    _require_capture_interface(config)


def _require_capture_interface(
    config: ExperimentConfig,
) -> None:
    network = config.network

    if network is None or network.capture is None:
        raise PreflightError(
            "componente de captura não configurado"
        )

    iface = network.capture.params.get("iface")

    if iface is None:
        return

    iface_path = Path("/sys/class/net") / str(iface)

    if not iface_path.exists():
        raise PreflightError(
            "interface de captura não encontrada: "
            f"{iface}"
        )


def _require_file(
    value: str | None,
    label: str,
) -> Path:
    if not value:
        raise PreflightError(
            f"{label} não configurado"
        )

    path = Path(value)

    if not path.is_file():
        raise PreflightError(
            f"{label} não encontrado: {path}"
        )

    return path


def _require_directory(
    value: str | None,
    label: str,
) -> Path:
    if not value:
        raise PreflightError(
            f"{label} não configurado"
        )

    path = Path(value)

    if not path.is_dir():
        raise PreflightError(
            f"{label} não encontrado: {path}"
        )

    return path
