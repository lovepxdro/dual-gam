from __future__ import annotations

import tomllib

from pathlib import Path
from typing import Any, Mapping

from adarena.core.config import (
    ComponentSelection,
    DataSplitConfig,
    ExperimentConfig,
    ExperimentMode,
    NetworkSettings,
    TrainingSettings,
)


class ConfigLoadError(ValueError):
    """Erro de estrutura ou conteúdo no arquivo TOML."""


def load_experiment_config(
    path: str | Path,
) -> ExperimentConfig:
    config_path = Path(path)

    if not config_path.is_file():
        raise ConfigLoadError(
            f"arquivo de configuração não encontrado: {config_path}"
        )

    try:
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigLoadError(
            f"TOML inválido em {config_path}: {exc}"
        ) from exc

    return experiment_config_from_dict(raw)


def experiment_config_from_dict(
    raw: Mapping[str, Any],
) -> ExperimentConfig:
    data = _mapping(raw, "raiz")

    _reject_unknown(
        data,
        {
            "mode",
            "seed",
            "device",
            "output_dir",
            "attacker",
            "defender",
            "attack_dataset",
            "benign_dataset",
            "protocol",
            "split",
            "training",
            "network",
            "metadata",
        },
        "raiz",
    )

    defender = _selection(
        data.get("defender"),
        "defender",
        required=True,
    )
    assert defender is not None

    mode_value = data.get(
        "mode",
        ExperimentMode.TRAIN.value,
    )

    try:
        mode = ExperimentMode(str(mode_value))
    except ValueError as exc:
        valid = ", ".join(
            item.value
            for item in ExperimentMode
        )
        raise ConfigLoadError(
            f"mode inválido: {mode_value}. "
            f"Valores aceitos: {valid}"
        ) from exc

    metadata = dict(
        _mapping(
            data.get("metadata", {}),
            "metadata",
        )
    )

    return ExperimentConfig(
        attacker=_selection(
            data.get("attacker"),
            "attacker",
        ),
        defender=defender,
        attack_dataset=_selection(
            data.get("attack_dataset"),
            "attack_dataset",
        ),
        benign_dataset=_selection(
            data.get("benign_dataset"),
            "benign_dataset",
        ),
        protocol=_selection(
            data.get("protocol"),
            "protocol",
        ),
        mode=mode,
        seed=int(data.get("seed", 42)),
        device=str(data.get("device", "cpu")),
        output_dir=str(
            data.get("output_dir", "/models")
        ),
        split=_split(data.get("split")),
        training=_training(
            data.get("training")
        ),
        network=_network(
            data.get("network")
        ),
        metadata=metadata,
    )


def _selection(
    raw: Any,
    section: str,
    *,
    required: bool = False,
) -> ComponentSelection | None:
    if raw is None:
        if required:
            raise ConfigLoadError(
                f"seção [{section}] é obrigatória"
            )
        return None

    data = _mapping(raw, section)

    _reject_unknown(
        data,
        {
            "component_id",
            "source",
            "params",
        },
        section,
    )

    component_id = data.get("component_id")

    if (
        not isinstance(component_id, str)
        or not component_id.strip()
    ):
        raise ConfigLoadError(
            f"{section}.component_id "
            "deve ser uma string não vazia"
        )

    source = data.get("source")

    if (
        source is not None
        and not isinstance(source, str)
    ):
        raise ConfigLoadError(
            f"{section}.source "
            "deve ser uma string"
        )

    params = dict(
        _mapping(
            data.get("params", {}),
            f"{section}.params",
        )
    )

    return ComponentSelection(
        component_id=component_id,
        params=params,
        source=source,
    )


def _split(
    raw: Any,
) -> DataSplitConfig:
    if raw is None:
        return DataSplitConfig()

    data = _mapping(raw, "split")

    _reject_unknown(
        data,
        {
            "test_size",
            "validation_size",
        },
        "split",
    )

    return DataSplitConfig(
        test_size=float(
            data.get("test_size", 0.2)
        ),
        validation_size=float(
            data.get(
                "validation_size",
                0.1,
            )
        ),
    )


def _training(
    raw: Any,
) -> TrainingSettings:
    if raw is None:
        return TrainingSettings()

    data = _mapping(raw, "training")

    _reject_unknown(
        data,
        {
            "noise_dim",
            "lr_defensor",
            "lr_atacante",
            "adam_betas",
            "epsilon",
            "classification_threshold",
            "epochs_pretrain",
            "epochs_por_rodada",
            "n_rodadas",
            "amostras_por_rodada",
            "amostras_avaliacao_adversarial",
            "batch_size",
        },
        "training",
    )

    defaults = TrainingSettings()

    adam_betas_raw = data.get(
        "adam_betas",
        defaults.adam_betas,
    )

    if (
        not isinstance(
            adam_betas_raw,
            (list, tuple),
        )
        or len(adam_betas_raw) != 2
    ):
        raise ConfigLoadError(
            "training.adam_betas deve conter "
            "exatamente dois valores"
        )

    return TrainingSettings(
        noise_dim=int(
            data.get(
                "noise_dim",
                defaults.noise_dim,
            )
        ),
        lr_defensor=float(
            data.get(
                "lr_defensor",
                defaults.lr_defensor,
            )
        ),
        lr_atacante=float(
            data.get(
                "lr_atacante",
                defaults.lr_atacante,
            )
        ),
        adam_betas=(
            float(adam_betas_raw[0]),
            float(adam_betas_raw[1]),
        ),
        epsilon=float(
            data.get(
                "epsilon",
                defaults.epsilon,
            )
        ),
        classification_threshold=float(
            data.get(
                "classification_threshold",
                defaults.classification_threshold,
            )
        ),
        epochs_pretrain=int(
            data.get(
                "epochs_pretrain",
                defaults.epochs_pretrain,
            )
        ),
        epochs_por_rodada=int(
            data.get(
                "epochs_por_rodada",
                defaults.epochs_por_rodada,
            )
        ),
        n_rodadas=int(
            data.get(
                "n_rodadas",
                defaults.n_rodadas,
            )
        ),
        amostras_por_rodada=int(
            data.get(
                "amostras_por_rodada",
                defaults.amostras_por_rodada,
            )
        ),
        amostras_avaliacao_adversarial=int(
            data.get(
                "amostras_avaliacao_adversarial",
                defaults.amostras_avaliacao_adversarial,
            )
        ),
        batch_size=int(
            data.get(
                "batch_size",
                defaults.batch_size,
            )
        ),
    )


def _network(
    raw: Any,
) -> NetworkSettings | None:
    if raw is None:
        return None

    data = _mapping(raw, "network")

    _reject_unknown(
        data,
        {
            "renderer",
            "network_backend",
            "capture",
            "extractor",
            "preprocessor_source",
            "sample_count",
            "packet_limit",
            "capture_duration",
            "classification_threshold",
            "dry_run",
            "observe",
        },
        "network",
    )

    defaults = NetworkSettings()

    threshold_raw = data.get(
        "classification_threshold",
        defaults.classification_threshold,
    )
    packet_limit_raw = data.get(
        "packet_limit",
        defaults.packet_limit,
    )

    return NetworkSettings(
        renderer=_selection(
            data.get("renderer"),
            "network.renderer",
        ),
        network_backend=_selection(
            data.get("network_backend"),
            "network.network_backend",
        ),
        capture=_selection(
            data.get("capture"),
            "network.capture",
        ),
        extractor=_selection(
            data.get("extractor"),
            "network.extractor",
        ),
        preprocessor_source=data.get(
            "preprocessor_source",
            defaults.preprocessor_source,
        ),
        sample_count=int(
            data.get(
                "sample_count",
                defaults.sample_count,
            )
        ),
        packet_limit=(
            None
            if packet_limit_raw is None
            else int(packet_limit_raw)
        ),
        capture_duration=float(
            data.get(
                "capture_duration",
                defaults.capture_duration,
            )
        ),
        classification_threshold=(
            None
            if threshold_raw is None
            else float(threshold_raw)
        ),
        dry_run=bool(
            data.get(
                "dry_run",
                defaults.dry_run,
            )
        ),
        observe=bool(
            data.get(
                "observe",
                defaults.observe,
            )
        ),
    )


def _mapping(
    value: Any,
    section: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigLoadError(
            f"{section} deve ser uma tabela TOML"
        )

    return value


def _reject_unknown(
    data: Mapping[str, Any],
    allowed: set[str],
    section: str,
) -> None:
    unknown = sorted(
        set(data) - allowed
    )

    if unknown:
        raise ConfigLoadError(
            f"campo(s) desconhecido(s) em "
            f"{section}: "
            + ", ".join(unknown)
        )
