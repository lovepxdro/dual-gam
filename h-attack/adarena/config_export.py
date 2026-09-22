from __future__ import annotations

import json
import math
import re

from pathlib import Path
from typing import Any, Mapping

from adarena.core.config import (
    ComponentSelection,
    ExperimentConfig,
)


_BARE_KEY = re.compile(
    r"^[A-Za-z0-9_-]+$"
)


def experiment_config_to_dict(
    config: ExperimentConfig,
) -> dict[str, Any]:
    """
    Converte ExperimentConfig para a mesma estrutura declarativa
    aceita pelo loader TOML da ADArena.

    Campos opcionais com valor None são omitidos porque TOML não
    possui um valor nulo.
    """

    data: dict[str, Any] = {
        "mode": config.mode.value,
        "seed": config.seed,
        "device": config.device,
        "output_dir": config.output_dir,
    }

    _put_selection(
        data,
        "attacker",
        config.attacker,
    )
    _put_selection(
        data,
        "defender",
        config.defender,
    )
    _put_selection(
        data,
        "attack_dataset",
        config.attack_dataset,
    )
    _put_selection(
        data,
        "benign_dataset",
        config.benign_dataset,
    )
    _put_selection(
        data,
        "protocol",
        config.protocol,
    )

    data["split"] = {
        "test_size": (
            config.split.test_size
        ),
        "validation_size": (
            config.split.validation_size
        ),
    }

    training = config.training

    data["training"] = {
        "noise_dim": training.noise_dim,
        "lr_defensor": (
            training.lr_defensor
        ),
        "lr_atacante": (
            training.lr_atacante
        ),
        "adam_betas": list(
            training.adam_betas
        ),
        "epsilon": training.epsilon,
        "classification_threshold": (
            training
            .classification_threshold
        ),
        "epochs_pretrain": (
            training.epochs_pretrain
        ),
        "epochs_por_rodada": (
            training.epochs_por_rodada
        ),
        "n_rodadas": (
            training.n_rodadas
        ),
        "amostras_por_rodada": (
            training
            .amostras_por_rodada
        ),
        "amostras_avaliacao_adversarial": (
            training
            .amostras_avaliacao_adversarial
        ),
        "batch_size": training.batch_size,
    }

    network = config.network

    if network is not None:
        network_data: dict[
            str,
            Any,
        ] = {
            "sample_count": (
                network.sample_count
            ),
            "capture_duration": (
                network.capture_duration
            ),
            "dry_run": network.dry_run,
            "observe": network.observe,
        }

        if (
            network.preprocessor_source
            is not None
        ):
            network_data[
                "preprocessor_source"
            ] = (
                network
                .preprocessor_source
            )

        if network.packet_limit is not None:
            network_data[
                "packet_limit"
            ] = network.packet_limit

        if (
            network
            .classification_threshold
            is not None
        ):
            network_data[
                "classification_threshold"
            ] = (
                network
                .classification_threshold
            )

        _put_selection(
            network_data,
            "renderer",
            network.renderer,
        )
        _put_selection(
            network_data,
            "network_backend",
            network.network_backend,
        )
        _put_selection(
            network_data,
            "capture",
            network.capture,
        )
        _put_selection(
            network_data,
            "extractor",
            network.extractor,
        )

        data["network"] = (
            network_data
        )

    if config.metadata:
        data["metadata"] = dict(
            config.metadata
        )

    return data


def save_experiment_config(
    config: ExperimentConfig,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Salva um ExperimentConfig em TOML reproduzível.

    Por padrão não sobrescreve um arquivo existente.
    """

    destination = Path(path)

    if (
        destination.exists()
        and not overwrite
    ):
        raise FileExistsError(
            "arquivo de configuração "
            f"já existe: {destination}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    text = dumps_experiment_config(
        config
    )

    destination.write_text(
        text,
        encoding="utf-8",
    )

    return destination


def dumps_experiment_config(
    config: ExperimentConfig,
) -> str:
    """
    Serializa a configuração para TOML.

    O writer é deliberadamente pequeno e cobre a estrutura de
    configuração da ADArena sem introduzir outra dependência.
    """

    data = experiment_config_to_dict(
        config
    )

    lines: list[str] = []

    _emit_table(
        lines,
        data,
        path=(),
        emit_header=False,
    )

    return (
        "\n".join(lines).rstrip()
        + "\n"
    )


def _put_selection(
    target: dict[str, Any],
    key: str,
    selection: (
        ComponentSelection
        | None
    ),
) -> None:
    if selection is None:
        return

    data: dict[str, Any] = {
        "component_id": (
            selection.component_id
        ),
    }

    if selection.source is not None:
        data["source"] = (
            selection.source
        )

    if selection.params:
        data["params"] = dict(
            selection.params
        )

    target[key] = data


def _emit_table(
    lines: list[str],
    data: Mapping[str, Any],
    *,
    path: tuple[str, ...],
    emit_header: bool,
) -> None:
    scalar_items: list[
        tuple[str, Any]
    ] = []

    table_items: list[
        tuple[str, Mapping[str, Any]]
    ] = []

    for key, value in data.items():
        if value is None:
            continue

        if isinstance(
            value,
            Mapping,
        ):
            table_items.append(
                (
                    str(key),
                    value,
                )
            )
        else:
            scalar_items.append(
                (
                    str(key),
                    value,
                )
            )

    if emit_header:
        if lines and lines[-1] != "":
            lines.append("")

        lines.append(
            "["
            + ".".join(
                _toml_key(part)
                for part in path
            )
            + "]"
        )

    for key, value in scalar_items:
        lines.append(
            f"{_toml_key(key)} = "
            f"{_toml_value(value)}"
        )

    for key, value in table_items:
        _emit_table(
            lines,
            value,
            path=(
                *path,
                key,
            ),
            emit_header=True,
        )


def _toml_key(
    value: str,
) -> str:
    if _BARE_KEY.fullmatch(
        value
    ):
        return value

    return json.dumps(
        value,
        ensure_ascii=False,
    )


def _toml_value(
    value: Any,
) -> str:
    if isinstance(value, bool):
        return (
            "true"
            if value
            else "false"
        )

    if isinstance(value, str):
        return json.dumps(
            value,
            ensure_ascii=False,
        )

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                "não é possível serializar "
                "float não finito para TOML"
            )

        return repr(value)

    if isinstance(
        value,
        (list, tuple),
    ):
        return (
            "["
            + ", ".join(
                _toml_value(item)
                for item in value
            )
            + "]"
        )

    raise TypeError(
        "tipo não suportado no TOML: "
        f"{type(value).__name__}"
    )
