from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)

from dataclasses import (
    dataclass,
    field,
)

from pathlib import Path

from typing import Any

import numpy as np

from adarena.core.config import (
    ExperimentConfig,
)

from adarena.core.registry import (
    ComponentRegistry,
)


@dataclass(slots=True)
class ProtocolContext:
    config: ExperimentConfig

    registry: ComponentRegistry

    run_id: str
    run_dir: Path

    # Protocolos de treinamento/simulação recebem
    # os splits preparados pelo Runner. Protocolos
    # puramente observacionais não dependem de
    # dataset e, portanto, recebem None nestes campos.
    X_train: np.ndarray | None
    X_val: np.ndarray | None
    X_test: np.ndarray | None

    y_train: np.ndarray | None
    y_val: np.ndarray | None
    y_test: np.ndarray | None

    # O preprocessador continua obrigatório para
    # qualquer protocolo que execute inferência.
    preprocessor: Any

    # Pode ser None em modos que observam apenas
    # tráfego já existente na rede.
    dataset_data: Any | None

    config_snapshot: dict[
        str,
        Any,
    ]


@dataclass(slots=True)
class ProtocolResult:
    history: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    final_metrics: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    evaluations: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    artifacts: dict[
        str,
        str,
    ] = field(
        default_factory=dict
    )


class ExperimentProtocol(ABC):

    @abstractmethod
    def run(
        self,
        context: ProtocolContext,
    ) -> ProtocolResult:
        raise NotImplementedError
