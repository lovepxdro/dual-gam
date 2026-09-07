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

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    preprocessor: Any
    dataset_data: Any

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
