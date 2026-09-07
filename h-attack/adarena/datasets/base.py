from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from adarena.core.components import DataRepresentation


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """
    Descreve a representação entregue por um DatasetAdapter.

    A ADArena trabalha com labels padronizados:
        0 = benigno
        1 = ataque
    """

    feature_names: tuple[str, ...]
    representation: DataRepresentation

    label_column: str | None = None

    label_mapping: Mapping[str, int] = field(
        default_factory=lambda: {
            "benign": 0,
            "attack": 1,
        }
    )

    def __post_init__(self) -> None:
        if not self.feature_names:
            raise ValueError(
                "Dataset precisa possuir ao menos uma feature"
            )

        if len(set(self.feature_names)) != len(
            self.feature_names
        ):
            raise ValueError(
                "Dataset possui nomes de features duplicados"
            )

        object.__setattr__(
            self,
            "label_mapping",
            MappingProxyType(
                dict(self.label_mapping)
            ),
        )


@dataclass(slots=True)
class DatasetData:
    """
    Dados já convertidos para a representação compreendida
    pela ADArena.
    """

    X: np.ndarray
    y: np.ndarray
    schema: DatasetSchema

    source: Path | None = None
    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.X = np.asarray(
            self.X,
            dtype=np.float32,
        )

        self.y = np.asarray(
            self.y,
            dtype=np.float32,
        )

        if self.X.ndim != 2:
            raise ValueError(
                "X deve possuir shape [N, features]"
            )

        if self.y.ndim != 1:
            raise ValueError(
                "y deve possuir shape [N]"
            )

        if len(self.X) != len(self.y):
            raise ValueError(
                "X e y possuem quantidades diferentes "
                "de amostras"
            )

        if (
            self.X.shape[1]
            != len(self.schema.feature_names)
        ):
            raise ValueError(
                "Número de features incompatível "
                "com o schema: "
                f"{self.X.shape[1]} != "
                f"{len(self.schema.feature_names)}"
            )

        if not np.isfinite(self.X).all():
            raise ValueError(
                "Dataset contém valores não finitos "
                "após o carregamento"
            )

    @property
    def attack_mask(self) -> np.ndarray:
        return self.y == 1

    @property
    def benign_mask(self) -> np.ndarray:
        return self.y == 0


class DatasetAdapter(ABC):
    """
    Contrato para datasets utilizados pela ADArena.

    Cada implementação é responsável por entender o formato
    original do dataset e convertê-lo para uma representação
    conhecida pela plataforma.
    """

    @abstractmethod
    def load(
        self,
        source: str | Path,
    ) -> DatasetData:
        raise NotImplementedError
