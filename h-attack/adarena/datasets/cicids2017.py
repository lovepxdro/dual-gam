from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from adarena.core.components import (
    DataRepresentation,
)

from .base import (
    DatasetAdapter,
    DatasetData,
    DatasetSchema,
)

logger = logging.getLogger(__name__)


class CICIDS2017Dataset(DatasetAdapter):
    """
    Adapter do CIC-IDS2017.

    Converte os labels originais para a semântica binária
    utilizada atualmente pela ADArena:

        0 = benigno
        1 = ataque

    Quando attack_labels é informado, apenas essas classes
    ofensivas e o tráfego benigno são carregados.
    """

    def __init__(
        self,
        *,
        label_column: str = "Label",
        benign_labels: tuple[str, ...] = (
            "BENIGN",
        ),
        attack_labels: (
            tuple[str, ...]
            | list[str]
            | None
        ) = None,
    ) -> None:
        self.label_column = label_column

        self.benign_labels = tuple(
            benign_labels
        )

        self.attack_labels = (
            tuple(attack_labels)
            if attack_labels is not None
            else None
        )

    @staticmethod
    def _normalize_label(
        value: object,
    ) -> str:
        return (
            str(value)
            .strip()
            .casefold()
        )

    @staticmethod
    def _counts(
        series: pd.Series,
    ) -> dict[str, int]:
        return {
            str(label): int(count)
            for label, count
            in series.value_counts().items()
        }

    def load(
        self,
        source: str | Path,
    ) -> DatasetData:

        source = Path(source)

        logger.info(
            "  Carregando dataset CIC-IDS2017: %s",
            source,
        )

        df = pd.read_parquet(source)

        if self.label_column not in df.columns:
            raise ValueError(
                "Dataset sem coluna obrigatória "
                f"'{self.label_column}'"
            )

        if len(df.columns) <= 1:
            raise ValueError(
                "Dataset não possui colunas de features"
            )

        raw_labels = (
            df[self.label_column]
            .astype(str)
        )

        normalized_labels = raw_labels.map(
            self._normalize_label
        )

        original_counts = self._counts(
            raw_labels
        )

        benign_values = {
            self._normalize_label(label)
            for label in self.benign_labels
        }

        # Se o usuário escolheu um ataque específico,
        # mantém somente benigno + ataque selecionado.
        if self.attack_labels is not None:

            attack_values = {
                self._normalize_label(label)
                for label in self.attack_labels
            }

            available = set(
                normalized_labels.unique()
            )

            missing = (
                attack_values
                - available
            )

            if missing:
                raise ValueError(
                    "Classes de ataque não encontradas "
                    "no dataset: "
                    + ", ".join(
                        sorted(missing)
                    )
                )

            selected_mask = (
                normalized_labels.isin(
                    benign_values
                    | attack_values
                )
            )

            filtered_samples = int(
                (~selected_mask).sum()
            )

            df = (
                df.loc[selected_mask]
                .copy()
            )

            raw_labels = (
                raw_labels.loc[selected_mask]
            )

            normalized_labels = (
                normalized_labels.loc[
                    selected_mask
                ]
            )

        else:
            filtered_samples = 0

        if df.empty:
            raise ValueError(
                "Nenhuma amostra permaneceu "
                "após filtrar o dataset"
            )

        # Remove label antes da preparação
        # do espaço de features.
        X_df = (
            df.drop(
                columns=[
                    self.label_column
                ]
            )
            .copy()
        )

        # Toda feature entregue aos modelos atuais
        # precisa ser numérica.
        X_df = X_df.apply(
            pd.to_numeric,
            errors="coerce",
        )

        X_df = (
            X_df
            .replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )
            .fillna(0.0)
        )

        feature_names = tuple(
            X_df.columns
            .astype(str)
            .tolist()
        )

        # O adapter é responsável pela semântica.
        # A partir daqui o restante da arquitetura
        # não precisa conhecer "BENIGN", "DDoS", etc.
        y = (
            ~normalized_labels.isin(
                benign_values
            )
        ).to_numpy(
            dtype=np.float32
        )

        selected_counts = self._counts(
            raw_labels
        )

        logger.info(
            "  Shape: %s | "
            "Classes: %s | "
            "Binário: benigno=%d ataque=%d",
            X_df.shape,
            selected_counts,
            int(
                (y == 0).sum()
            ),
            int(
                (y == 1).sum()
            ),
        )

        schema = DatasetSchema(
            feature_names=feature_names,
            representation=(
                DataRepresentation.FLOW_FEATURES
            ),
            label_column=self.label_column,
            label_mapping={
                "benign": 0,
                "attack": 1,
            },
        )

        return DatasetData(
            X=X_df.to_numpy(
                dtype=np.float32
            ),
            y=y,
            schema=schema,
            source=source,
            metadata={
                "dataset": "CIC-IDS2017",

                "original_label_counts": (
                    original_counts
                ),

                "selected_label_counts": (
                    selected_counts
                ),

                "benign_labels": list(
                    self.benign_labels
                ),

                "attack_labels": (
                    list(
                        self.attack_labels
                    )
                    if self.attack_labels
                    is not None
                    else None
                ),

                "filtered_samples": (
                    filtered_samples
                ),
            },
        )
