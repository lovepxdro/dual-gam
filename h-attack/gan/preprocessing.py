"""
Dual-GAM — Pré-processamento de dados

Carrega o CIC-IDS2017, realiza o split treino/teste e normaliza os dados.
O StandardScaler é ajustado exclusivamente sobre o conjunto de treino.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# Features do CIC-IDS2017 que mapeiam para parâmetros de rede reais.
# Usado pelo Translator para saber quais colunas correspondem ao quê.
FEATURE_MAP = {
    # índice: (nome_feature, tipo)
    # Baseado no CIC-IDS2017 — 77 features no total
    "flow_duration": 0,
    "fwd_packet_length_max": 4,
    "fwd_packet_length_mean": 6,
    "bwd_packet_length_max": 10,
    "flow_bytes_per_sec": 14,
    "flow_packets_per_sec": 15,
    "flow_iat_mean": 16,
    "fwd_iat_total": 21,
    "fwd_iat_mean": 22,
    "fwd_psh_flags": 28,
    "fwd_urg_flags": 29,
    "fwd_header_length": 31,
    "bwd_header_length": 32,
    "fwd_packets_per_sec": 33,
    "bwd_packets_per_sec": 34,
    "pkt_length_mean": 36,
    "pkt_length_std": 37,
    "fin_flag_count": 42,
    "syn_flag_count": 43,
    "rst_flag_count": 44,
    "psh_flag_count": 45,
    "ack_flag_count": 46,
    "urg_flag_count": 47,
    "avg_pkt_size": 51,
    "init_fwd_win_bytes": 60,
}


class Preprocessador:
    """Carrega e normaliza o dataset CIC-IDS2017."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: list[str] = []

    def carregar_parquet(
        self,
        path: str | Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Carrega e limpa o arquivo .parquet do CIC-IDS2017.

        Os dados ainda não são normalizados neste ponto, pois o split
        treino/teste deve ocorrer antes do ajuste do StandardScaler.
        """
        logger.info("Carregando dataset: %s", path)

        df = pd.read_parquet(path)

        logger.info(
            "Shape: %s | Classes: %s",
            df.shape,
            df["Label"].value_counts().to_dict(),
        )

        X = df.drop(columns=["Label"]).astype(np.float32)
        y = df["Label"]

        self.feature_names = X.columns.tolist()

        # Limpeza de valores inválidos.
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Encoding dos labels.
        # Benign = 0, DDoS = 1, conforme ordenação do LabelEncoder.
        y_encoded = self.label_encoder.fit_transform(y)

        logger.info(
            "Classes: %s",
            dict(
                zip(
                    self.label_encoder.classes_,
                    range(len(self.label_encoder.classes_)),
                )
            ),
        )

        return (
            X.to_numpy(dtype=np.float32),
            y_encoded.astype(np.float32),
        )

    def split_e_normalizar(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Divide os dados antes da normalização.

        O StandardScaler é ajustado exclusivamente sobre o conjunto
        de treino e depois aplicado separadamente ao treino e ao teste.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        # Importante:
        # fit somente no treino para evitar data leakage.
        X_train_scaled = self.scaler.fit_transform(X_train)

        # O teste utiliza as estatísticas aprendidas no treino.
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(
            "Split concluído — treino: %d | teste: %d",
            len(X_train_scaled),
            len(X_test_scaled),
        )

        # Verificação simples de que o scaler foi ajustado apenas no treino.
        logger.info(
            "Média absoluta do treino normalizado: %.6f",
            float(np.abs(X_train_scaled.mean(axis=0)).mean()),
        )

        logger.info(
            "Média absoluta do teste normalizado: %.6f",
            float(np.abs(X_test_scaled.mean(axis=0)).mean()),
        )

        return (
            X_train_scaled.astype(np.float32),
            X_test_scaled.astype(np.float32),
            y_train.astype(np.float32),
            y_test.astype(np.float32),
        )

    def salvar(self, path: str | Path) -> None:
        """Salva scaler e encoder para uso no Translator."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(path / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(path / "feature_names.pkl", "wb") as f:
            pickle.dump(self.feature_names, f)

        logger.info("Preprocessador salvo em %s", path)

    @classmethod
    def carregar(cls, path: str | Path) -> "Preprocessador":
        """Carrega scaler, encoder e nomes das features."""
        path = Path(path)

        obj = cls()

        with open(path / "scaler.pkl", "rb") as f:
            obj.scaler = pickle.load(f)

        with open(path / "label_encoder.pkl", "rb") as f:
            obj.label_encoder = pickle.load(f)

        with open(path / "feature_names.pkl", "rb") as f:
            obj.feature_names = pickle.load(f)

        return obj

    def desnormalizar(self, X_scaled: np.ndarray) -> np.ndarray:
        """Converte os dados normalizados de volta à escala original."""
        return self.scaler.inverse_transform(X_scaled)
