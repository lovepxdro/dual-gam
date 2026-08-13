"""
Dual-GAM — Pré-processamento de dados

Carrega o CIC-IDS2017, realiza auditoria do dataset, remove duplicatas exatas,
divide os dados em train/validation/test e normaliza os conjuntos.

Regras metodológicas:
- nenhuma estatística do conjunto de validação/teste participa do fit do scaler;
- duplicatas exatas (features + label) são removidas antes do split;
- os três conjuntos são auditados por hash e qualquer interseção interrompe a execução;
- o preprocessador persistido é exatamente o utilizado no treino.
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


# Mantido por compatibilidade com componentes antigos. O Translator atualizado
# prefere resolver as features pelo nome salvo em ``feature_names``.
FEATURE_MAP = {
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
    """Carrega, audita, separa e normaliza o dataset CIC-IDS2017."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names: list[str] = []
        self.audit_info: dict[str, object] = {}

    def carregar_parquet(
        self,
        path: str | Path,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Carrega e limpa o .parquet sem normalizar os dados."""
        logger.info("  Carregando dataset: %s", path)

        df = pd.read_parquet(path)
        if "Label" not in df.columns:
            raise ValueError("Dataset sem coluna obrigatória 'Label'")

        logger.info(
            "  Shape: %s | Classes: %s",
            df.shape,
            df["Label"].value_counts().to_dict(),
        )

        X = df.drop(columns=["Label"]).copy()
        y = df["Label"].copy()

        # Garante que todo o espaço de features seja numérico e finito.
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self.feature_names = X.columns.astype(str).tolist()
        y_encoded = self.label_encoder.fit_transform(y)

        logger.info(
            "  Classes mapeadas: %s",
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

    def _dataframe_com_label(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> pd.DataFrame:
        if X.ndim != 2:
            raise ValueError("X deve possuir shape [N, features]")
        if len(X) != len(y):
            raise ValueError("X e y possuem quantidades diferentes de amostras")
        if self.feature_names and X.shape[1] != len(self.feature_names):
            raise ValueError(
                "Número de features incompatível com feature_names: "
                f"{X.shape[1]} != {len(self.feature_names)}"
            )

        columns = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=columns)
        df["_label"] = y
        return df

    def auditar_duplicatas(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> int:
        """Conta duplicatas exatas considerando features + label."""
        df = self._dataframe_com_label(X, y)
        duplicadas = int(df.duplicated(keep="first").sum())

        if duplicadas:
            logger.warning("    Duplicatas exatas no dataset: %d", duplicadas)
        else:
            logger.info("    Duplicatas exatas no dataset: 0")

        return duplicadas

    def remover_duplicatas(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Remove duplicatas exatas antes do split.

        A remoção ocorre sobre ``features + label``. Isso evita que cópias da
        mesma observação terminem em conjuntos distintos e contaminem a avaliação.
        """
        df = self._dataframe_com_label(X, y)
        duplicate_mask = df.duplicated(keep="first")
        n_removed = int(duplicate_mask.sum())

        if n_removed == 0:
            return X, y, 0

        keep = ~duplicate_mask.to_numpy()
        X_unique = X[keep]
        y_unique = y[keep]

        logger.warning(
            "    Duplicatas removidas antes do split: %d | amostras restantes: %d",
            n_removed,
            len(X_unique),
        )
        return X_unique, y_unique, n_removed

    def _hashes(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> set[int]:
        df = self._dataframe_com_label(X, y)
        return set(
            pd.util.hash_pandas_object(
                df,
                index=False,
            ).to_numpy(dtype=np.uint64)
        )

    def auditar_isolamento_triplo(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict[str, int]:
        """Verifica interseções exatas entre treino, validação e teste."""
        hashes_train = self._hashes(X_train, y_train)
        hashes_val = self._hashes(X_val, y_val)
        hashes_test = self._hashes(X_test, y_test)

        intersecoes = {
            "treino_validacao": len(hashes_train & hashes_val),
            "treino_teste": len(hashes_train & hashes_test),
            "validacao_teste": len(hashes_val & hashes_test),
        }

        logger.info(
            "    Interseção treino/validação: %d",
            intersecoes["treino_validacao"],
        )
        logger.info(
            "    Interseção treino/teste: %d",
            intersecoes["treino_teste"],
        )
        logger.info(
            "    Interseção validação/teste: %d",
            intersecoes["validacao_teste"],
        )

        if any(intersecoes.values()):
            raise RuntimeError(
                "Data leakage detectado entre treino, validação e teste: "
                f"{intersecoes}"
            )

        logger.info("    Isolamento treino/validação/teste: OK")
        return intersecoes

    def split_e_normalizar(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        random_state: int = 42,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Remove duplicatas, divide em treino/validação/teste e normaliza.

        Por padrão: 70% treino, 10% validação e 20% teste.
        O StandardScaler é ajustado exclusivamente sobre o conjunto de treino.
        """
        if test_size <= 0 or validation_size <= 0:
            raise ValueError("test_size e validation_size devem ser > 0")
        if test_size + validation_size >= 1:
            raise ValueError("test_size + validation_size deve ser menor que 1")

        logger.info("  Auditoria:")
        n_duplicates = self.auditar_duplicatas(X, y)
        X, y, n_removed = self.remover_duplicatas(X, y)

        # 1) Separa o teste final.
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        # 2) Retira a validação do bloco restante mantendo a proporção global.
        validation_relative = validation_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=validation_relative,
            random_state=random_state,
            stratify=y_temp,
        )

        logger.info(
            "  Split: treino=%d | validação=%d | teste=%d",
            len(X_train),
            len(X_val),
            len(X_test),
        )

        intersecoes = self.auditar_isolamento_triplo(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )

        logger.info("  Normalização:")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        train_mean = float(np.abs(X_train_scaled.mean(axis=0)).mean())
        val_mean = float(np.abs(X_val_scaled.mean(axis=0)).mean())
        test_mean = float(np.abs(X_test_scaled.mean(axis=0)).mean())

        logger.info("    Média abs. treino:    %.6f", train_mean)
        logger.info("    Média abs. validação: %.6f", val_mean)
        logger.info("    Média abs. teste:     %.6f", test_mean)

        self.audit_info = {
            "duplicatas_detectadas": n_duplicates,
            "duplicatas_removidas": n_removed,
            "amostras_pos_deduplicacao": int(len(X)),
            "split": {
                "treino": int(len(X_train)),
                "validacao": int(len(X_val)),
                "teste": int(len(X_test)),
            },
            "intersecoes": intersecoes,
            "media_abs_normalizada": {
                "treino": train_mean,
                "validacao": val_mean,
                "teste": test_mean,
            },
            "random_state": int(random_state),
        }

        return (
            X_train_scaled.astype(np.float32),
            X_val_scaled.astype(np.float32),
            X_test_scaled.astype(np.float32),
            y_train.astype(np.float32),
            y_val.astype(np.float32),
            y_test.astype(np.float32),
        )

    def salvar(self, path: str | Path) -> None:
        """Salva scaler, encoder, nomes das features e auditoria."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)

        with open(path / "label_encoder.pkl", "wb") as f:
            pickle.dump(self.label_encoder, f)

        with open(path / "feature_names.pkl", "wb") as f:
            pickle.dump(self.feature_names, f)

        with open(path / "audit_info.pkl", "wb") as f:
            pickle.dump(self.audit_info, f)

        logger.info("  Preprocessador: OK")
        logger.debug("Preprocessador salvo em %s", path)

    @classmethod
    def carregar(
        cls,
        path: str | Path,
    ) -> "Preprocessador":
        """Carrega scaler, encoder e nomes das features."""
        path = Path(path)
        obj = cls()

        with open(path / "scaler.pkl", "rb") as f:
            obj.scaler = pickle.load(f)

        with open(path / "label_encoder.pkl", "rb") as f:
            obj.label_encoder = pickle.load(f)

        with open(path / "feature_names.pkl", "rb") as f:
            obj.feature_names = pickle.load(f)

        audit_path = path / "audit_info.pkl"
        if audit_path.exists():
            with open(audit_path, "rb") as f:
                obj.audit_info = pickle.load(f)

        return obj

    def desnormalizar(
        self,
        X_scaled: np.ndarray,
    ) -> np.ndarray:
        """Converte dados normalizados de volta à escala original."""
        X_scaled = np.asarray(X_scaled, dtype=np.float64)
        if X_scaled.ndim != 2:
            raise ValueError("X_scaled deve possuir shape [N, features]")
        if X_scaled.shape[1] != len(self.scaler.mean_):
            raise ValueError(
                "Número de features incompatível com o scaler: "
                f"{X_scaled.shape[1]} != {len(self.scaler.mean_)}"
            )
        return self.scaler.inverse_transform(X_scaled)
