"""
Dual-GAM — Loop de treinamento adversarial

Além do treinamento, este módulo implementa um protocolo de avaliação
adversarial reproduzível: cada A_r é avaliado sobre as mesmas amostras-base
e os mesmos vetores de ruído antes e depois da adaptação do Defensor.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .models import Atacante, Defensor

logger = logging.getLogger(__name__)


def definir_seed(seed: int) -> None:
    """Configura as principais fontes de aleatoriedade da execução."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=True)
    logger.info("  Seed: %d", seed)


@dataclass
class TrainingConfig:
    seed: int = 42

    input_dim: int = 77
    noise_dim: int = 32

    lr_defensor: float = 1e-3
    lr_atacante: float = 2e-4
    adam_betas: tuple = (0.5, 0.999)
    epsilon: float = 0.3
    classification_threshold: float = 0.5

    epochs_pretrain: int = 5
    epochs_por_rodada: int = 3
    n_rodadas: int = 20

    amostras_por_rodada: int = 5000
    amostras_avaliacao_adversarial: int = 5000
    batch_size: int = 512

    device: str = "cpu"
    checkpoint_dir: Path = Path("/models")

    historico: dict = field(
        default_factory=lambda: {
            # Compatibilidade: taxa_evasao representa a avaliação pré-adaptação.
            "taxa_evasao": [],
            "taxa_evasao_pre_adaptacao": [],
            "taxa_evasao_pos_adaptacao": [],
            "reducao_evasao_pp": [],
            "acuracia_defensor_validacao": [],
            "precision_defensor_validacao": [],
            "recall_defensor_validacao": [],
            "f1_defensor_validacao": [],
            "fpr_defensor_validacao": [],
            "fnr_defensor_validacao": [],
            "roc_auc_defensor_validacao": [],
            "matriz_confusao_defensor_validacao": [],
            "confianca_media_adv_pre": [],
            "confianca_media_adv_pos": [],
            "perturbacao_l1_media": [],
            "perturbacao_l2_media": [],
            "perturbacao_linf_media": [],
            "loss_atacante": [],
            "loss_defensor": [],
        }
    )


class AdversarialTrainer:
    """Orquestra o ciclo co-evolutivo Atacante ↔ Defensor."""

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        self.defensor = Defensor(config.input_dim).to(self.device)
        self.atacante = Atacante(
            config.noise_dim,
            config.input_dim,
        ).to(self.device)

        self.criterio = nn.BCELoss()

        self.opt_def = torch.optim.Adam(
            self.defensor.parameters(),
            lr=config.lr_defensor,
        )

        self.opt_atk = torch.optim.Adam(
            self.atacante.parameters(),
            lr=config.lr_atacante,
            betas=config.adam_betas,
        )

    # ── Dados ──────────────────────────────────────────────────────────────

    def carregar_dados(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Converte arrays, cria loaders e fixa o conjunto adversarial de avaliação."""
        self.X_train_t = torch.FloatTensor(X_train).to(self.device)
        self.y_train_t = torch.FloatTensor(y_train).to(self.device)
        self.X_val_t = torch.FloatTensor(X_val).to(self.device)
        self.y_val_t = torch.FloatTensor(y_val).to(self.device)
        self.X_test_t = torch.FloatTensor(X_test).to(self.device)
        self.y_test_t = torch.FloatTensor(y_test).to(self.device)

        self.train_loader = DataLoader(
            TensorDataset(self.X_train_t, self.y_train_t),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        mask_ddos_train = y_train == 1
        self.X_ddos_t = torch.FloatTensor(
            X_train[mask_ddos_train]
        ).to(self.device)

        self.ddos_loader = DataLoader(
            TensorDataset(self.X_ddos_t),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        # Avaliação adversarial: usa DDoS da validação, nunca amostras de treino.
        X_val_ddos = X_val[y_val == 1]
        if len(X_val_ddos) == 0:
            raise RuntimeError(
                "Conjunto de validação não contém amostras DDoS."
            )

        n_eval = min(
            self.cfg.amostras_avaliacao_adversarial,
            len(X_val_ddos),
        )

        # RNGs independentes e reproduzíveis. A seed do treino não é consumida aqui.
        rng = np.random.default_rng(self.cfg.seed)
        indices = rng.choice(
            len(X_val_ddos),
            size=n_eval,
            replace=False,
        )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.cfg.seed)
        z_eval = torch.randn(
            n_eval,
            self.cfg.noise_dim,
            generator=generator,
        )

        self.X_adv_eval_base = torch.FloatTensor(
            X_val_ddos[indices]
        ).to(self.device)
        self.z_adv_eval = z_eval.to(self.device)

        logger.info(
            "  Dados: treino=%d | validação=%d | teste=%d | DDoS treino=%d",
            len(X_train),
            len(X_val),
            len(X_test),
            mask_ddos_train.sum(),
        )
        logger.info(
            "  Avaliação adversarial fixa: %d amostras DDoS de validação | threshold=%.2f",
            n_eval,
            self.cfg.classification_threshold,
        )

    # ── Pré-treino ────────────────────────────────────────────────────────

    def pretreinar_defensor(self) -> None:
        logger.info(
            "  Pré-treino do Defensor: %d epochs",
            self.cfg.epochs_pretrain,
        )

        self.defensor.train()

        for epoch in range(self.cfg.epochs_pretrain):
            for X_batch, y_batch in self.train_loader:
                self.opt_def.zero_grad()
                pred = self.defensor(X_batch)
                loss = self.criterio(pred, y_batch)
                loss.backward()
                self.opt_def.step()

            logger.debug(
                "Pré-treino epoch %d/%d",
                epoch + 1,
                self.cfg.epochs_pretrain,
            )

        metricas = self._calcular_metricas_classificacao(
            self.X_val_t,
            self.y_val_t,
        )
        logger.info(
            "    Validação inicial: Acc %.2f%% | Precision %.2f%% | "
            "Recall %.2f%% | F1 %.2f%% | FNR %.2f%%",
            metricas["accuracy"] * 100,
            metricas["precision"] * 100,
            metricas["recall"] * 100,
            metricas["f1"] * 100,
            metricas["fnr"] * 100,
        )

    def salvar_defensor_inicial(self) -> None:
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.cfg.checkpoint_dir / "defensor_rodada_00.pth"
        torch.save(self.defensor.state_dict(), path)
        logger.info("    Checkpoint inicial: D0")
        logger.debug("Checkpoint D0 salvo em %s", path)

    # ── Ciclo adversarial ─────────────────────────────────────────────────

    def rodar_ciclo(self) -> dict:
        logger.info("  Ciclo adversarial: %d rodadas", self.cfg.n_rodadas)

        for rodada_idx in range(self.cfg.n_rodadas):
            rodada = rodada_idx + 1

            # 1) Atualiza A_r contra D_(r-1).
            loss_atk = self._treinar_atacante()

            # 2) Gera UMA vez o conjunto adversarial fixo de A_r.
            amostras_eval, perturbacao_eval = (
                self._gerar_amostras_avaliacao_adversarial()
            )

            # 3) Mede A_r × D_(r-1).
            adv_pre = self._avaliar_adversarial(amostras_eval)

            # 4) Adapta o Defensor e mede A_r × D_r sobre AS MESMAS amostras.
            loss_def = self._retreinar_defensor()
            adv_pos = self._avaliar_adversarial(amostras_eval)

            # 5) Mede a magnitude da perturbação produzida por A_r.
            normas = self._medir_normas_perturbacao(perturbacao_eval)

            # 6) Desempenho convencional durante o desenvolvimento usa validação.
            metricas_val = self._calcular_metricas_classificacao(
                self.X_val_t,
                self.y_val_t,
            )

            evasao_pre = adv_pre["evasao"]
            evasao_pos = adv_pos["evasao"]
            reducao_pp = (evasao_pre - evasao_pos) * 100.0

            self.cfg.historico["taxa_evasao"].append(evasao_pre)
            self.cfg.historico["taxa_evasao_pre_adaptacao"].append(evasao_pre)
            self.cfg.historico["taxa_evasao_pos_adaptacao"].append(evasao_pos)
            self.cfg.historico["reducao_evasao_pp"].append(reducao_pp)
            self.cfg.historico["acuracia_defensor_validacao"].append(metricas_val["accuracy"])
            self.cfg.historico["precision_defensor_validacao"].append(metricas_val["precision"])
            self.cfg.historico["recall_defensor_validacao"].append(metricas_val["recall"])
            self.cfg.historico["f1_defensor_validacao"].append(metricas_val["f1"])
            self.cfg.historico["fpr_defensor_validacao"].append(metricas_val["fpr"])
            self.cfg.historico["fnr_defensor_validacao"].append(metricas_val["fnr"])
            self.cfg.historico["roc_auc_defensor_validacao"].append(metricas_val["roc_auc"])
            self.cfg.historico["matriz_confusao_defensor_validacao"].append(metricas_val["confusion_matrix"])
            self.cfg.historico["confianca_media_adv_pre"].append(adv_pre["confianca_media"])
            self.cfg.historico["confianca_media_adv_pos"].append(adv_pos["confianca_media"])
            self.cfg.historico["perturbacao_l1_media"].append(normas["l1"])
            self.cfg.historico["perturbacao_l2_media"].append(normas["l2"])
            self.cfg.historico["perturbacao_linf_media"].append(normas["linf"])
            self.cfg.historico["loss_atacante"].append(loss_atk)
            self.cfg.historico["loss_defensor"].append(loss_def)

            logger.info(
                "    Rodada %02d/%02d | A%d×D%d: %5.1f%% → A%d×D%d: %5.1f%% "
                "| Δ: %+5.1f pp | F1 %.2f%% | Recall %.2f%% | FNR %.2f%%",
                rodada,
                self.cfg.n_rodadas,
                rodada,
                rodada - 1,
                evasao_pre * 100,
                rodada,
                rodada,
                evasao_pos * 100,
                -reducao_pp,
                metricas_val["f1"] * 100,
                metricas_val["recall"] * 100,
                metricas_val["fnr"] * 100,
            )
            logger.debug(
                "Rodada %02d | Acc %.4f | Precision %.4f | Recall %.4f | "
                "F1 %.4f | FPR %.4f | FNR %.4f | ROC-AUC %.4f | "
                "Confusão %s | Confiança adv %.4f→%.4f | "
                "Perturbação L1 %.4f | L2 %.4f | Linf %.4f",
                rodada,
                metricas_val["accuracy"],
                metricas_val["precision"],
                metricas_val["recall"],
                metricas_val["f1"],
                metricas_val["fpr"],
                metricas_val["fnr"],
                metricas_val["roc_auc"],
                metricas_val["confusion_matrix"],
                adv_pre["confianca_media"],
                adv_pos["confianca_media"],
                normas["l1"],
                normas["l2"],
                normas["linf"],
            )

            self._salvar_checkpoint(rodada)

        return self.cfg.historico

    def _treinar_atacante(self) -> float:
        """Atualiza o Atacante contra o Defensor atual."""
        for p in self.defensor.parameters():
            p.requires_grad = False

        self.atacante.train()
        self.defensor.eval()
        loss_total = 0.0

        for _ in range(self.cfg.epochs_por_rodada):
            for (X_ddos_batch,) in self.ddos_loader:
                batch_size = X_ddos_batch.shape[0]
                z = torch.randn(
                    batch_size,
                    self.cfg.noise_dim,
                    device=self.device,
                )

                perturbacao = self.atacante(z)
                amostras = X_ddos_batch + self.cfg.epsilon * perturbacao
                predicoes = self.defensor(amostras)
                alvo = torch.zeros(batch_size, device=self.device)

                loss = self.criterio(predicoes, alvo)
                self.opt_atk.zero_grad()
                loss.backward()
                self.opt_atk.step()
                loss_total += loss.item()

        for p in self.defensor.parameters():
            p.requires_grad = True

        return loss_total / (
            self.cfg.epochs_por_rodada * len(self.ddos_loader)
        )

    def _gerar_amostras_avaliacao_adversarial(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gera A_r sobre bases/ruído fixos e retorna amostras + perturbação aplicada."""
        self.atacante.eval()
        with torch.no_grad():
            perturbacao = self.cfg.epsilon * self.atacante(self.z_adv_eval)
            amostras = self.X_adv_eval_base + perturbacao
            return amostras, perturbacao

    def _avaliar_adversarial(
        self,
        amostras: torch.Tensor,
    ) -> dict:
        """Mede evasão e confiança média do Defensor sobre DDoS adversarial."""
        self.defensor.eval()
        with torch.no_grad():
            pred = self.defensor(amostras)
            return {
                "evasao": (
                    (pred < self.cfg.classification_threshold)
                    .float()
                    .mean()
                    .item()
                ),
                "confianca_media": pred.mean().item(),
            }

    def _medir_normas_perturbacao(
        self,
        perturbacao: torch.Tensor,
    ) -> dict:
        """Calcula magnitude média da perturbação por amostra."""
        flat = perturbacao.reshape(perturbacao.shape[0], -1)
        return {
            "l1": torch.linalg.vector_norm(flat, ord=1, dim=1).mean().item(),
            "l2": torch.linalg.vector_norm(flat, ord=2, dim=1).mean().item(),
            "linf": torch.linalg.vector_norm(flat, ord=float("inf"), dim=1).mean().item(),
        }

    def _retreinar_defensor(self) -> float:
        """Adapta o Defensor com amostras de treino perturbadas."""
        self.defensor.train()
        self.atacante.eval()

        with torch.no_grad():
            n = min(self.cfg.amostras_por_rodada, len(self.X_ddos_t))
            z = torch.randn(
                n,
                self.cfg.noise_dim,
                device=self.device,
            )

            idx = torch.randperm(
                len(self.X_ddos_t),
                device=self.device,
            )[:n]

            X_ddos_sample = self.X_ddos_t[idx]
            amostras_perturbadas = (
                X_ddos_sample + self.cfg.epsilon * self.atacante(z)
            )

        idx_real = torch.randperm(
            len(self.X_train_t),
            device=self.device,
        )[:n]

        X_real = self.X_train_t[idx_real]
        y_real = self.y_train_t[idx_real]

        X_comb = torch.cat([X_real, amostras_perturbadas])
        y_comb = torch.cat(
            [
                y_real,
                torch.ones(n, device=self.device),
            ]
        )

        loader = DataLoader(
            TensorDataset(X_comb, y_comb),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        loss_total = 0.0

        for _ in range(self.cfg.epochs_por_rodada):
            for X_batch, y_batch in loader:
                self.opt_def.zero_grad()
                pred = self.defensor(X_batch)
                loss = self.criterio(pred, y_batch)
                loss.backward()
                self.opt_def.step()
                loss_total += loss.item()

        return loss_total / (
            self.cfg.epochs_por_rodada * len(loader)
        )

    def _calcular_metricas_classificacao(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> dict:
        """Calcula métricas binárias do Defensor a partir de probabilidades."""
        self.defensor.eval()
        with torch.no_grad():
            probs = self.defensor(X).detach().cpu().numpy().reshape(-1)

        y_true = y.detach().cpu().numpy().astype(int).reshape(-1)
        y_pred = (probs >= self.cfg.classification_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        fpr = fp / max(fp + tn, 1)
        fnr = fn / max(fn + tp, 1)

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "confusion_matrix": [
                [int(tn), int(fp)],
                [int(fn), int(tp)],
            ],
        }


    def avaliar_matriz_checkpoints(self) -> dict:
        """
        Avalia todos os Atacantes A1..An contra todos os Defensores D0..Dn.

        A avaliação reutiliza exatamente o mesmo conjunto adversarial fixo
        utilizado durante as rodadas: mesmas amostras-base de validação,
        mesmos vetores de ruído, mesmo epsilon e mesmo threshold. Nenhum
        modelo é atualizado durante esta etapa.
        """
        checkpoint_dir = self.cfg.checkpoint_dir
        n = self.cfg.n_rodadas

        attacker_paths = [
            checkpoint_dir / f"atacante_rodada_{i:02d}.pth"
            for i in range(1, n + 1)
        ]
        defender_paths = [
            checkpoint_dir / f"defensor_rodada_{i:02d}.pth"
            for i in range(0, n + 1)
        ]

        ausentes = [
            str(path)
            for path in attacker_paths + defender_paths
            if not path.exists()
        ]
        if ausentes:
            raise FileNotFoundError(
                "Checkpoints ausentes para avaliação cruzada: "
                + ", ".join(ausentes)
            )

        # Carrega os estados dos Defensores uma única vez. Os modelos são
        # pequenos e isso evita reler os mesmos 21 arquivos para cada A_i.
        estados_defensores = [
            torch.load(path, map_location=self.device)
            for path in defender_paths
        ]

        atacante_eval = Atacante(
            self.cfg.noise_dim,
            self.cfg.input_dim,
        ).to(self.device)
        defensor_eval = Defensor(
            self.cfg.input_dim
        ).to(self.device)

        atacante_eval.eval()
        defensor_eval.eval()

        matriz: list[list[float]] = []

        logger.info("  Avaliação cruzada de checkpoints: A1..A%d × D0..D%d", n, n)

        with torch.no_grad():
            for atacante_idx, path_atacante in enumerate(attacker_paths, start=1):
                atacante_eval.load_state_dict(
                    torch.load(path_atacante, map_location=self.device)
                )

                perturbacao = (
                    self.cfg.epsilon
                    * atacante_eval(self.z_adv_eval)
                )
                amostras = self.X_adv_eval_base + perturbacao

                linha: list[float] = []
                for estado_defensor in estados_defensores:
                    defensor_eval.load_state_dict(estado_defensor)
                    probs = defensor_eval(amostras)
                    evasao = (
                        (probs < self.cfg.classification_threshold)
                        .float()
                        .mean()
                        .item()
                    )
                    linha.append(float(evasao))

                matriz.append(linha)
                logger.debug(
                    "Matriz checkpoints A%d: %s",
                    atacante_idx,
                    [round(v, 4) for v in linha],
                )

        # Robustez do Defensor final contra todos os ataques históricos.
        evasao_contra_defensor_final = [
            linha[n] for linha in matriz
        ]

        # Para cada D_j (j>=1), mede a evasão média dos ataques que ele já
        # deveria ter enfrentado: A1..A_j. Queda dessa qualidade ao longo do
        # tempo é um sinal compatível com catastrophic forgetting.
        evasao_media_ataques_vistos: list[float] = []
        pior_evasao_ataques_vistos: list[float] = []
        for defensor_idx in range(1, n + 1):
            valores = [
                matriz[atacante_idx][defensor_idx]
                for atacante_idx in range(defensor_idx)
            ]
            evasao_media_ataques_vistos.append(
                float(np.mean(valores))
            )
            pior_evasao_ataques_vistos.append(
                float(np.max(valores))
            )

        resultado = {
            "linhas_atacantes": list(range(1, n + 1)),
            "colunas_defensores": list(range(0, n + 1)),
            "matriz_evasao": matriz,
            "evasao_contra_defensor_final": evasao_contra_defensor_final,
            "evasao_media_ataques_vistos_por_defensor": evasao_media_ataques_vistos,
            "pior_evasao_ataques_vistos_por_defensor": pior_evasao_ataques_vistos,
        }

        media_final = float(np.mean(evasao_contra_defensor_final))
        pior_final = float(np.max(evasao_contra_defensor_final))
        pior_atacante = int(np.argmax(evasao_contra_defensor_final)) + 1

        logger.info("    D%d contra ataques históricos:", n)
        logger.info("      Evasão média: %.2f%%", media_final * 100)
        logger.info(
            "      Pior caso: A%d → %.2f%%",
            pior_atacante,
            pior_final * 100,
        )

        # Mostra apenas ataques com evasão residual, em linhas curtas e
        # alinhadas, sem despejar a matriz completa no terminal.
        resistentes = [
            (i + 1, valor)
            for i, valor in enumerate(evasao_contra_defensor_final)
            if valor > 0.0
        ]
        if resistentes:
            logger.info("      Evasões residuais:")
            for i, valor in resistentes:
                logger.info(
                    "        A%02d × D%d: %6.2f%%",
                    i,
                    n,
                    valor * 100,
                )
        else:
            logger.info("      Evasões residuais: nenhuma")

        return resultado

    def medir_metricas_teste_final(self) -> dict:
        """Avalia o Defensor final no teste reservado, uma única vez."""
        return self._calcular_metricas_classificacao(
            self.X_test_t,
            self.y_test_t,
        )

    # ── Checkpoints ───────────────────────────────────────────────────────

    def _salvar_checkpoint(self, rodada: int) -> None:
        self.cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.atacante.state_dict(),
            self.cfg.checkpoint_dir / f"atacante_rodada_{rodada:02d}.pth",
        )
        torch.save(
            self.defensor.state_dict(),
            self.cfg.checkpoint_dir / f"defensor_rodada_{rodada:02d}.pth",
        )

    def carregar_checkpoint(
        self,
        path_atacante: Path,
        path_defensor: Optional[Path] = None,
    ) -> None:
        self.atacante.load_state_dict(
            torch.load(path_atacante, map_location=self.device)
        )

        if path_defensor:
            self.defensor.load_state_dict(
                torch.load(path_defensor, map_location=self.device)
            )

        logger.info("Checkpoint carregado: %s", path_atacante)
