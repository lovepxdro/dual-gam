"""
Dual-GAM — Loop de treinamento adversarial

Adaptado da seção de perturbação adversarial do experimento inicial,
que foi a única abordagem que funcionou (75.1% de evasão na rodada 1).
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
from torch.utils.data import DataLoader, TensorDataset

from .models import Atacante, Defensor

logger = logging.getLogger(__name__)


def definir_seed(seed: int) -> None:
    """
    Configura as principais fontes de aleatoriedade da execução.

    Isso permite reproduzir:
    - inicialização dos modelos;
    - ordem dos batches;
    - geração de ruído;
    - seleção aleatória de amostras.
    """

    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Solicita operações determinísticas sempre que possível.
    torch.use_deterministic_algorithms(True, warn_only=True)

    logger.info("Seed definida: %d", seed)


@dataclass
class TrainingConfig:
    # Reprodutibilidade
    seed: int = 42

    # Dimensões
    input_dim: int = 77
    noise_dim: int = 32

    # Hiperparâmetros
    lr_defensor: float = 1e-3
    lr_atacante: float = 2e-4
    adam_betas: tuple = (0.5, 0.999)
    epsilon: float = 0.3

    epochs_pretrain: int = 5
    epochs_por_rodada: int = 3
    n_rodadas: int = 20

    amostras_por_rodada: int = 5000
    batch_size: int = 512

    # Dispositivo
    device: str = "cpu"

    # Onde salvar checkpoints
    checkpoint_dir: Path = Path("/models")

    historico: dict = field(
        default_factory=lambda: {
            "taxa_evasao": [],
            "acuracia_defensor": [],
            "loss_atacante": [],
            "loss_defensor": [],
        }
    )


class AdversarialTrainer:
    """
    Orquestra o ciclo co-evolutivo Atacante ↔ Defensor.

    Fluxo de cada rodada:
      1. Atacante aprende perturbações que enganam o Defensor atual
      2. Defensor retreina com as amostras perturbadas
      3. Medir taxa de evasão e acurácia
    """

    def __init__(self, config: TrainingConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        self.defensor = Defensor(
            config.input_dim
        ).to(self.device)

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
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        """Converte arrays NumPy para tensores e cria DataLoaders."""

        self.X_train_t = torch.FloatTensor(
            X_train
        ).to(self.device)

        self.y_train_t = torch.FloatTensor(
            y_train
        ).to(self.device)

        self.X_test_t = torch.FloatTensor(
            X_test
        ).to(self.device)

        self.y_test_t = torch.FloatTensor(
            y_test
        ).to(self.device)

        # Loader geral: benigno + DDoS.
        self.train_loader = DataLoader(
            TensorDataset(
                self.X_train_t,
                self.y_train_t,
            ),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        # Loader apenas DDoS.
        mask_ddos = y_train == 1

        self.X_ddos_t = torch.FloatTensor(
            X_train[mask_ddos]
        ).to(self.device)

        self.ddos_loader = DataLoader(
            TensorDataset(self.X_ddos_t),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        logger.info(
            "Dados carregados — treino: %d | DDoS: %d | teste: %d",
            len(X_train),
            mask_ddos.sum(),
            len(X_test),
        )

    # ── Pré-treino ────────────────────────────────────────────────────────

    def pretreinar_defensor(self) -> None:
        """Treina o Defensor com dados reais antes do ciclo adversarial."""

        logger.info(
            "Pré-treinando Defensor por %d epochs...",
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

        acc = self._medir_acuracia()

        logger.info(
            "Defensor pré-treinado — acurácia: %.2f%%",
            acc * 100,
        )

    # ── Ciclo adversarial ─────────────────────────────────────────────────

    def rodar_ciclo(self) -> dict:
        """
        Executa as rodadas de co-evolução Atacante ↔ Defensor.

        Retorna o histórico completo.
        """

        logger.info(
            "Iniciando ciclo adversarial (%d rodadas)",
            self.cfg.n_rodadas,
        )

        for rodada in range(self.cfg.n_rodadas):
            loss_atk, taxa_evasao = (
                self._treinar_atacante()
            )

            loss_def = self._retreinar_defensor()
            acc = self._medir_acuracia()

            self.cfg.historico[
                "taxa_evasao"
            ].append(taxa_evasao)

            self.cfg.historico[
                "acuracia_defensor"
            ].append(acc)

            self.cfg.historico[
                "loss_atacante"
            ].append(loss_atk)

            self.cfg.historico[
                "loss_defensor"
            ].append(loss_def)

            logger.info(
                "Rodada %2d/%d | Evasão: %5.1f%% | "
                "Acc Defensor: %.2f%%",
                rodada + 1,
                self.cfg.n_rodadas,
                taxa_evasao * 100,
                acc * 100,
            )

            self._salvar_checkpoint(
                rodada + 1
            )

        return self.cfg.historico

    def _treinar_atacante(
        self,
    ) -> tuple[float, float]:
        """
        Fase 1: Atacante aprende a perturbar DDoS
        para enganar o Defensor.
        """

        # Congelar Defensor.
        for p in self.defensor.parameters():
            p.requires_grad = False

        self.atacante.train()
        self.defensor.eval()

        loss_total = 0.0
        evasoes = 0
        total = 0

        for epoch in range(
            self.cfg.epochs_por_rodada
        ):
            for (X_ddos_batch,) in self.ddos_loader:
                batch_size = X_ddos_batch.shape[0]

                z = torch.randn(
                    batch_size,
                    self.cfg.noise_dim,
                    device=self.device,
                )

                perturbacao = self.atacante(z)

                amostras = (
                    X_ddos_batch
                    + self.cfg.epsilon
                    * perturbacao
                )

                predicoes = self.defensor(
                    amostras
                )

                alvo = torch.zeros(
                    batch_size,
                    device=self.device,
                )

                loss = self.criterio(
                    predicoes,
                    alvo,
                )

                self.opt_atk.zero_grad()
                loss.backward()
                self.opt_atk.step()

                loss_total += loss.item()

                if (
                    epoch
                    == self.cfg.epochs_por_rodada - 1
                ):
                    evasoes += (
                        predicoes < 0.5
                    ).sum().item()

                    total += batch_size

        taxa_evasao = (
            evasoes / max(total, 1)
        )

        loss_media = (
            loss_total
            / (
                self.cfg.epochs_por_rodada
                * len(self.ddos_loader)
            )
        )

        # Descongelar Defensor.
        for p in self.defensor.parameters():
            p.requires_grad = True

        return loss_media, taxa_evasao

    def _retreinar_defensor(self) -> float:
        """
        Fase 2: Defensor aprende a detectar
        as amostras perturbadas.
        """

        self.defensor.train()
        self.atacante.eval()

        # Gerar amostras perturbadas.
        with torch.no_grad():
            n = self.cfg.amostras_por_rodada

            z = torch.randn(
                n,
                self.cfg.noise_dim,
                device=self.device,
            )

            idx = torch.randperm(
                len(self.X_ddos_t),
                device=self.device,
            )[:n]

            X_ddos_sample = (
                self.X_ddos_t[idx]
            )

            amostras_perturbadas = (
                X_ddos_sample
                + self.cfg.epsilon
                * self.atacante(z)
            )

        # Amostra de dados reais.
        idx_real = torch.randperm(
            len(self.X_train_t),
            device=self.device,
        )[:n]

        X_real = self.X_train_t[
            idx_real
        ]

        y_real = self.y_train_t[
            idx_real
        ]

        # Combinar dados reais + perturbados.
        X_comb = torch.cat(
            [
                X_real,
                amostras_perturbadas,
            ]
        )

        y_comb = torch.cat(
            [
                y_real,
                torch.ones(
                    n,
                    device=self.device,
                ),
            ]
        )

        loader = DataLoader(
            TensorDataset(
                X_comb,
                y_comb,
            ),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        loss_total = 0.0

        for epoch in range(
            self.cfg.epochs_por_rodada
        ):
            for X_batch, y_batch in loader:
                self.opt_def.zero_grad()

                pred = self.defensor(
                    X_batch
                )

                loss = self.criterio(
                    pred,
                    y_batch,
                )

                loss.backward()
                self.opt_def.step()

                loss_total += loss.item()

        return (
            loss_total
            / (
                self.cfg.epochs_por_rodada
                * len(loader)
            )
        )

    def _medir_acuracia(self) -> float:
        """Avalia o Defensor no conjunto de teste."""

        self.defensor.eval()

        with torch.no_grad():
            pred = self.defensor(
                self.X_test_t
            )

            return (
                (
                    (pred >= 0.5).float()
                    == self.y_test_t
                )
                .float()
                .mean()
                .item()
            )

    # ── Checkpoints ───────────────────────────────────────────────────────

    def _salvar_checkpoint(
        self,
        rodada: int,
    ) -> None:
        """Salva Atacante e Defensor da rodada atual."""

        self.cfg.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.atacante.state_dict(),
            self.cfg.checkpoint_dir
            / f"atacante_rodada_{rodada:02d}.pth",
        )

        torch.save(
            self.defensor.state_dict(),
            self.cfg.checkpoint_dir
            / f"defensor_rodada_{rodada:02d}.pth",
        )

    def carregar_checkpoint(
        self,
        path_atacante: Path,
        path_defensor: Optional[Path] = None,
    ) -> None:
        """Carrega checkpoints previamente salvos."""

        self.atacante.load_state_dict(
            torch.load(
                path_atacante,
                map_location=self.device,
            )
        )

        if path_defensor:
            self.defensor.load_state_dict(
                torch.load(
                    path_defensor,
                    map_location=self.device,
                )
            )

        logger.info(
            "Checkpoint carregado: %s",
            path_atacante,
        )
