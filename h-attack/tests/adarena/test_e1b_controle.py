"""Testes do controle pareado offline do E1-B, sem depender do dataset completo."""
from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCRIPT = Path(__file__).resolve().parents[3] / "docs/scientific/scripts/e1b_controle.py"
spec = importlib.util.spec_from_file_location("e1b_controle_testmod", SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class TinyDefender(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Dropout(0.3), nn.Linear(8, 1), nn.Sigmoid())

    def forward(self, x):
        return self.net(x).squeeze(1)


class ToyTrainer:
    """Atualização adversarial simulada com epsilon 0, para testar paridade."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.cfg = SimpleNamespace(
            seed=42, noise_dim=4, amostras_por_rodada=12,
            epochs_por_rodada=2, batch_size=8, classification_threshold=0.5,
        )
        self.defensor = TinyDefender()
        self.opt_def = torch.optim.Adam(self.defensor.parameters(), lr=1e-3)
        self.criterio = nn.BCELoss()
        X = torch.randn(60, 3)
        y = torch.cat((torch.zeros(20), torch.ones(40)))
        self.X_train_t, self.y_train_t = X, y
        self.X_ddos_t = X[y == 1]
        self.X_val_t = torch.randn(25, 3)
        self.y_val_t = torch.cat((torch.zeros(12), torch.ones(13)))
        # Preencher Adam com momentum para testar a cópia de estado real.
        self.defensor.train()
        self.opt_def.zero_grad()
        loss = self.criterio(self.defensor(X[:20]), y[:20])
        loss.backward()
        self.opt_def.step()

    def _retreinar_defensor(self):
        n = min(self.cfg.amostras_por_rodada, len(self.X_ddos_t))
        self.defensor.train()
        with torch.no_grad():
            _ = torch.randn(n, self.cfg.noise_dim, device=self.device)
            idx = torch.randperm(len(self.X_ddos_t), device=self.device)[:n]
            X_ddos = self.X_ddos_t[idx]  # Epsilon=0, duas condições idênticas
        idx_real = torch.randperm(len(self.X_train_t), device=self.device)[:n]
        X_real, y_real = self.X_train_t[idx_real], self.y_train_t[idx_real]
        X_comb = torch.cat([X_real, X_ddos])
        y_comb = torch.cat([y_real, torch.ones(n)])
        loader = DataLoader(TensorDataset(X_comb, y_comb), batch_size=self.cfg.batch_size, shuffle=True)
        losses = []
        for _ in range(self.cfg.epochs_por_rodada):
            for X_batch, y_batch in loader:
                self.opt_def.zero_grad()
                loss = self.criterio(self.defensor(X_batch), y_batch)
                loss.backward()
                self.opt_def.step()
                losses.append(loss.item())
        return sum(losses) / len(losses)


def test_rng_roundtrip():
    random.seed(44)
    np.random.seed(44)
    torch.manual_seed(44)
    before = mod.snapshot_rng()
    sample1 = (random.random(), np.random.random(), torch.rand(4))
    mod.restore_rng(before)
    sample2 = (random.random(), np.random.random(), torch.rand(4))
    assert sample1[0] == sample2[0]
    assert sample1[1] == sample2[1]
    assert torch.equal(sample1[2], sample2[2])


def test_bifurcacao_pesos_adam_e_dropout_pareados():
    torch.manual_seed(42)
    trainer = ToyTrainer()
    initial_state = mod.snapshot_rng()
    X_adv_val = trainer.X_val_t[trainer.y_val_t == 1]
    row = mod.comparar_rodada(trainer, 3, X_adv_val)
    assert row["n_real"] == 12
    assert row["passos_por_braco"] == 6  # ceil(24/8) * 2 epochs
    assert row["beneficio_adversarial_pp"] == pytest.approx(0.0, abs=1e-6)
    assert row["divergencia_parametros_max_abs"] == pytest.approx(0.0, abs=1e-8)
    assert row["fpr_adversarial"] == pytest.approx(row["fpr_convencional"], abs=1e-8)
    assert row["f1_adversarial"] == pytest.approx(row["f1_convencional"], abs=1e-8)
    assert row["loss_adversarial"] == pytest.approx(row["loss_convencional"], abs=1e-8)
    # A segunda execução não compartilhou pesos com a original nem alterou o RNG
    assert not torch.equal(initial_state.torch_cpu, mod.snapshot_rng().torch_cpu)


def test_diferencas_iniciais_sao_detectadas():
    with pytest.raises(AssertionError):
        mod.assert_same_structure({"a": torch.tensor([1])}, {"a": torch.tensor([2])})
