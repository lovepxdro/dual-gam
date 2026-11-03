"""Testes rápidos do protocolo E2-B (sem CIC-IDS2017)."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

SCRIPT = Path(__file__).resolve().parents[3] / 'docs/scientific/scripts/e2b_prospectivo.py'
spec = importlib.util.spec_from_file_location('e2b_experiment_script', SCRIPT)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


class TinyDefender(nn.Module):
    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(nn.Linear(3, 9), nn.ReLU(), nn.Linear(9, 1), nn.Sigmoid())

    def forward(self, x):
        return self.rede(x).squeeze(1)


class TinyAttacker(nn.Module):
    def __init__(self):
        super().__init__()
        self.rede = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3), nn.Tanh())

    def forward(self, z):
        return self.rede(z)


def test_montar_avaliacao_seleciona_somente_ddos_e_e_reproduzivel():
    X = np.arange(120, dtype=np.float32).reshape(-1, 3)
    y = np.array([0] * 20 + [1] * 20)
    a, z1 = mod.montar_avaliacao(X, y, 42, 8, 4)
    b, z2 = mod.montar_avaliacao(X, y, 42, 8, 4)
    assert torch.equal(a, b) and torch.equal(z1, z2)
    assert a.shape == (8, 3)
    assert all(float(v) >= 60 for v in a[:, 0])


def test_medir_evasao_nao_altera_modelos():
    defender = TinyDefender().eval()
    attacker = TinyAttacker().eval()
    x = torch.randn(21, 3)
    z = torch.randn(21, 4)
    before = {k: v.clone() for k, v in defender.state_dict().items()}
    out = mod.medir_evasao(defender, x, z, attacker, .3, .5, batch_size=7)
    assert 0 <= out['evasao'] <= 1
    assert out['perturbacao_linf_media'] <= .30001
    assert all(torch.equal(v, defender.state_dict()[k]) for k, v in before.items())


def test_treino_determinista_e_defensor_congelado():
    torch.manual_seed(7)
    defender = TinyDefender()
    attacker = TinyAttacker()
    X_ddos = torch.randn(30, 3)
    X_eval = torch.randn(12, 3)
    z_eval = torch.randn(12, 4)
    before = {k: v.clone() for k, v in defender.state_dict().items()}
    kwargs = dict(seed=101, epochs=2, batch_size=8, noise_dim=4, epsilon=.3,
                  lr=.001, betas=(.5, .999), avaliacoes={'D20': defender},
                  x_eval=X_eval, z_eval=z_eval, threshold=.5)
    a, curva_a = mod.treinar_atacante(defender, X_ddos, attacker, **kwargs)
    b, curva_b = mod.treinar_atacante(defender, X_ddos, attacker, **kwargs)
    assert len(curva_a) == len(curva_b) == 2
    assert curva_a == curva_b
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    assert all(torch.equal(v, defender.state_dict()[k]) for k, v in before.items())


def test_validacao_baseline_detecta_diferencas(tmp_path):
    path = tmp_path / 'matriz_checkpoints.json'
    matriz = [[0.0] * 21 for _ in range(20)]
    matriz[19][19] = .1
    matriz[19][20] = .2
    path.write_text(json.dumps({'matriz_evasao': matriz}))
    mod.validar_baseline_historico(path, .1, .2)
    with pytest.raises(AssertionError, match='Baseline A20'):
        mod.validar_baseline_historico(path, .1, .201)


def test_inicializacao_a20_copia_pesos_sem_compartilhar_estado():
    torch.manual_seed(123)
    random_attacker = TinyAttacker()
    a20 = TinyAttacker()
    original = {k: v.clone() for k, v in a20.state_dict().items()}
    inicial = mod.selecionar_atacante_inicial("a20", random_attacker, a20)
    assert all(torch.equal(inicial.state_dict()[k], v) for k, v in original.items())
    with torch.no_grad():
        next(inicial.parameters()).add_(1.0)
    assert all(torch.equal(a20.state_dict()[k], v) for k, v in original.items())
    rand = mod.selecionar_atacante_inicial("random", random_attacker, a20)
    assert all(torch.equal(rand.state_dict()[k], v) for k, v in random_attacker.state_dict().items())
    with pytest.raises(ValueError, match="Inicialização desconhecida"):
        mod.selecionar_atacante_inicial("invalid", random_attacker, a20)
