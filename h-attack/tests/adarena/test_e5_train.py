"""Testes isolados do protocolo E5-A; não modificam checkpoints E0."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'docs' / 'scientific' / 'scripts'))
# O diretório 3 pais de tests/adarena/test_x é a raiz do projeto.
from e5_train import (deduplicate_indices, evaluate, init_attacker, select_ddos,
                      split_indices, stable_seed, train_one)
from gan.models import Atacante, Defensor


class SimpleDefender(nn.Module):
    def forward(self, x):
        return torch.sigmoid(10 * x[:, 0])


class FirstFeatureDelta(nn.Module):
    def __init__(self, magnitude: float):
        super().__init__()
        self.magnitude = magnitude
    def forward(self, z):
        out = torch.zeros((len(z), 3))
        out[:, 0] = self.magnitude
        return out


def test_split_matches_train_test_split_and_isolation():
    y = np.tile(np.array([0, 1]), 40)
    a, b, c = split_indices(y, 42, .2, .1)
    assert (len(a), len(b), len(c)) == (56, 8, 16)
    assert not (set(a) & set(b) or set(a) & set(c) or set(b) & set(c))
    assert set(np.unique(y[a])) == {0, 1}
    assert np.array_equal(c, split_indices(y, 42, .2, .1)[2])


def test_dedup_features_and_label_uses_keep_first():
    x = np.asarray([[1, 2], [1, 2], [1, 2], [3, 4]], dtype=np.float64)
    y = np.asarray([0, 0, 1, 1])
    assert deduplicate_indices(x, y).tolist() == [0, 2, 3]
    assert np.array_equal(~pd.DataFrame({'a':[1,1,1,3], 'b':[2,2,2,4], 'y':y}).duplicated(keep='first').to_numpy(), np.array([True,False,True,True]))


def test_select_ddos_is_seeded_and_never_selects_benign():
    ids = np.arange(60)
    y = np.tile([0, 1], 30)
    a = select_ddos(ids, y, 7, 101)
    assert len(a) == 7 and np.all(y[a] == 1)
    assert np.array_equal(a, select_ddos(ids, y, 7, 101))


def test_asr_conditions_on_clean_detection_and_independent_evasion():
    x = np.asarray([[.1, 0, 0], [-.1, 0, 0]], dtype=np.float32)
    m = FirstFeatureDelta(+1.0)
    result = evaluate(m, 'A0', SimpleDefender(), x, noise_dim=2, noise_seed=123,
                      epsilon=.3, batch_size=2)
    assert result['n'] == 2
    assert result['n_detectados_limpos'] == 1
    assert result['n_sucesso'] == 0
    assert result['asr_condicional'] == 0.0
    assert result['evasao_nao_condicional'] == 0.0  # um FN limpo foi corrigido pelo delta
    neg = evaluate(FirstFeatureDelta(-1.0), 'A0', SimpleDefender(), x,
                   noise_dim=2, noise_seed=123, epsilon=.3, batch_size=2)
    assert neg['asr_condicional'] == 1.0
    assert neg['evasao_nao_condicional'] == 1.0


def test_transfer_identical_scalers_identical_to_local():
    rng = np.random.default_rng(88)
    raw = rng.normal(size=(16, 3))
    raw[:, 0] += 3.0
    sc = StandardScaler().fit(raw)
    xs = sc.transform(raw).astype(np.float32)
    model = FirstFeatureDelta(-.5)
    d = SimpleDefender()
    a = evaluate(model, 'A0', d, xs, noise_dim=2, noise_seed=53, batch_size=6)
    b = evaluate(model, 'A0', d, xs, noise_dim=2, noise_seed=53, batch_size=6,
                 x_raw=raw, source_scaler=sc, target_scaler=sc)
    assert a['n_detectados_limpos'] == b['n_detectados_limpos']
    assert a['n_sucesso'] == b['n_sucesso']
    assert a['diversidade_l2_mesma_amostra'] == b['diversidade_l2_mesma_amostra']


def test_different_scaler_transfer_preserves_physical_delta():
    raw = np.array([[2., 5., 1.], [3., 4., 2.], [4., 3., 3.], [5., 2., 4.]])
    s1 = StandardScaler().fit(raw)
    s2 = StandardScaler().fit(raw * 2 + 2)
    xs = s1.transform(raw).astype(np.float32)
    result = evaluate(FirstFeatureDelta(-.2), 'A0', SimpleDefender(), xs,
                      noise_dim=2, noise_seed=5, x_raw=raw, source_scaler=s1,
                      target_scaler=s2)
    assert result['n'] == 4
    assert result['delta_linf_media'] <= .3 + 1e-6


def test_initial_models_match_checkpoint_and_d20_stays_frozen():
    a20 = Atacante(noise_dim=2, output_dim=3).eval()
    state = a20.state_dict()
    a0 = init_attacker('A0', state, input_dim=3, noise_dim=2, init_seed=123)
    a1 = init_attacker('A1', state, input_dim=3, noise_dim=2, init_seed=123)
    a0.eval(); a1.eval()
    torch.manual_seed(123)
    z, x = torch.randn(8, 2), torch.randn(8, 3)
    with torch.no_grad():
        assert torch.equal(a0(z), a1(x, z))
        assert torch.equal(a1(x,z), a1(x + .2,z))
    d = SimpleDefender()
    src = torch.randn(19, 3)
    train_one(a0, 'A0', d, src, src[:8].numpy(), seed_e0=20, rep_seed=1042,
              epochs=1, batch_size=8, noise_dim=2, epsilon=.3, lr=.0002,
              betas=(.5, .999), eval_batch=8)
    assert all(p.grad is None for p in d.parameters())


def test_train_a1_conditional_runs_on_small_batch():
    state = Atacante(noise_dim=2, output_dim=3).state_dict()
    a = init_attacker('A1', state, input_dim=3, noise_dim=2, init_seed=42)
    d = SimpleDefender()
    x = torch.randn(18, 3)
    curve, _ = train_one(a, 'A1', d, x, x[:6].numpy(), seed_e0=42, rep_seed=1042,
                         epochs=1, batch_size=8, noise_dim=2, epsilon=.3,
                         lr=.0002, betas=(.5,.999), eval_batch=6)
    assert len(curve) == 1 and np.isfinite(curve[0]['loss_media'])
    assert a.ramo[-1].weight.grad is not None


def test_stable_seed_different_namespaces():
    assert stable_seed('e5-val', 42, 1042) == stable_seed('e5-val', 42, 1042)
    assert stable_seed('e5-val', 42, 1042) != stable_seed('e5-test', 42, 1042)


def test_end_to_end_small_paired_run_and_resume(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import e5_train as module
    from gan.models import Defensor
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(400, 77))
    y = np.tile(np.asarray([0, 1]), 200)
    state_a = Atacante(noise_dim=32, output_dim=77).state_dict()
    run_info = {}
    for seed in (20, 42):
        a, b, c = split_indices(y, seed, .2, .1)
        scaler = StandardScaler().fit(x[a])
        run_info[seed] = module.Run(seed, tmp_path, {
            'treinamento': {'lr_atacante': .0002, 'adam_betas_atacante': [.5, .999]}
        }, SimpleNamespace(scaler=scaler), a, b, c, state_a,
            Defensor(input_dim=77).eval().requires_grad_(False))
    monkeypatch.setattr(module, 'prepare_runs', lambda *args: (x, y, run_info, 0, 'synthetic.parquet'))
    preflight = tmp_path / 'e5_preflight.json'
    preflight.write_text('{"status":"aprovado"}', encoding='utf8')
    args = SimpleNamespace(runs_root=tmp_path, preflight=preflight, seeds=[20,42],
                           replicas=[1042], epochs=1, batch_size=16,
                           max_train_samples=60, val_samples=10, test_samples=12,
                           eval_batch=16, device='cpu', output_dir=tmp_path/'out', resume=False)
    module.run(args)
    summary = json.loads((args.output_dir/'resumo_e5a.json').read_text())
    assert summary['n_treinamentos'] == 4
    assert all(r['n_teste_comum_ddos'] == summary['n_teste_comum_ddos'] for r in summary['resultados'])
    assert all(r['n_treino_ddos'] == 60 for r in summary['resultados'])
    assert (args.output_dir/'seed_20'/'rep_1042'/'A1'/'atacante_final.pth').exists()
    args.resume = True
    module.run(args)  # Deve pular os braços completos sem treinar de novo.
    assert json.loads((args.output_dir/'resumo_e5a.json').read_text())['n_treinamentos'] == 4
    args.resume = False
    with pytest.raises(FileExistsError):
        module.run(args)
