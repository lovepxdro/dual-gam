"""E5-A3: ablação de capacidade sem entrada x; não altera E0/E5-A2."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'docs' / 'scientific' / 'scripts'))

import e5a3_capacity as runner
from e5a3_capacity import checked_baseline, expected_config
from e5a3_compare import compare_dirs
from e5_train import Run, select_ddos, sha256_indices, split_indices, stable_seed, train_one
from gan.e5_capacity import AtacanteCapacidade
from gan.e5_conditional import AtacanteCondicional
from gan.models import Atacante, Defensor


def test_parametros_quase_identicos_a_a1_e_inicializacao_exata():
    torch.manual_seed(11)
    a0 = Atacante(noise_dim=32, output_dim=77).eval()
    a1 = AtacanteCondicional(noise_dim=32, input_dim=77).eval()
    a2 = AtacanteCapacidade(noise_dim=32, input_dim=77).eval()
    a1.carregar_base_a20(a0.state_dict())
    a2.carregar_base_a20(a0.state_dict())
    assert sum(p.numel() for p in a0.parameters()) == 64013
    assert sum(p.numel() for p in a1.parameters()) == 104538
    assert sum(p.numel() for p in a2.parameters()) == 104482
    assert abs(104538 - 104482) / 104538 < .001
    x, z = torch.randn(16, 77), torch.randn(16, 32)
    with torch.no_grad():
        assert torch.equal(a2(x, z), a0(z))
        assert torch.equal(a2(x + 5, z), a2(x, z))


def test_invariancia_a_x_apos_atualizacao_real_e_limite_linf():
    torch.set_num_threads(1)
    model = AtacanteCapacidade(noise_dim=3, input_dim=4, hidden_dim=12)
    model.eval()
    optim = torch.optim.Adam(model.parameters(), lr=.001)
    x = torch.randn(20, 4)
    z = torch.randn(20, 3)
    for _ in range(3):
        loss = model(x, z).square().sum()
        optim.zero_grad()
        loss.backward()
        optim.step()
    with torch.no_grad():
        a = model(x, z)
        b = model(torch.full_like(x, 99), z)
        assert torch.equal(a, b)
        assert a.abs().max() <= 1.000001
        assert (x + .3 * a - x).abs().max() <= .300001
    assert model.ramo[0].weight.grad is not None
    assert model.ramo[0].weight.grad.abs().sum() > 0


def test_logit_training_com_a2_e_defender_congelado():
    torch.set_num_threads(1)
    a = AtacanteCapacidade(noise_dim=2, input_dim=3, hidden_dim=9)
    d = Defensor(input_dim=3).eval().requires_grad_(False)
    before = {k: v.clone() for k, v in d.state_dict().items()}
    X = torch.randn(32, 3)
    hist, _ = train_one(a, 'A2', d, X, X[:8].numpy(), seed_e0=20, rep_seed=1042,
                        epochs=1, batch_size=8, noise_dim=2, epsilon=.3,
                        lr=.0002, betas=(.5, .999), eval_batch=8, loss_mode='logits')
    assert len(hist) == 1
    assert np.isfinite(hist[0]['loss_media'])
    assert hist[0]['grad_ramo_l2_media'] is not None
    assert all(torch.equal(v, d.state_dict()[k]) for k, v in before.items())


def test_rejeita_shapemismatch():
    model = AtacanteCapacidade(noise_dim=2, input_dim=3)
    with pytest.raises(ValueError, match='Shapes'):
        model(torch.zeros(3, 3), torch.zeros(3, 5))


def test_end_to_end_sintetico_e_comparador(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    X = rng.normal(size=(300, 77))
    y = np.tile(np.asarray([0, 1]), 150)
    state_a = Atacante(noise_dim=32, output_dim=77).state_dict()
    runs = {}
    for seed in (20, 42):
        train, val, test = split_indices(y, seed, .2, .1)
        sc = StandardScaler().fit(X[train])
        d = Defensor(input_dim=77)
        with torch.no_grad():
            d.rede[-2].bias.fill_(5.0)  # todas as amostras limpas serão detectadas no teste sintético
        runs[seed] = Run(seed, tmp_path,
                         {'treinamento': {'lr_atacante': .0002, 'adam_betas_atacante': [.5, .999]}},
                         SimpleNamespace(scaler=sc), train, val, test, state_a,
                         d.eval().requires_grad_(False))
    monkeypatch.setattr(runner, 'prepare_runs', lambda *args: (X, y, runs, 0, 'synthetic.parquet'))
    monkeypatch.setattr(runner, 'check_preflight', lambda *args: {})
    preflight = tmp_path / 'preflight.json'
    preflight.write_text('{"status":"aprovado"}', encoding='utf-8')
    baseline_dir = tmp_path / 'e5a2'
    baseline_dir.mkdir()
    args = SimpleNamespace(
        runs_root=tmp_path, preflight=preflight, baseline_dir=baseline_dir,
        seeds=[20, 42], replicas=[1042], epochs=1, batch_size=16,
        max_train_samples=32, val_samples=6, test_samples=8,
        eval_batch=8, device='cpu', output_dir=tmp_path / 'e5a3', resume=False)
    config_path = baseline_dir / 'config_e5a2.json'
    config_path.write_text('{}', encoding='utf-8')
    config = expected_config(args)
    baseline_config = {k: config[k] for k in runner.COMPARABLE}
    baseline_config.update({'experiment': 'E5-A2', 'loss_mode': 'logits'})
    config_path.write_text(json.dumps(baseline_config), encoding='utf-8')
    common = np.intersect1d(runs[20].test_ids, runs[42].test_ids, assume_unique=True)
    selected = select_ddos(common, y, 8, stable_seed('e5-shared-test', 20, 42))
    shared_sha = sha256_indices(selected)
    rows = []
    for seed in (20, 42):
        other = 42 if seed == 20 else 20
        n_train = 32
        val_ids = select_ddos(runs[seed].val_ids, y, 6, stable_seed('e5-val-selection', seed))
        for arm in ('A0', 'A1'):
            rows.append({
                'seed_e0': seed, 'replica': 1042, 'braco': arm,
                'n_treino_ddos': n_train, 'n_teste_comum_ddos': len(selected),
                'seed_defender_transferencia': other,
                'asr_alvo': .01 if arm == 'A0' else .03,
                'asr_transferencia': .00, 'diversidade_l2_alvo': 1.0,
                'tempo_treino_segundos': 1.0,
            })
            folder = baseline_dir / f'seed_{seed}' / 'rep_1042' / arm
            folder.mkdir(parents=True)
            (folder/'resultado.json').write_text(json.dumps({
                'n_treino_ddos': n_train, 'n_validacao_ddos': len(val_ids),
                'ids_teste_comum_sha256': shared_sha,
                'noise_seed_teste': stable_seed('e5-test', 20, 42, 1042),
                'defender_transferencia': other,
                'optimizer': {'lr': .0002, 'betas': [.5, .999]},
                'n_parametros': 64013 if arm == 'A0' else 104538,
                'loss_mode': 'logits', 'epochs': 1, 'batch_size': 16, 'epsilon': .3,
            }), encoding='utf-8')
    (baseline_dir/'resumo_e5a.json').write_text(json.dumps({
        'experiment': 'E5-A2', 'loss_mode': 'logits',
        'seeds': [20, 42], 'replicas': [1042],
        'ids_teste_comum_sha256': shared_sha,
        'n_teste_comum_ddos': len(selected), 'resultados': rows,
    }), encoding='utf-8')
    summary = runner.execute(args)
    assert summary['n_treinamentos'] == 2
    assert summary['parametros_a2'] == 104482
    result = compare_dirs(baseline_dir, args.output_dir, tmp_path / 'comparacao')
    assert result['comparabilidade'] == 'aprovada'
    assert result['n_pares_seed_replica'] == 2
    assert (tmp_path/'comparacao'/'comparacao_e5a3_pareada.csv').is_file()
    args.resume = True
    summary_second = runner.execute(args)
    assert summary_second['n_treinamentos'] == 2
    # Alterar x deve continuar sem afetar A2 no estado treinado.
    checkpoint = args.output_dir/'seed_20'/'rep_1042'/'A2'/'atacante_final.pth'
    model = AtacanteCapacidade().eval()
    model.load_state_dict(torch.load(checkpoint, weights_only=True))
    z, x = torch.randn(5, 32), torch.randn(5, 77)
    with torch.no_grad():
        assert torch.equal(model(x, z), model(x + 100.0, z))
    with pytest.raises(ValueError, match='configurações diferentes'):
        bad = dict(baseline_config, epochs=99)
        config_path.write_text(json.dumps(bad), encoding='utf-8')
        checked_baseline(baseline_dir, load_json_config(args, baseline_dir))


def load_json_config(args, baseline_dir):
    return json.loads((args.output_dir/'config_e5a3.json').read_text(encoding='utf-8'))
