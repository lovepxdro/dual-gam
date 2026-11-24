"""Controles de segurança numérica e de regressão do E5-A2."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'docs' / 'scientific' / 'scripts'))
from e5_train import (diagnosticar_perdas_iniciais, init_attacker, logits_defender,
                      norma_gradiente, perda_e5, train_one)
from gan.models import Atacante, Defensor


def test_pre_sigmoid_logits_reproduz_forward_defensor_no_modo_eval():
    d = Defensor(input_dim=3).eval().requires_grad_(False)
    x = torch.randn(12, 3)
    with torch.no_grad():
        logits = logits_defender(d, x)
        assert torch.equal(torch.sigmoid(logits), d(x))


def test_rejeita_defender_que_nao_seja_o_original():
    with pytest.raises(TypeError):
        logits_defender(nn.Sequential(nn.Linear(3, 1), nn.Tanh()), torch.randn(4, 3))


def test_logit_estavel_preserva_gradiente_em_saturacao():
    """BCE(prob=1) pode produzir gradiente zero; BCE(logit=40) não."""
    d = Defensor(input_dim=3).eval().requires_grad_(False)
    linear = nn.Linear(3, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.tensor([[80.0, 0.0, 0.0]]))
        linear.bias.zero_()
    d.rede = nn.Sequential(linear, nn.Sigmoid())
    x = torch.tensor([[0.5, 0, 0]], requires_grad=True)
    loss_prob, prob, _ = perda_e5(d, x, 'probs')
    loss_prob.backward()
    grad_prob = float(x.grad[0, 0]); x.grad = None
    loss_logits, prob_logits, logit = perda_e5(d, x, 'logits')
    loss_logits.backward()
    grad_logits = float(x.grad[0, 0])
    assert prob.item() == prob_logits.item() == 1.0
    assert logit.item() == 40.0
    assert grad_prob == 0.0
    assert grad_logits == pytest.approx(80.0, abs=1e-4)
    assert float(loss_logits.detach()) == pytest.approx(40.0, rel=1e-6)


def test_formula_loss_equivale_quando_nao_ha_saturacao():
    d = Defensor(input_dim=3).eval().requires_grad_(False)
    x = torch.randn(32, 3)
    p, _, _ = perda_e5(d, x, 'probs')
    l, _, _ = perda_e5(d, x, 'logits')
    assert float(p) == pytest.approx(float(l), rel=1e-5, abs=1e-6)


@pytest.mark.parametrize('arm', ['A0', 'A1'])
def test_diagnostico_gradientes_nao_atualiza_pesos_nem_batchnorm(arm):
    torch.manual_seed(4)
    a0 = Atacante(noise_dim=2, output_dim=3).eval()
    attacker = init_attacker(arm, a0.state_dict(), input_dim=3, noise_dim=2, init_seed=13)
    attacker.train()
    d = Defensor(input_dim=3).eval().requires_grad_(False)
    previous_state = {k:v.clone() for k,v in attacker.state_dict().items()}
    d_state = {k:v.clone() for k,v in d.state_dict().items()}
    report = diagnosticar_perdas_iniciais(attacker, arm, d, torch.randn(32, 3),
                                         seed_e0=20, rep_seed=1042, noise_dim=2, epsilon=.3)
    assert attacker.training is True
    assert report['n_amostras_treino'] == 32
    assert report['probs']['gradiente_l2'] >= 0
    assert report['logits']['gradiente_l2'] >= 0
    assert report['logits']['media_logit'] is not None
    assert all(torch.equal(v, attacker.state_dict()[k]) for k,v in previous_state.items())
    assert all(torch.equal(v, d.state_dict()[k]) for k,v in d_state.items())
    assert all(p.grad is None for p in attacker.parameters())


@pytest.mark.parametrize('arm', ['A0', 'A1'])
def test_train_logits_com_gradiente_e_defender_congelado(arm):
    torch.manual_seed(8)
    baseline = Atacante(noise_dim=2, output_dim=3)
    a = init_attacker(arm, baseline.state_dict(), input_dim=3, noise_dim=2, init_seed=777)
    d = Defensor(input_dim=3).eval().requires_grad_(False)
    original_defender = {k:v.clone() for k,v in d.state_dict().items()}
    x = torch.randn(22, 3)
    curves, elapsed = train_one(a, arm, d, x, x[:8].numpy(), seed_e0=42, rep_seed=1042,
                                 epochs=2, batch_size=8, noise_dim=2, epsilon=.3,
                                 lr=.0002, betas=(.5,.999), eval_batch=8, loss_mode='logits')
    assert elapsed >= 0
    assert len(curves) == 2
    for curve in curves:
        assert np.isfinite(curve['loss_media'])
        assert np.isfinite(curve['grad_l2_media'])
        assert 0 <= curve['fracao_batches_grad_zero'] <= 1
        assert 0 <= curve['fracao_prob_exatamente_um'] <= 1
        assert curve['grad_l2_media'] >= 0
        if arm == 'A1':
            assert curve['grad_ramo_l2_media'] is not None
        else:
            assert curve['grad_ramo_l2_media'] is None
    assert all(torch.equal(v, d.state_dict()[k]) for k,v in original_defender.items())
    assert all(p.grad is None for p in d.parameters())


def test_e5a2_end_to_end_and_resume(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import json
    import e5_train as module
    from sklearn.preprocessing import StandardScaler

    torch.set_num_threads(1)
    rng = np.random.default_rng(17)
    x = rng.normal(size=(260, 77))
    y = np.tile(np.asarray([0, 1]), 130)
    state_a = Atacante(noise_dim=32, output_dim=77).state_dict()
    run_info = {}
    for seed in (20, 42):
        a, b, c = module.split_indices(y, seed, .2, .1)
        scaler = StandardScaler().fit(x[a])
        run_info[seed] = module.Run(seed, tmp_path,
            {'treinamento': {'lr_atacante': .0002, 'adam_betas_atacante': [.5, .999]}},
            SimpleNamespace(scaler=scaler), a, b, c, state_a,
            Defensor(input_dim=77).eval().requires_grad_(False))
    monkeypatch.setattr(module, 'prepare_runs', lambda *args: (x, y, run_info, 0, 'synthetic.parquet'))
    preflight = tmp_path/'preflight.json'
    preflight.write_text('{"status":"aprovado"}')
    args = SimpleNamespace(runs_root=tmp_path, preflight=preflight, seeds=[20, 42],
        replicas=[1042], epochs=1, batch_size=16, max_train_samples=38,
        val_samples=8, test_samples=10, eval_batch=8, device='cpu',
        output_dir=tmp_path/'a2', resume=False, loss_mode='logits')
    module.run(args)
    out = args.output_dir
    assert (out/'config_e5a2.json').exists()
    summary = json.loads((out/'resumo_e5a.json').read_text())
    assert summary['experiment'] == 'E5-A2'
    assert summary['n_treinamentos'] == 4
    assert len(summary['diagnosticos_iniciais']) == 4
    record = json.loads((out/'seed_20'/'rep_1042'/'A1'/'resultado.json').read_text())
    assert 'diagnostico_pre_treino' in record
    assert record['curva_validacao'][0]['grad_ramo_l2_media'] is not None
    args.resume = True
    module.run(args)
    assert json.loads((out/'resumo_e5a.json').read_text())['n_treinamentos'] == 4
