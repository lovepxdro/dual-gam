"""Testes de invariantes do E3, sem necessidade do CIC-IDS2017."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

SCRIPTS = Path(__file__).resolve().parents[3] / "docs" / "scientific" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from e3_preflight import split_indices, conferir_scaler  # noqa: E402
from e3_cross_seed import (  # noqa: E402
    converter_perturbacao,
    select_eval_ids,
    noise_for_pair,
    avaliar_modelo,
)


def test_split_reproduzivel_isolado():
    y = np.array([0] * 500 + [1] * 500)
    a = split_indices(y, 42, .2, .1)
    b = split_indices(y, 42, .2, .1)
    assert all(np.array_equal(x, z) for x, z in zip(a, b))
    assert [len(x) for x in a] == [700, 100, 200]
    assert not (set(a[0]) & set(a[1]) | set(a[0]) & set(a[2]) | set(a[1]) & set(a[2]))


def test_scaler_valida_split_e_falha_com_incorreto():
    rng = np.random.default_rng(77)
    X = rng.normal(size=(1000, 3)).astype(np.float32)
    y = np.array([0]*500 + [1]*500)
    train, _, _ = split_indices(y, 42, .2, .1)
    scaler = StandardScaler().fit(X[train])
    conferir_scaler(scaler, X[train])
    other = StandardScaler().fit(X[:700] + 20)
    with pytest.raises(AssertionError, match="Scaler"):
        conferir_scaler(other, X[train])


def test_representacao_entre_scalers_preserva_perturbacao_fisica():
    rng = np.random.default_rng(12)
    scale_src = StandardScaler().fit(rng.normal(size=(100, 2)) * [10, 20] + [100, 50])
    scale_dst = StandardScaler().fit(rng.normal(size=(100, 2)) * [3, 9] + [0, 400])
    x = np.array([[110., 77.], [130., 45.]], dtype=np.float64)
    d = np.array([[.4, -.2], [-.3, .8]])
    result, raw_adv = converter_perturbacao(x, d, scale_src, scale_dst, .3)
    expected_raw = x + .3*d*scale_src.scale_
    assert np.allclose(raw_adv, expected_raw, rtol=1e-8, atol=1e-8)
    assert np.allclose(result, scale_dst.transform(expected_raw).astype(np.float32))


def test_amostras_comuns_e_ruido_em_duas_direcoes():
    y = np.ones(30, dtype=np.float32)
    s20 = np.array([0, 1, 4, 8, 10, 12, 18])
    s42 = np.array([1, 4, 10, 12, 25])
    one = select_eval_ids(s20, s42, y, 20, 42, 200, 3)
    two = select_eval_ids(s42, s20, y, 42, 20, 200, 3)
    assert len(one) == 3
    assert np.array_equal(one, two)
    assert set(one).issubset({1, 4, 10, 12})
    assert torch.equal(noise_for_pair(20, 42, 101, 3, 2), noise_for_pair(42, 20, 101, 3, 2))


class DummyAttacker(torch.nn.Module):
    def forward(self, z):
        return torch.ones(len(z), 2)  # desloca em +epsilon no espaço normalizado


class DummyDefender(torch.nn.Module):
    def forward(self, x):
        return torch.sigmoid(x[:, 0])


def test_avaliacao_conta_evasao_condicional_e_limpa():
    scl = StandardScaler().fit(np.array([[-1., 0.], [1., 0.]]))
    x = np.array([[0.5, 0.], [-.5, 0.], [2., 0.]])
    z = torch.randn(3, 2)
    # eps negativo não é o cenário original mas isola o cálculo de métricas:
    out = avaliar_modelo(DummyAttacker(), DummyDefender(), scl, scl, x, z,
                         epsilon=-1., threshold=.5, batch=2)
    assert out["n"] == 3
    assert out["evasao_limpa"] == pytest.approx(1/3)
    assert out["evasao"] == pytest.approx(2/3)
    assert out["n_detectadas_limpas"] == 2
    assert out["n_sucesso"] == 1
    assert out["sucesso_condicional"] == pytest.approx(.5)


def test_piloto_sintetico_fim_a_fim(tmp_path, monkeypatch):
    """Verifica integração de configs, splittings, checkpoints e CSV de métricas."""
    import json
    import pickle
    import types
    from types import SimpleNamespace
    from e3_preflight import preparar
    from e3_cross_seed import executar

    rng = np.random.default_rng(5)
    X = rng.normal(size=(400, 2)).astype(np.float32)
    y = np.array([0, 1] * 200, dtype=np.float32)
    features = ["f1", "f2"]

    class ToyAttacker(torch.nn.Module):
        def __init__(self, noise_dim, output_dim):
            super().__init__()
            self.rede = torch.nn.Sequential(torch.nn.Linear(noise_dim, output_dim), torch.nn.Tanh())
        def forward(self, z):
            return self.rede(z)

    class ToyDefender(torch.nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.rede = torch.nn.Sequential(torch.nn.Linear(input_dim, 1), torch.nn.Sigmoid())
        def forward(self, x):
            return self.rede(x).view(-1)

    class DummyPre:
        def __init__(self, scaler, feature_names, audit_info):
            self.scaler = scaler
            self.feature_names = feature_names
            self.audit_info = audit_info
        @classmethod
        def carregar(cls, path):
            with (path / "dummy_pre.pkl").open("rb") as f:
                data = pickle.load(f)
            return cls(**data)

    dmod = types.ModuleType("adarena")
    bmod = types.ModuleType("adarena.builtin")
    class FakeRegistry:
        def create(self, component_id, **kwargs):
            assert component_id == "adarena.cicids2017"
            return SimpleNamespace(load=lambda source: SimpleNamespace(
                X=X, y=y, schema=SimpleNamespace(feature_names=features)))
    bmod.create_default_registry = lambda: FakeRegistry()
    gmod = types.ModuleType("gan")
    pmod = types.ModuleType("gan.preprocessing")
    pmod.Preprocessador = DummyPre
    mmod = types.ModuleType("gan.models")
    mmod.Atacante = ToyAttacker
    mmod.Defensor = ToyDefender
    for name, mod in (("adarena", dmod), ("adarena.builtin", bmod),
                      ("gan", gmod), ("gan.preprocessing", pmod), ("gan.models", mmod)):
        monkeypatch.setitem(sys.modules, name, mod)

    root = tmp_path / "models" / "e0" / "experiments"
    configs = tmp_path / "configs" / "e0"
    configs.mkdir(parents=True)
    for s in (20, 42):
        path = root / f"run_mock_seed{s}"
        (path / "preprocessador").mkdir(parents=True)
        (path / "checkpoints").mkdir()
        train, val, test = split_indices(y, s, .2, .1)
        scaler = StandardScaler().fit(X[train])
        audit = {"duplicatas_removidas": 0, "random_state": s,
                 "split": {"treino": len(train), "validacao": len(val), "teste": len(test)}}
        with (path / "preprocessador" / "dummy_pre.pkl").open("wb") as f:
            pickle.dump({"scaler": scaler, "feature_names": features, "audit_info": audit}, f)
        for checkpoint in ("atacante_rodada_03.pth", "atacante_rodada_20.pth"):
            torch.save(ToyAttacker(2, 2).state_dict(), path / "checkpoints" / checkpoint)
        for checkpoint in ("defensor_rodada_02.pth", "defensor_rodada_19.pth", "defensor_rodada_20.pth"):
            torch.save(ToyDefender(2).state_dict(), path / "checkpoints" / checkpoint)
        cfg = {"mode": "train", "seed": s,
               "components": {"attacker": "fake.attacker", "defender": "fake.defender"},
               "dados": {"input_dim": 2, "noise_dim": 2, "test_size": .2, "validation_size": .1, "feature_names": features},
               "treinamento": {"epsilon": .3, "classification_threshold": .5}}
        (path / "config_execucao.json").write_text(json.dumps(cfg))
        (configs / f"seed-{s}.toml").write_text(
            f'seed = {s}\n[attack_dataset]\ncomponent_id = "adarena.cicids2017"\nsource = "fake.parquet"\n'
            '[attacker]\ncomponent_id = "fake.attacker"\n'
            '[defender]\ncomponent_id = "fake.defender"\n'
            '[training]\nepsilon = 0.3\nclassification_threshold = 0.5\nnoise_dim = 2\n')

    X_out, y_out, feat_out, runs, removed = preparar((20, 42), root, configs)
    assert removed == 0 and feat_out == features
    rows = executar(X_out, y_out, runs, n_eval=5, noise_seed=1234, selection_seed=333)
    assert len(rows) == 2*2*2*3
    assert {row["avaliacao"] for row in rows} == {
        "destino", "controle_mesma_trajetoria", "efetividade_na_origem"}
    assert all(row["n"] > 0 for row in rows)
    assert all(0 <= row["evasao"] <= 1 for row in rows)
    assert len({row["base_test_ids_sha256"] for row in rows}) == 1
