"""Testes unitários isolados do script científico de E2-A."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[3]/'docs/scientific/scripts/analisar_e2.py'
spec=importlib.util.spec_from_file_location('analisar_e2',SCRIPT)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_mock():
    values=np.zeros((20,21))
    for j in range(1,21):
        values[j-1,j-1]=j/1000
    m={'linhas_atacantes':list(range(1,21)),'colunas_defensores':list(range(21)),'matriz_evasao':values.tolist()}
    h={'taxa_evasao':[j/1000 for j in range(1,21)]}
    return m,h


def test_validate_dimensions_and_units():
    m,h=make_mock()
    np.testing.assert_allclose(module.validate_matrix(1,m,h).shape,(20,21))
    m['matriz_evasao'][0][0]=1.2
    with pytest.raises(ValueError,match='frações'):
        module.validate_matrix(1,m,h)


def test_future_nonadjacent_pair_counts():
    m,h=make_mock();m=module.validate_matrix(1,m,h)
    df=module.make_pairs([(1,m)])
    assert (df.tipo=='futuro_nao_adjacente').sum()==171
    assert (df.tipo=='futuro_adjacente').sum()==19
    assert df.query('tipo=="futuro_nao_adjacente"').eval('atacante >= defensor+2').all()
    assert df.query('tipo=="futuro_nao_adjacente"').eval('defensor >=1 and defensor <=19').all()


def test_matrix_pre_history_parity():
    m,h=make_mock();m['matriz_evasao'][2][2]=0.2
    with pytest.raises(ValueError,match='inconsistentes'):
        module.validate_matrix(1,m,h)


def test_bootstrap_ci_degenerate():
    ci=module.seed_bootstrap_ci(np.ones(15)*3.3,np.random.default_rng(42))
    assert ci == pytest.approx((3.3,3.3))
