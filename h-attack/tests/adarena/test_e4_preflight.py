"""Testes isolados do auditor E4-P, sem dataset CIC-IDS2017."""
from pathlib import Path
import importlib.util
import json
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / 'docs/scientific/scripts/e4_preflight.py'
spec = importlib.util.spec_from_file_location('e4_preflight', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_schema_exato():
    a = m.schema_report([' F1', 'F2', ' Label'], ['F1', 'F2'])
    assert a['status'] == 'compativel_nomes'
    assert not a['reordenacao_necessaria']


def test_schema_faltantes_e_extras():
    a = m.schema_report(['F1', 'Foo', 'Label'], ['F1', 'F2'])
    assert a['features_faltantes'] == ['F2']
    assert a['features_extras'] == ['Foo']
    assert a['status'] == 'incompativel_nomes'


def test_schema_reordenavel():
    a = m.schema_report(['F2', 'F1', 'Label'], ['F1', 'F2'])
    assert a['status'] == 'compativel_nomes'
    assert a['reordenacao_necessaria']


def test_audit_csv_labels_valores(tmp_path):
    path = tmp_path / 'wednesday.csv'
    pd.DataFrame({'F1': [1, float('inf'), None], 'F2': [2,3,'no'],
                  ' Label': ['BENIGN', 'DoS Hulk', 'DoS Hulk']}).to_csv(path, index=False)
    cols = m.header(path)
    s = m.schema_report(cols, ['F1', 'F2'])
    q = m.audit_rows(path, columns=cols, expected=['F1', 'F2'], label_col=s['label_col_original'], chunksize=2)
    assert q['n_linhas'] == 3
    assert q['rotulos_originais_apos_strip']['DoS Hulk'] == 2
    assert q['qualidade_features_coincidentes']['F1']['infinitos'] == 1
    assert q['qualidade_features_coincidentes']['F1']['ausentes'] == 1
    assert q['qualidade_features_coincidentes']['F2']['nao_numericos'] == 1


def test_audit_bloqueia_incompatibilidade(tmp_path):
    out = tmp_path / 'ext.csv'
    pd.DataFrame({'F1':[1], 'Label':['BENIGN']}).to_csv(out,index=False)
    run = tmp_path/'run';run.mkdir()
    (run/'config_execucao.json').write_text(json.dumps({'dados':{'feature_names':['F1','F2']}, 'seed':42}))
    d = m.audit(out, run, chunksize=2, do_duplicates=False)
    assert d['estado'] == 'bloqueado'
    assert d['schema']['features_faltantes'] == ['F2']


def test_parquet_categorico_benign_e_dos(tmp_path):
    pytest.importorskip('pyarrow')
    path = tmp_path / 'wednesday.parquet'
    labels = pd.Categorical(['Benign','DoS Hulk','DoS GoldenEye','DoS slowloris','DoS Slowhttptest',None])
    pd.DataFrame({'F1':[1., 2., 3., 4., 5., 6.], 'Label': labels}).to_parquet(path)
    run = tmp_path / 'run'; run.mkdir()
    (run / 'config_execucao.json').write_text(json.dumps({'dados':{'feature_names':['F1']}, 'seed':42}))
    q = m.audit(path, run, chunksize=3, do_duplicates=False)
    assert q['benign_presente'] is True
    assert q['estado'] == 'apto_a_validacao_semantica'
    assert q['auditoria']['rotulos_originais_apos_strip']['<AUSENTE>'] == 1
