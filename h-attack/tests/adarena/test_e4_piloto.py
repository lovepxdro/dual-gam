"""Testes sintéticos do E4-A, sem dados reais."""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[3] / 'docs/scientific/scripts/e4_piloto.py'
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location('e4_piloto', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_labels_categoricos_sem_perder_heartbleed():
    s = pd.Series(pd.Categorical(['Benign', 'BENIGN', 'DoS Hulk', 'Heartbleed', None]))
    assert list(m.labels_padronizados(s)) == ['Benign', 'Benign', 'DoS Hulk', 'Heartbleed', '<AUSENTE>']


def test_intersecao_exata_features():
    train = np.array([[1.,2.],[3.,4.]])
    other = np.array([[3.,4.],[5.,6.]])
    assert m.overlap_count(other, m.raw_rows_bytes(train)) == 1


def test_matriz_confusao_binarizada():
    truth = np.array([0, 0, 1, 1])
    scores = np.array([.1, .6, .4, .9])
    assert m.confusao(truth, scores, .5) == [[1, 1], [1, 1]]


def test_mascara_exclui_por_seed_sem_interferir_na_outra():
    seed20 = m.raw_rows_bytes(np.array([[1.,2.],[4.,5.]]))
    seed42 = m.raw_rows_bytes(np.array([[3.,4.]]))
    rows = np.array([[1.,2.],[3.,4.],[7.,8.]])
    assert m.mascara_intersecao(rows, seed20).tolist() == [True, False, False]
    assert m.mascara_intersecao(rows, seed42).tolist() == [False, True, False]
    assert m.overlap_count(rows, seed20) == 1


def test_filtro_exclui_heartbleed_e_nao_duplica_contagens():
    labels = np.array(['Benign', 'DoS Hulk', 'Heartbleed', 'DoS GoldenEye'])
    overlap = np.array([True, False, True, False])
    assert m.mascara_avaliavel(labels, overlap).tolist() == [False, True, False, True]
    assert m.contagem_por_classe(labels, overlap) == {'Benign': 1, 'Heartbleed': 1}


def test_mascara_rejeita_nan():
    import pytest
    with pytest.raises(ValueError):
        m.mascara_intersecao(np.array([[float('nan'), 2.]]), set())


def test_piloto_faz_inferencia_apenas_em_dados_elegiveis(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    # Seeds compartilham features, mas possuem treinos distintos.
    X = np.array([[1.,2.],[5.,6.],[3.,4.]])
    y = np.array([0, 1, 0])
    runs = {
        20: SimpleNamespace(seed=20, train_ids=np.array([0, 1]),
                            config={'treinamento': {'classification_threshold': .5}},
                            pre=SimpleNamespace(scaler=object())),
        42: SimpleNamespace(seed=42, train_ids=np.array([2]),
                            config={'treinamento': {'classification_threshold': .5}},
                            pre=SimpleNamespace(scaler=object())),
    }
    monkeypatch.setattr(m, 'preparar', lambda *args: (X, y, ['a', 'b'], runs, 0))
    monkeypatch.setattr(m, 'carregar_defender', lambda run: run.seed)
    monkeypatch.setattr(m, 'verificar_baseline', lambda run, *_: {'seed': run.seed, 'reproduzido': True})
    monkeypatch.setattr(m, 'sha256_file', lambda _: 'knownhash')
    monkeypatch.setattr(m, 'header', lambda _: ['a','b','Label'])
    monkeypatch.setattr(m, 'schema_report', lambda *_: {'status':'compativel_nomes','mesma_ordem':True})
    df = pd.DataFrame([
        (1.,2., 'Benign'),           # seed 20 exclui
        (3.,4., 'Benign'),           # seed 42 exclui
        (7.,8., 'Benign'),           # ambas avaliam
        (5.,6., 'DoS Hulk'),         # seed 20 exclui
        (9.,10.,'DoS Hulk'),
        (11.,12.,'DoS GoldenEye'),
        (13.,14.,'DoS Slowhttptest'),
        (15.,16.,'DoS slowloris'),
        (17.,18.,'Heartbleed'),      # ambas excluem
    ], columns=['a','b','Label'])
    monkeypatch.setattr(m, 'read_chunks', lambda *args: (df.iloc[i:i+3] for i in range(0,len(df),3)))
    observed = {20: [], 42: []}
    def prever_fake(model, scaler, data, batch_size):
        observed[model].extend(tuple(row) for row in data)
        return np.full(len(data), .75)
    monkeypatch.setattr(m, 'prever', prever_fake)

    audit = {
        'estado':'apto_a_validacao_semantica', 'sha256':'knownhash',
        'schema': {'status':'compativel_nomes','mesma_ordem':True,
                   'n_features_esperadas':2,'label_col_original':'Label'},
        'qualidade_features_com_problemas':{},
        'auditoria':{'duplicatas_checadas':True,'possiveis_duplicatas_hash64':0,
                     'n_linhas':len(df),
                     'rotulos_originais_apos_strip': df.Label.value_counts().to_dict()},
    }
    audit_file = tmp_path/'audit.json'
    audit_file.write_text(json.dumps(audit))
    m.iniciar(tmp_path/'fake.parquet', audit_file, (20,42), tmp_path, tmp_path,
              tmp_path/'out', 3, 3)
    report = json.loads((tmp_path/'out'/'auditoria_intersecao.json').read_text())
    summary = json.loads((tmp_path/'out'/'resumo.json').read_text())
    assert report['n_possiveis_sobreposicoes_exatas_por_seed'] == {'20':2,'42':1}
    assert report['sobreposicoes_por_seed_e_classe']['20']['Benign'] == 1
    assert report['sobreposicoes_por_seed_e_classe']['20']['DoS Hulk'] == 1
    assert report['sobreposicoes_por_seed_e_classe']['42']['Benign'] == 1
    assert report['heartbleed_excluido_sem_duplicar_intersecao_por_seed'] == {'20':1,'42':1}
    assert summary['total_amostras_avaliadas_por_seed'] == {'20':6,'42':7}
    assert len(observed[20]) == 6
    assert len(observed[42]) == 7
    assert (1.,2.) not in observed[20]
    assert (3.,4.) not in observed[42]
    assert (5.,6.) not in observed[20]
    assert (17.,18.) not in observed[20]
    assert (17.,18.) not in observed[42]
    rows = pd.read_csv(tmp_path/'out'/'metricas_por_seed_familia.csv')
    assert len(rows) == 10
    assert rows[(rows.seed == 20)&(rows.familia == 'DoS Hulk')]['n'].iloc[0] == 1
    assert rows[(rows.seed == 42)&(rows.familia == 'DoS Hulk')]['n'].iloc[0] == 2


def test_indice_vetorizado_equivale_ao_filtro_original_com_duplicatas():
    rng = np.random.default_rng(11)
    base = rng.integers(-8, 8, size=(120, 77)).astype(np.float64)
    base[30] = base[2]  # mesma linha em IDs de treino distintos
    base[40] = base[2]
    seeds = {20: np.array([2, 5, 9]), 42: np.array([7, 8, 40])}
    idx = m.IndiceIntersecao(base, seeds)
    outros = np.concatenate([base[[0, 2, 5, 40, 100]],
                              rng.integers(20, 30, size=(20, 77)).astype(np.float64)])
    ids, hit = idx.localizar(outros)
    for seed, train_ids in seeds.items():
        old = m.mascara_intersecao(outros, m.raw_rows_bytes(base[train_ids]))
        assert np.array_equal(idx.mascara(seed, ids, hit), old)
    assert idx.mascara(20, ids, hit)[3]  # base[40] tem mesmo valor de base[2]
    assert idx.mascara(42, ids, hit)[1]  # base[2] tem mesmo valor de base[40]


def test_indice_exato_distingue_zero_com_sinal_e_valores_proximos():
    base = np.array([[-0.0, 1.0], [2.0, 3.0]], dtype='<f8')
    idx = m.IndiceIntersecao(base, {20: np.array([0]), 42: np.array([1])})
    x = np.array([[+0.0, 1.0], [-0.0, 1.0], [2.0, np.nextafter(3., 4.)], [2., 3.]])
    ids, hit = idx.localizar(x)
    assert idx.mascara(20, ids, hit).tolist() == [False, True, False, False]
    assert idx.mascara(42, ids, hit).tolist() == [False, False, False, True]


def test_indice_rejeita_nao_finitos_e_schema_divergente():
    import pytest
    idx = m.IndiceIntersecao(np.array([[1., 2.]]), {20: np.array([0])})
    with pytest.raises(ValueError):
        idx.localizar(np.array([[float('inf'), 2.]]))
    with pytest.raises(ValueError):
        idx.localizar(np.array([[1., 2., 3.]]))
