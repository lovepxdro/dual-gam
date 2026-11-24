"""E4-A: avaliação OFFLINE dos D20 treinados na sexta-feira sobre quarta-feira.

Requer E4-P corrigido com status apto_a_validacao_semantica, os runs E0,
e os scripts E3 já incorporados ao repositório. Os pesos e scalers do E0
nunca são alterados. A execução aborta em dados divergentes e filtra correspondências exatas ao treino por seed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e3_preflight import preparar
from e4_preflight import read_chunks, header, sha256_file, schema_report

LOG = logging.getLogger('e4-piloto')
FAMILIAS = ('Benign', 'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest')
EXCLUIDO = 'Heartbleed'


def labels_padronizados(labels: pd.Series) -> pd.Series:
    """Normaliza somente caixa de Benign, não altera classes de ataque."""
    s = labels.astype('string').fillna('<AUSENTE>').str.strip()
    s = s.mask(s.str.casefold() == 'benign', 'Benign')
    return s


def raw_rows_bytes(X: np.ndarray) -> set[bytes]:
    """Representação por valor de 77 features, sem rótulo, em float64 little-endian."""
    a = np.ascontiguousarray(np.asarray(X, dtype='<f8'))
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Array inválido para checagem de sobreposição')
    return {a[i].tobytes() for i in range(len(a))}


def mascara_intersecao(X: np.ndarray, seen: set[bytes]) -> np.ndarray:
    """Identifica features idênticas às do treino bruto, sem usar os rótulos.

    Preserva o mesmo critério da auditoria anterior: 77 valores float64 LE.
    Não considera igualdade aproximada e não certifica identidade de fluxo.
    """
    a = np.ascontiguousarray(np.asarray(X, dtype='<f8'))
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Array inválido para checagem de sobreposição')
    return np.fromiter((a[i].tobytes() in seen for i in range(len(a))),
                       count=len(a), dtype=bool)


def overlap_count(X: np.ndarray, seen: set[bytes]) -> int:
    """Compatibilidade com o diagnóstico anterior do E4-A."""
    return int(mascara_intersecao(X, seen).sum())


class IndiceIntersecao:
    """Índice vetorizado EXATO para todas as seeds sobre a mesma origem E0.

    As linhas são comparadas como bytes de float64 little-endian, sem hash.
    Uma busca na origem compartilhada serve para TODAS as seeds; as diferenças
    de treino ficam em máscaras booleanas pequenas indexadas pelo ID da linha.
    """

    def __init__(self, X: np.ndarray, train_ids_por_seed: dict[int, np.ndarray]):
        keys = self._keys(X)
        self.chaves, inverso = np.unique(keys, return_inverse=True)
        self.treino = {}
        for seed, ids in train_ids_por_seed.items():
            ids = np.asarray(ids, dtype=np.intp)
            if ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(keys)):
                raise ValueError(f'Seed {seed}: índices de treino inválidos')
            mask = np.zeros(len(self.chaves), dtype=bool)
            mask[inverso[ids]] = True
            self.treino[seed] = mask

    @staticmethod
    def _keys(X: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(np.asarray(X, dtype='<f8'))
        if a.ndim != 2 or a.shape[1] == 0 or not np.isfinite(a).all():
            raise ValueError('Array inválido para checagem de sobreposição')
        return a.view(np.dtype((np.void, a.shape[1] * 8))).reshape(-1)

    def localizar(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Busca exata feita uma vez por chunk, com retorno de ID e acerto."""
        keys = self._keys(X)
        if keys.dtype != self.chaves.dtype:
            raise ValueError('Quantidade de features difere do E0')
        ids = np.searchsorted(self.chaves, keys)
        hit = ids < len(self.chaves)
        validos = np.flatnonzero(hit)
        hit[validos] = self.chaves[ids[validos]] == keys[validos]
        return ids, hit

    def mascara(self, seed: int, ids: np.ndarray, hit: np.ndarray) -> np.ndarray:
        """Seleciona a presença no treino da seed, sem nova busca de linhas."""
        mask = np.zeros(len(hit), dtype=bool)
        validos = np.flatnonzero(hit)
        mask[validos] = self.treino[seed][ids[validos]]
        return mask


def contagem_por_classe(labels: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    """Relata todos os rótulos que coincidem com o treino, inclusive Heartbleed."""
    values, counts = np.unique(labels[mask], return_counts=True)
    return {str(label): int(n) for label, n in zip(values, counts)}


def mascara_avaliavel(labels: np.ndarray, sobreposicao: np.ndarray) -> np.ndarray:
    """Remove somente interseção com treino da seed e o rótulo Heartbleed."""
    return (~sobreposicao) & np.isin(labels, FAMILIAS)


def carregar_defender(run) -> torch.nn.Module:
    from gan.models import Defensor
    model = Defensor(input_dim=int(run.config['dados']['input_dim']))
    path = run.path / 'checkpoints' / 'defensor_rodada_20.pth'
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def prever(model, scaler, data: np.ndarray, batch_size: int) -> np.ndarray:
    result = []
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            batch = np.asarray(data[start:start+batch_size], dtype=np.float64)
            xs = scaler.transform(batch).astype(np.float32)
            if not np.isfinite(xs).all():
                raise ValueError('Dados não finitos após scaler: abortado')
            output = model(torch.from_numpy(xs)).reshape(-1).detach().cpu().numpy()
            result.append(np.asarray(output, dtype=np.float64))
    return np.concatenate(result) if result else np.array([], dtype=np.float64)


def confusao(y: np.ndarray, scores: np.ndarray, threshold: float) -> list[list[int]]:
    pred = scores >= threshold
    return [[int(np.sum((y == 0) & (~pred))), int(np.sum((y == 0) & pred))],
            [int(np.sum((y == 1) & (~pred))), int(np.sum((y == 1) & pred))]]


def verificar_baseline(run, model, X, y, batch_size: int) -> dict:
    cfg = run.config
    threshold = float(cfg['treinamento']['classification_threshold'])
    ids = run.test_ids
    scores = prever(model, run.pre.scaler, X[ids], batch_size)
    matrix = confusao(y[ids], scores, threshold)
    summary = json.loads((run.path / 'summary.json').read_text(encoding='utf-8'))
    expected = summary['teste_final']['confusion_matrix']
    if matrix != expected:
        raise AssertionError(f'Seed {run.seed}: reprodução da matriz E0 falhou. Obtida {matrix}, esperada {expected}')
    return {'seed': run.seed, 'n_teste': len(ids), 'confusion_matrix': matrix,
            'reproduzido': True, 'threshold': threshold}


def iniciar(dataset: Path, audit_file: Path, seeds: tuple[int, ...], run_root: Path,
            config_dir: Path, output: Path, batch_size: int, chunksize: int):
    """Audita sobreposições, filtra por seed e avalia apenas registros elegíveis."""
    output.mkdir(parents=True, exist_ok=True)
    audit = json.loads(audit_file.read_text(encoding='utf-8'))
    if audit.get('estado') != 'apto_a_validacao_semantica':
        raise RuntimeError('E4-P não está apto: corrija e repita o preflight')
    if sha256_file(dataset) != audit.get('sha256'):
        raise RuntimeError('Hash SHA-256 difere do dataset auditado: execute novamente o E4-P')
    X, y, features, runs, _ = preparar(seeds, run_root, config_dir)
    if audit['schema']['status'] != 'compativel_nomes' or not audit['schema']['mesma_ordem']:
        raise RuntimeError('Features não compatíveis ou exigem reordenação explícita')
    if audit['schema']['n_features_esperadas'] != len(features):
        raise RuntimeError('Número de features difere')
    actual_columns = header(dataset)
    schema_atual = schema_report(actual_columns, features)
    if schema_atual['status'] != 'compativel_nomes' or not schema_atual['mesma_ordem']:
        raise RuntimeError('Schema de quarta-feira incompatível com origem')
    if audit['qualidade_features_com_problemas']:
        raise RuntimeError('Auditoria de qualidade identificou problemas')
    if not audit['auditoria'].get('duplicatas_checadas'):
        raise RuntimeError('E4-P sem checagem de duplicatas: use --check-duplicates')
    if audit['auditoria'].get('possiveis_duplicatas_hash64', 0) > 0:
        raise RuntimeError('Possíveis duplicatas internas no destino: revisar a auditoria E4-P')

    models = {seed: carregar_defender(run) for seed, run in runs.items()}
    baselines = [verificar_baseline(run, models[seed], X, y, batch_size)
                 for seed, run in sorted(runs.items())]
    (output / 'validacao_origem.json').write_text(
        json.dumps(baselines, indent=2, ensure_ascii=False), encoding='utf-8')
    LOG.info('Reprodução do E0 confirmada nas seeds %s', list(seeds))

    # Uma indexação exata sobre o E0 compartilhado: cada seed só mantém
    # uma pequena máscara indicando quais registros pertencem ao seu treino.
    indice = IndiceIntersecao(X, {seed: run.train_ids for seed, run in runs.items()})
    LOG.info('Índice exato de interseção preparado: %d padrões de origem para %d seeds',
             len(indice.chaves), len(seeds))
    overlaps = {seed: collections.Counter() for seed in seeds}
    label_count = collections.Counter()
    n_rows = 0
    for part in read_chunks(dataset, chunksize):
        labels = labels_padronizados(part[audit['schema']['label_col_original']]).to_numpy(dtype=str)
        label_count.update(collections.Counter(labels))
        raw = np.asarray(part.loc[:, features], dtype=np.float64)
        if not np.isfinite(raw).all():
            raise RuntimeError('Dados de quarta-feira não finitos')
        ids, hit = indice.localizar(raw)  # busca única para todas as seeds
        for seed in seeds:
            mask = indice.mascara(seed, ids, hit)
            overlaps[seed].update(contagem_por_classe(labels, mask))
        n_rows += len(part)
        if n_rows % (chunksize * 4) < len(part):
            LOG.info('Auditoria de sobreposições: %d/%d linhas',
                     n_rows, audit['auditoria']['n_linhas'])

    if n_rows != audit['auditoria']['n_linhas'] or dict(label_count) != audit['auditoria']['rotulos_originais_apos_strip']:
        raise RuntimeError('Contagens diferentes da auditoria E4-P')
    unknown = set(label_count) - set(FAMILIAS) - {EXCLUIDO}
    if unknown:
        raise RuntimeError(f'Rótulos não previstos: {sorted(unknown)}')
    if any(label_count.get(family, 0) == 0 for family in FAMILIAS):
        raise RuntimeError('Família esperada sem amostras no dataset auditado')

    # Excluímos Heartbleed uma única vez: registros Heartbleed que coincidem com
    # o treino já estão na contagem de sobreposição e não são contados duas vezes.
    overlap_per_seed = {str(seed): int(sum(overlaps[seed].values())) for seed in seeds}
    overlap_by_class = {str(seed): {label: int(overlaps[seed].get(label, 0))
                                    for label in sorted(label_count)} for seed in seeds}
    excluded_heartbleed = {str(seed): int(label_count.get(EXCLUIDO, 0) - overlaps[seed].get(EXCLUIDO, 0))
                           for seed in seeds}
    retained_by_class = {
        str(seed): {label: int(label_count[label] - overlaps[seed].get(label, 0))
                    for label in FAMILIAS} for seed in seeds
    }
    for seed in seeds:
        if any(retained_by_class[str(seed)][label] <= 0 for label in FAMILIAS):
            raise RuntimeError(f'Seed {seed}: alguma família não tem amostras após a exclusão')
    overlap_result = {
        'estado': 'filtragem_conservadora_aplicada',
        'dataset_sha256': audit['sha256'], 'n_rows': n_rows,
        'n_possiveis_sobreposicoes_exatas_por_seed': overlap_per_seed,
        'sobreposicoes_por_seed_e_classe': overlap_by_class,
        'heartbleed_total_no_dataset': int(label_count.get(EXCLUIDO, 0)),
        'heartbleed_excluido_sem_duplicar_intersecao_por_seed': excluded_heartbleed,
        'amostras_retidas_por_seed_e_classe': retained_by_class,
        'labels': dict(sorted(label_count.items())),
        'procedimento': ('Para cada seed, excluir registros de quarta-feira cujas 77 features brutas '
                        'convertidas a float64 little-endian são idênticas às do treino E0 dessa seed; '
                        'excluir Heartbleed separadamente, sem contagem dupla'),
        'observacao': ('Correspondência exata de valores não identifica necessariamente o mesmo fluxo; '
                       'a filtragem pode remover amostras legítimas repetidas e não detecta '
                       'sobreposições com features numericamente diferentes'),
    }
    (output / 'auditoria_intersecao.json').write_text(
        json.dumps(overlap_result, indent=2, ensure_ascii=False), encoding='utf-8')
    for seed in seeds:
        LOG.info('Seed %d: sobreposições excluídas=%d (%s) | Heartbleed adicional=%d',
                 seed, overlap_per_seed[str(seed)], overlap_by_class[str(seed)],
                 excluded_heartbleed[str(seed)])

    counters = {(seed, lab): {'n': 0, 'pred_ataque': 0, 'soma_score': 0.0}
                for seed in seeds for lab in FAMILIAS}
    aval_count = {seed: 0 for seed in seeds}
    inferidos = 0
    for part in read_chunks(dataset, chunksize):
        labels = labels_padronizados(part[audit['schema']['label_col_original']]).to_numpy(dtype=str)
        raw = np.asarray(part.loc[:, features], dtype=np.float64)
        if not np.isfinite(raw).all():
            raise RuntimeError('Dados de quarta-feira não finitos')
        ids, hit = indice.localizar(raw)  # busca única para todas as seeds
        for seed, run in runs.items():
            sobreposicao = indice.mascara(seed, ids, hit)
            elegiveis = mascara_avaliavel(labels, sobreposicao)
            if not np.any(elegiveis):
                continue
            raw_elegiveis = raw[elegiveis]
            labels_elegiveis = labels[elegiveis]
            threshold = float(run.config['treinamento']['classification_threshold'])
            # A inferência recebe SOMENTE os registros elegíveis da seed.
            scores = prever(models[seed], run.pre.scaler, raw_elegiveis, batch_size)
            if len(scores) != len(raw_elegiveis) or not np.isfinite(scores).all():
                raise RuntimeError(f'Seed {seed}: saída do Defender inválida')
            pred = scores >= threshold
            aval_count[seed] += len(raw_elegiveis)
            for lab in FAMILIAS:
                subset = labels_elegiveis == lab
                st = counters[(seed, lab)]
                st['n'] += int(subset.sum())
                st['pred_ataque'] += int(np.sum(pred[subset]))
                st['soma_score'] += float(np.sum(scores[subset]))
        inferidos += len(part)
        if inferidos % (chunksize * 4) < len(part):
            LOG.info('Inferência concluída para %d/%d linhas do destino (%d seeds)',
                     inferidos, n_rows, len(seeds))
    rows = []
    for (seed, lab), st in sorted(counters.items()):
        n = st['n']; positive = st['pred_ataque']
        expected = retained_by_class[str(seed)][lab]
        if not n or n != expected:
            raise RuntimeError(f'Seed {seed}, classe {lab}: esperado {expected}, avaliado {n}')
        rows.append({'seed': seed, 'familia': lab, 'n': n, 'pred_ataque': positive,
                     'pred_benigno': n-positive, 'taxa': positive/n,
                     'tipo_taxa': 'FPR' if lab == 'Benign' else 'Recall',
                     'score_medio_ataque': st['soma_score']/n,
                     'excluidas_por_intersecao': overlap_by_class[str(seed)][lab]})
    columns = ['seed', 'familia', 'n', 'pred_ataque', 'pred_benigno', 'taxa',
               'tipo_taxa', 'score_medio_ataque', 'excluidas_por_intersecao']
    with (output / 'metricas_por_seed_familia.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    result = {'experimento': 'E4-A', 'etapa': 'campanha' if len(seeds) > 2 else 'piloto', 'estado': 'concluido',
              'seeds': list(seeds), 'dataset_sha256': audit['sha256'],
              'total_amostras_destino': n_rows, 'excluidas_heartbleed_originais': int(label_count.get(EXCLUIDO, 0)),
              'excluidas_por_intersecao_por_seed': overlap_per_seed,
              'heartbleed_adicional_excluido_por_seed': excluded_heartbleed,
              'total_amostras_avaliadas_por_seed': {str(s): int(aval_count[s]) for s in seeds},
              'sem_novos_treinamentos': True,
              'observacao': ('Mudança de dia e família de ataque simultâneas; não separa efeitos causais. '
                             'Exclusão conservadora baseada em igualdade exata das features'),
              'metricas': rows}
    (output / 'resumo.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    for row in rows:
        LOG.info('seed %d | %-18s | %s %.2f%% | N=%d | excluídas=%d',
                 row['seed'], row['familia'], row['tipo_taxa'], row['taxa']*100,
                 row['n'], row['excluidas_por_intersecao'])
    LOG.info('E4-A concluído. Arquivos em %s', output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--audit-file', type=Path, default=Path('models/e4/preflight/auditoria_dataset.json'))
    p.add_argument('--seeds', type=int, nargs='+', default=[20, 42])
    p.add_argument('--run-root', type=Path, default=Path('models/e0/experiments'))
    p.add_argument('--config-dir', type=Path, default=Path('configs/e0'))
    p.add_argument('--output-dir', type=Path, default=Path('models/e4/piloto'))
    p.add_argument('--batch-size', type=int, default=2048)
    p.add_argument('--chunksize', type=int, default=25000)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if args.batch_size < 1 or args.chunksize < 1:
        p.error('Tamanhos de bloco devem ser positivos')
    iniciar(args.dataset, args.audit_file, tuple(sorted(set(args.seeds))),
            args.run_root, args.config_dir, args.output_dir, args.batch_size, args.chunksize)


if __name__ == '__main__':
    main()
