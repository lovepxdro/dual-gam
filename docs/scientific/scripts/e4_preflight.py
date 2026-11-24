"""E4-P: auditoria SEM TREINAMENTO do dataset externo (quarta-feira CIC-IDS2017).

Leitura em blocos: registra schema, rótulos, NaN, inf, valores não numéricos,
possíveis duplicatas e SHA-256. Falhas de schema impedem o avanço para inferência.
Não faz renomeação especulativa, limpeza silenciosa nem fit de scaler.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger('e4-preflight')
META = {
    'flow id', 'source ip', 'src ip', 'source port', 'src port',
    'destination ip', 'dst ip', 'destination port', 'dst port',
    'timestamp', 'simillarhttp',
}
ATTACKS = {'DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'}
IGNORABLE_EXT = {'.csv', '.parquet', '.gz', '.zip'}


def normalized_name(name: str) -> str:
    return str(name).strip().lstrip('\ufeff')


def detect_label(columns: list[str]) -> str | None:
    candidates = [col for col in columns if normalized_name(col).lower() in ('label', 'labels', 'class')]
    return candidates[0] if len(candidates) == 1 else None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def find_dataset(folder: Path, base: str = 'ddos-friday-no-metadata') -> Path:
    candidates = []
    for p in sorted(folder.rglob('*')):
        if not p.is_file():
            continue
        name = p.name.lower()
        if not (name.endswith('.csv') or name.endswith('.csv.gz') or name.endswith('.parquet')):
            continue
        if base in name:
            continue
        candidates.append(p)
    if len(candidates) != 1:
        names = ', '.join(map(str, candidates)) or '(nenhum CSV/Parquet externo encontrado)'
        raise RuntimeError(f'Escolha o arquivo com --dataset. Candidatos: {names}')
    return candidates[0]


def find_reference_run(root: Path, seed: int) -> Path:
    candidates = []
    for p in sorted(root.glob(f'run_*_seed{seed}')):
        cfg = p / 'config_execucao.json'
        if cfg.is_file() and int(json.loads(cfg.read_text(encoding='utf-8')).get('seed', -1)) == seed:
            candidates.append(p)
    if len(candidates) != 1:
        raise RuntimeError(f'Referência seed {seed}: esperado 1 run, encontrados {len(candidates)}: {candidates}')
    return candidates[0]


def schema_report(columns: list[str], expected: list[str]) -> dict:
    cleaned = [normalized_name(x) for x in columns]
    dups = [name for name, n in collections.Counter(cleaned).items() if n > 1]
    label_col = detect_label(columns)
    payload = [normalized_name(x) for x in columns if x != label_col]
    ignored_metadata = [x for x in payload if x.lower() in META or re.match(r'^unnamed: ?\d+$', x.lower())]
    features = [x for x in payload if x not in ignored_metadata]
    missing = sorted(set(expected) - set(features))
    extra = sorted(set(features) - set(expected))
    same_order = features == expected
    compatible = label_col is not None and not dups and not missing and not extra
    return {
        'status': 'compativel_nomes' if compatible else 'incompativel_nomes',
        'label_col_original': label_col,
        'n_features_esperadas': len(expected),
        'n_features_encontradas': len(features),
        'colunas_com_nomes_duplicados_apos_strip': dups,
        'colunas_metadados_ignoraveis': ignored_metadata,
        'features_faltantes': missing,
        'features_extras': extra,
        'mesma_ordem': same_order,
        'reordenacao_necessaria': bool(compatible and not same_order),
        'nota': 'Compatibilidade nominal NÃO comprova equivalência semântica, unidade ou implementação de extração.'
    }


def csv_chunks(path: Path, *, chunksize: int):
    return pd.read_csv(path, chunksize=chunksize, low_memory=False, encoding='utf-8-sig')


def parquet_chunks(path: Path, *, chunksize: int):
    import pyarrow.parquet as pq
    pqfile = pq.ParquetFile(path)
    for rb in pqfile.iter_batches(batch_size=chunksize):
        yield rb.to_pandas()


def header(path: Path) -> list[str]:
    name = path.name.lower()
    if name.endswith('.csv') or name.endswith('.csv.gz'):
        return list(pd.read_csv(path, nrows=0, encoding='utf-8-sig').columns)
    if name.endswith('.parquet'):
        import pyarrow.parquet as pq
        return pq.read_schema(path).names
    raise ValueError('Formato não suportado: use CSV, CSV.GZ ou Parquet extraído')


def read_chunks(path: Path, chunksize: int):
    if path.name.lower().endswith('.parquet'):
        return parquet_chunks(path, chunksize=chunksize)
    return csv_chunks(path, chunksize=chunksize)


def audit_rows(path: Path, *, columns: list[str], expected: list[str], label_col: str | None,
               chunksize: int, do_duplicates: bool = False) -> dict:
    by_original_label = collections.Counter()
    per_column = collections.defaultdict(lambda: {'ausentes': 0, 'nao_numericos': 0, 'infinitos': 0})
    total, candidates_dupe = 0, 0
    seen_hashes = set() if do_duplicates else None
    # Não renomear features, mesmo que não coincidam com o E0. A auditoria é diagnóstica.
    shared_actual = [x for x in columns if normalized_name(x) in set(expected)]
    for part in read_chunks(path, chunksize):
        n = len(part)
        total += n
        if label_col is not None:
            values = part[label_col].astype('string').fillna('<AUSENTE>').str.strip()
            by_original_label.update(values.value_counts(dropna=False).to_dict())
        for col in shared_actual:
            series = part[col]
            missing = series.isna()
            numeric = pd.to_numeric(series, errors='coerce')
            nonnum = numeric.isna() & ~missing
            infinite = np.isinf(numeric.to_numpy(dtype=np.float64))
            key = normalized_name(col)
            per_column[key]['ausentes'] += int(missing.sum())
            per_column[key]['nao_numericos'] += int(nonnum.sum())
            per_column[key]['infinitos'] += int(infinite.sum())
        if do_duplicates:
            # Apenas sinalização de hashes iguais (hash64), não certificação de duplicatas exatas.
            payload = part.drop(columns=[x for x in columns if normalized_name(x).lower() in META], errors='ignore')
            hashes = pd.util.hash_pandas_object(payload, index=False).to_numpy(dtype=np.uint64)
            for h in hashes:
                v = int(h)
                if v in seen_hashes:
                    candidates_dupe += 1
                else:
                    seen_hashes.add(v)
        if total % (chunksize * 10) == 0:
            LOG.info('Auditadas %d linhas', total)
    return {
        'n_linhas': total,
        'rotulos_originais_apos_strip': dict(sorted(by_original_label.items())),
        'qualidade_features_coincidentes': dict(sorted(per_column.items())),
        'n_colunas_de_features_coincidentes': len(shared_actual),
        'possiveis_duplicatas_hash64': candidates_dupe if do_duplicates else None,
        'duplicatas_checadas': do_duplicates,
    }


def audit(path: Path, run_dir: Path, *, chunksize: int, do_duplicates: bool):
    cfg_path = run_dir / 'config_execucao.json'
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    expected = cfg['dados']['feature_names']
    cols = header(path)
    schema = schema_report(cols, expected)
    content = audit_rows(path, columns=cols, expected=expected,
                         label_col=schema['label_col_original'], chunksize=chunksize,
                         do_duplicates=do_duplicates)
    known = content['rotulos_originais_apos_strip']
    missing_groups = sorted(ATTACKS - set(known))
    col_quality = content['qualidade_features_coincidentes']
    bad_quality = {c: d for c, d in col_quality.items() if any(d.values())}
    benign_present = any(str(k).casefold() == 'benign' for k in known)
    labels_status = not missing_groups and benign_present
    schema_status = schema['status'] == 'compativel_nomes'
    status = 'apto_a_validacao_semantica' if schema_status and labels_status and not bad_quality else 'bloqueado'
    return {
        'experimento': 'E4-P', 'estado': status, 'dataset': str(path), 'sha256': sha256_file(path),
        'bytes': path.stat().st_size, 'referencia_seed': cfg['seed'],
        'run_ref': str(run_dir), 'schema': schema, 'auditoria': content,
        'rotulos_esperados_ausentes': missing_groups,
        'benign_presente': benign_present,
        'qualidade_features_com_problemas': bad_quality,
        'conclusao': 'Somente auditoria; não houve inferência, alteração de dados ou treinamento.',
        'pendencias_antes_do_piloto': ['confirmar equivalencia semantica/unidades',
                                       'conferir intersecao com treino E0',
                                       'validar carregamento scaler e reproducao E0'],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, help='CSV/Parquet da quarta-feira; omita se único externo em data/')
    p.add_argument('--data-dir', type=Path, default=Path('data'))
    p.add_argument('--run-root', type=Path, default=Path('models/e0/experiments'))
    p.add_argument('--reference-seed', type=int, default=42)
    p.add_argument('--chunksize', type=int, default=25000)
    p.add_argument('--check-duplicates', action='store_true', help='Consome RAM para hashes das linhas')
    p.add_argument('--output-dir', type=Path, default=Path('models/e4/preflight'))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if args.chunksize <= 0:
        p.error('--chunksize deve ser positivo')
    path = args.dataset if args.dataset else find_dataset(args.data_dir)
    if not path.is_file():
        p.error(f'Dataset não encontrado: {path}')
    run_dir = find_reference_run(args.run_root, args.reference_seed)
    result = audit(path, run_dir, chunksize=args.chunksize, do_duplicates=args.check_duplicates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / 'auditoria_dataset.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    LOG.info('Arquivo: %s | linhas: %d | labels: %s', path, result['auditoria']['n_linhas'],
             result['auditoria']['rotulos_originais_apos_strip'])
    LOG.info('Schema: %s | faltantes: %d | extras: %d', result['schema']['status'],
             len(result['schema']['features_faltantes']), len(result['schema']['features_extras']))
    LOG.info('Estado: %s | Relatório: %s', result['estado'], output)
    if result['estado'] != 'apto_a_validacao_semantica':
        LOG.warning('Auditoria com bloqueios; NÃO executar inferência ainda. Consulte o JSON.')
        raise SystemExit(2)


if __name__ == '__main__':
    main()
