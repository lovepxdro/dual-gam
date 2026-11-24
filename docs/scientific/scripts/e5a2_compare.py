"""Compara E5-A (BCE pós-Sigmoid) e E5-A2 (BCEWithLogits) em pares.

Usa apenas resultados persistidos. Recusa configurações/splits diferentes.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median


CONSTRAINTS = (
    'seeds', 'replicas', 'epochs', 'batch_size', 'max_train_samples',
    'val_samples', 'test_samples', 'eval_batch', 'device', 'epsilon',
    'threshold', 'init_from', 'checkpoint_selection', 'preflight_hash',
    'run_root', 'dataset',
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def key(record: dict) -> tuple:
    return (record['seed_e0'], record['replica'], record['braco'])


def compare_dirs(base: Path, logits: Path, output: Path) -> dict:
    original_cfg = load_json(base / 'config_e5a.json')
    new_cfg = load_json(logits / 'config_e5a2.json')
    if original_cfg.get('experiment') != 'E5-A' or new_cfg.get('experiment') != 'E5-A2':
        raise ValueError('Experimentos diferentes do esperado')
    if new_cfg.get('loss_mode') != 'logits':
        raise ValueError('E5-A2 não foi executado com loss_mode=logits')
    divergencias = [field for field in CONSTRAINTS if original_cfg.get(field) != new_cfg.get(field)]
    if divergencias:
        raise ValueError(f'Configurações incompatíveis: {divergencias}')
    orig_summary = load_json(base / 'resumo_e5a.json')
    new_summary = load_json(logits / 'resumo_e5a.json')
    for field in ('seeds', 'replicas', 'n_teste_comum_ddos', 'ids_teste_comum_sha256'):
        if orig_summary.get(field) != new_summary.get(field):
            raise ValueError(f'Divergência no resumo: {field}')
    records0 = {key(row): row for row in orig_summary['resultados']}
    records1 = {key(row): row for row in new_summary['resultados']}
    if (len(records0) != len(orig_summary['resultados']) or
        len(records1) != len(new_summary['resultados']) or
        set(records0) != set(records1)):
        raise ValueError('Combinações seed/replica/arquitetura incompletas ou duplicadas')
    rows = []
    for k in sorted(records0):
        r0, r1 = records0[k], records1[k]
        for field in ('seed_defender_transferencia', 'n_treino_ddos', 'n_teste_comum_ddos'):
            if r0[field] != r1[field]:
                raise ValueError(f'Par {k} divergiu em {field}')
        result = {
            'seed_e0': k[0], 'replica': k[1], 'braco': k[2],
            'asr_e5a_pct': 100 * r0['asr_alvo'],
            'asr_e5a2_pct': 100 * r1['asr_alvo'],
            'delta_asr_pp': 100 * (r1['asr_alvo'] - r0['asr_alvo']),
            'asr_transferencia_e5a_pct': 100 * r0['asr_transferencia'],
            'asr_transferencia_e5a2_pct': 100 * r1['asr_transferencia'],
            'delta_transferencia_pp': 100 * (r1['asr_transferencia'] - r0['asr_transferencia']),
            'diversidade_e5a': r0['diversidade_l2_alvo'],
            'diversidade_e5a2': r1['diversidade_l2_alvo'],
            'delta_diversidade': r1['diversidade_l2_alvo'] - r0['diversidade_l2_alvo'],
        }
        rows.append(result)
    # Diferença de diferenças: efeito da nova loss sobre a vantagem condicional A1-A0.
    keys = sorted({(r['seed_e0'], r['replica']) for r in rows})
    rows_by_key = {(r['seed_e0'], r['replica'], r['braco']): r for r in rows}
    diffs = []
    for seed, rep in keys:
        r0 = rows_by_key[(seed, rep, 'A0')]
        r1 = rows_by_key[(seed, rep, 'A1')]
        diffs.append({
            'seed_e0': seed, 'replica': rep,
            'vantagem_a1_e5a_pp': r1['asr_e5a_pct'] - r0['asr_e5a_pct'],
            'vantagem_a1_e5a2_pp': r1['asr_e5a2_pct'] - r0['asr_e5a2_pct'],
            'dif_em_dif_pp': r1['delta_asr_pp'] - r0['delta_asr_pp'],
        })
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('comparacao_pareada.csv', rows), ('vantagem_condicional.csv', diffs)]:
        with (output / name).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    result = {
        'experimento': 'E5-A2',
        'comparabilidade': 'aprovada',
        'n_pares_arquitetura_repeticao': len(rows),
        'n_pares_seed_repeticao': len(diffs),
        'delta_asr_pp_media_por_arquitetura': {
            arm: mean(r['delta_asr_pp'] for r in rows if r['braco'] == arm)
            for arm in ('A0', 'A1')
        },
        'delta_transferencia_pp_media_por_arquitetura': {
            arm: mean(r['delta_transferencia_pp'] for r in rows if r['braco'] == arm)
            for arm in ('A0', 'A1')
        },
        'dif_em_dif_pp_media': mean(r['dif_em_dif_pp'] for r in diffs),
        'dif_em_dif_pp_mediana': median(r['dif_em_dif_pp'] for r in diffs),
        'cautela': 'N=6 pares de seed/replica; estatística descritiva, sem inferência populacional',
        'configuracoes_iguais': list(CONSTRAINTS),
    }
    (output / 'comparacao_resumo.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir', type=Path, default=Path('models/e5/piloto'))
    p.add_argument('--logits-dir', type=Path, default=Path('models/e5/a2_logits'))
    p.add_argument('--output-dir', type=Path, default=Path('models/e5/a2_comparacao'))
    args = p.parse_args()
    print(json.dumps(compare_dirs(args.baseline_dir, args.logits_dir, args.output_dir),
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
