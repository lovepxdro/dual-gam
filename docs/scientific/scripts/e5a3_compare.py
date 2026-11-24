"""Compara A0(z), A1(x,z) e A2(z) usando os mesmos pares E5-A2/E5-A3.

A0/A1 vêm do E5-A2; A2 vem do E5-A3. Não carrega checkpoints
nem consulta treino/teste; verifica comparabilidade pelos metadados.
"""
from __future__ import annotations

import argparse
import hashlib
import csv
import json
from pathlib import Path
from statistics import mean, median

from e5a3_capacity import COMPARABLE, checked_baseline, load_json


def compare_dirs(baseline_dir: Path, capacity_dir: Path, output_dir: Path) -> dict:
    a3_cfg = load_json(capacity_dir / 'config_e5a3.json')
    if a3_cfg.get('experiment') != 'E5-A3' or a3_cfg.get('loss_mode') != 'logits' or not a3_cfg.get('a2_ignora_x'):
        raise ValueError('Configuração E5-A3 inválida')
    _, original = checked_baseline(baseline_dir, a3_cfg)
    if a3_cfg['base_e5a2_config_sha256'] != hashlib.sha256(
            (baseline_dir / 'config_e5a2.json').read_bytes()).hexdigest():
        raise ValueError('Hash do config E5-A2 divergiu')
    a3 = load_json(capacity_dir / 'resumo_e5a3.json')
    if a3.get('experiment') != 'E5-A3' or a3.get('status') != 'treinamentos_concluidos':
        raise ValueError('Resumo E5-A3 incompleto')
    for field in ('seeds', 'replicas', 'n_teste_comum_ddos', 'ids_teste_comum_sha256'):
        if original.get(field) != a3.get(field):
            raise ValueError(f'Divergência entre conjuntos: {field}')
    r01 = {(r['seed_e0'], r['replica'], r['braco']): r for r in original['resultados']}
    r2_rows = a3.get('resultados', [])
    r2 = {(r['seed_e0'], r['replica'], r['braco']): r for r in r2_rows}
    expected_pairs = {(s, rep) for s in a3_cfg['seeds'] for rep in a3_cfg['replicas']}
    if (len(r2_rows) != len(expected_pairs) or
        set(r2) != {(s, rep, 'A2') for s, rep in expected_pairs}):
        raise ValueError('E5-A3 incompleto, duplicado ou com braço incorreto')
    if any(arm not in r01 for s, rep in expected_pairs for arm in [(s, rep, 'A0'), (s, rep, 'A1')]):
        raise ValueError('E5-A2 não possui todos os pares')
    rows = []
    for seed, rep in sorted(expected_pairs):
        a0, a1, a2 = (r01[(seed, rep, 'A0')], r01[(seed, rep, 'A1')], r2[(seed, rep, 'A2')])
        for name in ('seed_defender_transferencia', 'n_treino_ddos', 'n_teste_comum_ddos'):
            if len({a0[name], a1[name], a2[name]}) != 1:
                raise ValueError(f'Metadados divergentes em {seed}/{rep}: {name}')
        if any(r.get('asr_alvo') is None or r.get('asr_transferencia') is None
               for r in (a0, a1, a2)):
            raise ValueError(f'ASR indefinida (nenhum DDoS limpo detectado) em {seed}/{rep}')
        row = {'seed_e0': seed, 'replica': rep}
        for arm, record in [('A0', a0), ('A1', a1), ('A2', a2)]:
            row[f'asr_{arm}_pct'] = 100 * record['asr_alvo']
            row[f'transferencia_{arm}_pct'] = 100 * record['asr_transferencia']
            row[f'diversidade_{arm}_l2'] = record['diversidade_l2_alvo']
            row[f'tempo_{arm}_s'] = record['tempo_treino_segundos']
        row['delta_a1_a2_pp'] = row['asr_A1_pct'] - row['asr_A2_pct']
        row['delta_a2_a0_pp'] = row['asr_A2_pct'] - row['asr_A0_pct']
        row['delta_a1_a0_pp'] = row['asr_A1_pct'] - row['asr_A0_pct']
        rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / 'comparacao_e5a3_pareada.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        'experimento': 'E5-A3', 'comparabilidade': 'aprovada',
        'n_pares_seed_replica': len(rows), 'n_seeds_independentes': len(a3_cfg['seeds']),
        'arquiteturas': {
            'A0': 'ruído puro, 64013 parâmetros',
            'A1': 'condicional x+z, 104538 parâmetros',
            'A2': 'ruído puro, residual z-only 153x153, 104482 parâmetros',
        },
        'media_asr_pct': {arm: mean(row[f'asr_{arm}_pct'] for row in rows) for arm in ('A0','A1','A2')},
        'media_transferencia_pct': {arm: mean(row[f'transferencia_{arm}_pct'] for row in rows) for arm in ('A0','A1','A2')},
        'media_diversidade_l2': {arm: mean(row[f'diversidade_{arm}_l2'] for row in rows) for arm in ('A0','A1','A2')},
        'vantagem_condicionamento_a1_menos_a2_pp': {
            'media': mean(row['delta_a1_a2_pp'] for row in rows),
            'mediana': median(row['delta_a1_a2_pp'] for row in rows),
            'pares_a1_supera_a2': sum(row['delta_a1_a2_pp'] > 0 for row in rows),
            'pares_empate': sum(row['delta_a1_a2_pp'] == 0 for row in rows),
        },
        'ganho_capacidade_a2_menos_a0_pp': {
            'media': mean(row['delta_a2_a0_pp'] for row in rows),
            'pares_a2_supera_a0': sum(row['delta_a2_a0_pp'] > 0 for row in rows),
        },
        'interpretacao_limite': 'Comparação controlada em 2 seeds x 3 réplicas; A2 possui quase mesmo número de parâmetros, mas difere na largura/capacidade funcional. Não isola causalidade perfeitamente. Não inferir significância populacional de 6 pares.',
        'campos_comparados': list(COMPARABLE),
    }
    (output_dir / 'resumo_comparacao_e5a3.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-dir', type=Path, default=Path('models/e5/a2_logits'))
    p.add_argument('--capacity-dir', type=Path, default=Path('models/e5/a3_capacidade'))
    p.add_argument('--output-dir', type=Path, default=Path('models/e5/a3_comparacao'))
    args = p.parse_args(argv)
    print(json.dumps(compare_dirs(args.baseline_dir, args.capacity_dir, args.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
