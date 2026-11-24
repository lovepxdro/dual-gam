"""E5-B: valida comparabilidade da campanha e consolida por seed e por par direcionado.

Não trata réplicas/pares como observações independentes na inferência.
Usa métricas já produzidas pelo avaliador; sem retreino/reavaliação.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

from e5b_campaign import ARMS, SEEDS, atomic_json, load_json


def rows_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def emit_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError('Não é possível salvar CSV vazio')
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def validate_and_index(targets: list[dict], transfers: list[dict], seeds: tuple[int,...], replicas: tuple[int,...]):
    t, x = {}, {}
    for r in targets:
        k = (int(r['seed']), int(r['replica']), r['braco'])
        if k in t:
            raise ValueError(f'Duplicata no alvo: {k}')
        if r['asr'] == '' or int(r['detectados_limpos']) == 0:
            raise ValueError(f'ASR de alvo indefinida: {k}')
        t[k] = r
    for r in transfers:
        k = (int(r['origem']), int(r['destino']), int(r['replica']), r['braco'])
        if k in x or k[0] == k[1]:
            raise ValueError(f'Duplicata/auto-transferência: {k}')
        if r['asr'] == '' or int(r['detectados_limpos']) == 0:
            raise ValueError(f'ASR de transferência indefinida: {k}')
        x[k] = r
    expected_t = {(s,r,a) for s in seeds for r in replicas for a in ARMS}
    expected_x = {(s,d,r,a) for s in seeds for d in seeds if d != s for r in replicas for a in ARMS}
    if set(t) != expected_t or set(x) != expected_x:
        raise ValueError(f'Campanha incompleta: alvo {len(t)}/{len(expected_t)}, transferência {len(x)}/{len(expected_x)}')
    for s in seeds:
        for rep in replicas:
            group = [t[(s,rep,arm)] for arm in ARMS]
            for field in ('n','ids_sha256','noise_seed','detectados_limpos'):
                if len({r[field] for r in group}) != 1:
                    raise ValueError(f'Comparação do alvo não pareada: {s}/{rep}/{field}')
    for s in seeds:
        for d in seeds:
            if s == d: continue
            for rep in replicas:
                group = [x[(s,d,rep,arm)] for arm in ARMS]
                for field in ('n','ids_sha256','noise_seed','detectados_limpos'):
                    if len({r[field] for r in group}) != 1:
                        raise ValueError(f'Transferência não pareada: {s}/{d}/{rep}/{field}')
                for arm in ARMS:
                    reciprocal = x[(d,s,rep,arm)]
                    for field in ('n','ids_sha256','noise_seed'):
                        if group[ARMS.index(arm)][field] != reciprocal[field]:
                            raise ValueError(f'Par test/test não é simétrico: {s}/{d}/{rep}/{arm}/{field}')
    return t, x


def sample_bootstrap_ci(seed_rows: list[dict], field: str, *, n_boot: int = 2000) -> list[float]:
    """IC descritivo sobre seeds E0; não é IC sobre dados CIC-IDS externos."""
    values = np.asarray([r[field] for r in seed_rows], dtype=np.float64)
    rng = np.random.default_rng(205)
    matrix = rng.integers(0, len(values), (n_boot, len(values)))
    estimates = values[matrix].mean(axis=1)
    return [float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))]


def analyze(root: Path, out: Path) -> dict:
    cfg = load_json(root / 'config_e5b.json')
    summary = load_json(root / 'resumo_e5b.json')
    if cfg.get('experiment') != 'E5-B' or summary.get('status') != 'avaliacao_concluida':
        raise ValueError('E5-B incompleto')
    seeds = tuple(cfg['seeds']); replicas = tuple(cfg['replicas'])
    if tuple(seeds) != SEEDS or cfg.get('arms') != list(ARMS) or summary['n_seeds'] != len(SEEDS):
        raise ValueError('Seeds/arquiteturas não correspondem ao plano E5-B')
    t, x = validate_and_index(rows_csv(root/'metricas_alvo.csv'),
                              rows_csv(root/'metricas_transferencia.csv'), seeds, replicas)
    out.mkdir(parents=True, exist_ok=True)
    detail = []
    seed_rows = []
    for seed in seeds:
        pairs = []
        for rep in replicas:
            row = {'seed':seed,'replica':rep}
            for arm in ARMS:
                item = t[(seed,rep,arm)]
                row[f'asr_{arm}_pct'] = float(item['asr'])*100
                row[f'diversidade_{arm}_l2'] = float(item['diversidade_l2'])
                outbound = [float(x[(seed,d,rep,arm)]['asr'])*100 for d in seeds if d != seed]
                row[f'transf_{arm}_pct'] = mean(outbound)
            row['delta_a1_a2_pp'] = row['asr_A1_pct']-row['asr_A2_pct']
            row['delta_a2_a0_pp'] = row['asr_A2_pct']-row['asr_A0_pct']
            row['delta_a1_a0_pp'] = row['asr_A1_pct']-row['asr_A0_pct']
            detail.append(row)
            pairs.append(row)
        row = {'seed':seed}
        for col in ('asr_A0_pct','asr_A1_pct','asr_A2_pct',
                    'transf_A0_pct','transf_A1_pct','transf_A2_pct',
                    'diversidade_A0_l2','diversidade_A1_l2','diversidade_A2_l2',
                    'delta_a1_a2_pp','delta_a2_a0_pp','delta_a1_a0_pp'):
            row[col]=mean(r[col] for r in pairs)
        row['replicas_a1_supera_a2']=sum(r['delta_a1_a2_pp']>0 for r in pairs)
        seed_rows.append(row)
    emit_csv(out/'comparacao_pareada_seed_replica.csv',detail)
    emit_csv(out/'resumo_por_seed.csv',seed_rows)
    values = [r['delta_a1_a2_pp'] for r in seed_rows]
    result={
        'experiment':'E5-B', 'validacao_pareamento':'aprovada',
        'n_seeds':len(seeds),'n_replicas':len(replicas),
        'n_treinamentos':len(t),'n_transferencias_direcionadas':len(x),
        'medias_por_seed':{
            arm:{
                'asr_alvo_pct':mean(s[f'asr_{arm}_pct'] for s in seed_rows),
                'asr_transferencia_pct':mean(s[f'transf_{arm}_pct'] for s in seed_rows),
                'diversidade_l2':mean(s[f'diversidade_{arm}_l2'] for s in seed_rows),
            } for arm in ARMS
        },
        'ablacao_condicionamento_a1_menos_a2':{
            'media_pp':mean(values), 'mediana_seeds_pp':median(values),
            'seeds_com_ganho':sum(v>0 for v in values),
            'seeds_com_empate':sum(v==0 for v in values),
            'ci_exploratorio_bootstrap_seeds_95_pp':sample_bootstrap_ci(seed_rows,'delta_a1_a2_pp'),
        },
        'ablacao_capacidade_a2_menos_a0':{
            'media_pp':mean(r['delta_a2_a0_pp'] for r in seed_rows),
            'seeds_com_ganho':sum(r['delta_a2_a0_pp']>0 for r in seed_rows),
        },
        'comparacao_a1_menos_a0':{'media_pp':mean(r['delta_a1_a0_pp'] for r in seed_rows)},
        'notas_metodologicas':[
            'Unidade de sumarizacao da incerteza: seed E0, com as 3 replicas agregadas',
            'IC por bootstrap somente descreve variabilidade entre seeds, nao generaliza a datasets independentes',
            'Transferencia calculada com intersecao test/test por par, 14 destinos por seed e por replica',
            'ASR condicional a amostras DDoS limpas corretamente detectadas pelo defensor avaliado',
            'A1 e A2 tem parametros quase iguais, mas arquiteturas internas nao identicas',
            'Perturbacoes ocorrem no espaco das 77 features padronizadas, sem demonstracao de realizabilidade em fluxos reais',
        ],
    }
    atomic_json(out/'analise_e5b.json',result)
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign-dir',type=Path,default=Path('models/e5/b_campanha'))
    p.add_argument('--output-dir',type=Path,default=Path('models/e5/b_analise'))
    args=p.parse_args(argv)
    analyze(args.campaign_dir,args.output_dir)


if __name__=='__main__':main()
