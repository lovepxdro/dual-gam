#!/usr/bin/env python3
"""Analise E1-A a partir de ZIPs dos runs E0, sem retraining.

Uso: python analisar_e1a.py --runs-dir /caminho/com/zips --out-dir /caminho/saida
"""
import argparse
import csv
import io
import json
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import matplotlib.pyplot as plt

SEEDS = [*range(1, 11), 20, 40, 42, 80, 100]
RODADAS = range(1, 21)

def load_json(z: ZipFile, suffix: str):
    matches = [n for n in z.namelist() if n.endswith('/' + suffix)]
    if len(matches) != 1:
        raise ValueError(f'Esperado arquivo unico {suffix}, encontrados {len(matches)}')
    return json.loads(z.read(matches[0]))

def load_csv(z: ZipFile, suffix: str):
    matches = [n for n in z.namelist() if n.endswith('/' + suffix)]
    if len(matches) != 1:
        raise ValueError(f'Esperado CSV unico {suffix}, encontrados {len(matches)}')
    return list(csv.DictReader(io.StringIO(z.read(matches[0]).decode('utf-8-sig'))))

def save_csv(path, rows):
    if not rows:
        raise ValueError('CSV sem linhas')
    with open(path, 'w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def bootstrap_seed_means(vals, iterations=10000):
    """IC bootstrap usando seed como unidade experimental."""
    vals = np.array(vals, dtype=float)
    rng = np.random.default_rng(20261020)
    samples = rng.choice(vals, size=(iterations, len(vals)), replace=True).mean(axis=1)
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    immediate=[]
    historical=[]
    fpr=[]
    seed_rows=[]
    for seed in SEEDS:
        matches = list(args.runs_dir.glob(f'run_*_seed{seed}.zip'))
        if len(matches) != 1:
            raise ValueError(f'Seed {seed}: esperado um ZIP, encontrados {len(matches)}')
        with ZipFile(matches[0]) as z:
            if z.testzip() is not None:
                raise ValueError(f'ZIP corrompido: {matches[0]}')
            history=load_json(z,'historico_treino.json')
            matrix=load_json(z,'matriz_checkpoints.json')
            summary=load_json(z,'summary.json')
            defender_metrics=load_csv(z,'metrics/defender_metrics.csv')
            if summary['seed']!=seed:
                raise ValueError(f'Conflito de seed em {matches[0]}')
            mat=np.array(matrix['matriz_evasao'], dtype=float)
            if mat.shape != (20,21):
                raise ValueError(f'Matriz invalida seed {seed}: {mat.shape}')
            pre=np.array(history['taxa_evasao_pre_adaptacao'], dtype=float)*100
            post=np.array(history['taxa_evasao_pos_adaptacao'], dtype=float)*100
            if len(pre)!=20 or len(post)!=20 or len(defender_metrics)!=20:
                raise ValueError(f'Historico incompleto seed {seed}')
            for idx, r in enumerate(RODADAS):
                # Matriz em frações: Atacante A_r contra D_(r-1) e D_r
                if abs(mat[idx,r-1]*100 - pre[idx]) > 1e-4 or abs(mat[idx,r]*100 - post[idx]) > 1e-4:
                    raise ValueError(f'Matriz inconsistente seed={seed} rodada={r}')
                immediate.append({
                    'seed':seed, 'rodada':r,
                    'evasao_pre_pp':float(pre[idx]),
                    'evasao_pos_pp':float(post[idx]),
                    'ganho_pp':float(pre[idx]-post[idx])
                })
                dm=defender_metrics[idx]
                fpr.append({'seed':seed,'rodada':r,'fpr_validacao_pp':float(dm['fpr'])*100,
                            'f1_validacao_pct':float(dm['f1'])*100,
                            'recall_validacao_pct':float(dm['recall'])*100})
                # E(A_r, D_j), checkpoint atual ou posterior
                for j in range(r,21):
                    historical.append({
                        'seed':seed,'attacker':r, 'defender':j,
                        'evasao_pp':float(mat[r-1,j]*100),
                        'delta_vs_adaptacao_pp':float((mat[r-1,j]-mat[r-1,r])*100)
                    })
            historical_final=mat[:,20]*100
            drift=(mat[:,20] - np.diag(mat[:,1:]))*100
            seed_rows.append({
                'seed':seed,
                'ganho_medio_pp':float(np.mean(pre-post)),
                'ganho_mediano_pp':float(np.median(pre-post)),
                'pos_media_pp':float(np.mean(post)),
                'ganhos_positivos_rodadas':int(np.sum(pre-post>1e-6)),
                'ganhos_negativos_rodadas':int(np.sum(pre-post < -1e-6)),
                'a3_d3_pp':float(mat[2,3]*100),
                'a3_d20_pp':float(mat[2,20]*100),
                'deriva_a3_pp':float((mat[2,20]-mat[2,3])*100),
                'a3_pior_historico':int(np.argmax(historical_final) == 2),
                'pior_historico_attacker':int(np.argmax(historical_final)+1),
                'fpr_teste_final_pp':float(summary['teste_final']['fpr']*100),
                'deriva_historica_media_pp':float(np.mean(drift)),
                'historicos_deriva_positiva':int(np.sum(drift > 1e-6))
            })
    save_csv(args.out_dir/'ganho_imediato_por_rodada.csv', immediate)
    save_csv(args.out_dir/'trajetorias_historicas.csv', historical)
    save_csv(args.out_dir/'fpr_validacao_por_rodada.csv',fpr)
    save_csv(args.out_dir/'resumo_por_seed.csv',seed_rows)
    # indicadores e ICs somente no nivel de seed
    im=np.array([[q['ganho_pp'] for q in immediate if q['seed']==s] for s in SEEDS]);
    post=np.array([[q['evasao_pos_pp'] for q in immediate if q['seed']==s] for s in SEEDS]);
    pre=np.array([[q['evasao_pre_pp'] for q in immediate if q['seed']==s] for s in SEEDS]);
    drift_a3=np.array([q['deriva_a3_pp'] for q in seed_rows]);
    a3_d20=np.array([q['a3_d20_pp'] for q in seed_rows]);
    fpr_teste=np.array([q['fpr_teste_final_pp'] for q in seed_rows]);
    all_rows=im.size
    stats={
        'seeds':SEEDS,
        'total_rodadas':all_rows,
        'ganho_media_seed_pp':float(np.mean(im.mean(axis=1))),
        'ganho_ic95_bootstrap_seeds_pp':bootstrap_seed_means(im.mean(axis=1)),
        'ganho_positivo':int(np.sum(im>1e-6)),
        'ganho_nulo':int(np.sum(abs(im)<=1e-6)),
        'ganho_negativo':int(np.sum(im < -1e-6)),
        'pos_menor_1pp':int(np.sum(post<1)),
        'pre_maior_igual_5pp':int(np.sum(pre>=5)),
        'pre_5_pos_menor_1':int(np.sum((pre>=5)&(post<1))),
        'pre_media_rodada_1_pp':float(pre[:,0].mean()),
        'pre_media_rodada_20_pp':float(pre[:,-1].mean()),
        'pre_media_rodadas_1_5_pp':float(pre[:,:5].mean()),
        'pre_media_rodadas_16_20_pp':float(pre[:,-5:].mean()),
        'deriva_a3_positiva_seeds':int(np.sum(drift_a3>1e-6)),
        'deriva_a3_media_pp':float(drift_a3.mean()),
        'deriva_a3_mediana_pp':float(np.median(drift_a3)),
        'deriva_a3_ic95_bootstrap_seeds_pp':bootstrap_seed_means(drift_a3),
        'a3_pior_historico_seeds':int(sum(q['a3_pior_historico'] for q in seed_rows)),
        'a3_d3_media_pp':float(np.mean([q['a3_d3_pp'] for q in seed_rows])),
        'a3_d20_media_pp':float(a3_d20.mean()),
        'fpr_teste_media_pp':float(fpr_teste.mean()),
        'fpr_teste_min_pp':float(fpr_teste.min()),
        'fpr_teste_max_pp':float(fpr_teste.max())
    }
    (args.out_dir/'estatisticas_e1a.json').write_text(json.dumps(stats,indent=2,ensure_ascii=False), encoding='utf-8')

    rounds=np.array(list(RODADAS));
    fig,ax=plt.subplots(figsize=(9,4.6))
    means=im.mean(axis=0); stdev=im.std(axis=0,ddof=1)
    ax.plot(rounds,means,marker='o',markersize=3,label='Ganho médio')
    ax.fill_between(rounds,means-stdev,means+stdev,alpha=.2,label='± 1 desvio padrão entre seeds')
    ax.set(xlabel='Rodada',ylabel='Redução de evasão (pontos percentuais)',title='E1-A · Ganho de adaptação imediato por rodada')
    ax.set_xticks(rounds[::2]); ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(args.out_dir/'01_ganho_imediato.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.6))
    ax.plot(rounds,pre.mean(axis=0),marker='o',markersize=3,label='Pré-adaptação')
    ax.plot(rounds,post.mean(axis=0),marker='o',markersize=3,label='Pós-adaptação')
    ax.set(xlabel='Rodada',ylabel='Evasão média (%)',title='E1-A · Evasão antes e após atualização')
    ax.set_xticks(rounds[::2]);ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(args.out_dir/'02_evasao_pre_pos.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.6))
    for seed in SEEDS:
        pts=[r['evasao_pp'] for r in historical if r['seed']==seed and r['attacker']==3]
        ax.plot(np.arange(3,21),pts,alpha=.3,linewidth=1)
    a3_mean=np.array([[r['evasao_pp'] for r in historical if r['seed']==s and r['attacker']==3] for s in SEEDS]).mean(axis=0)
    ax.plot(np.arange(3,21),a3_mean,linewidth=2.5,label='Média de 15 seeds')
    ax.set(xlabel='Checkpoint do Defender',ylabel='Evasão (%)',title='E1-A · A3 contra checkpoints D3–D20')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(args.out_dir/'03_trajetoria_a3.png',dpi=170);plt.close(fig)

    fpr_mat=np.array([[r['fpr_validacao_pp'] for r in fpr if r['seed']==seed] for seed in SEEDS]);
    fig,ax=plt.subplots(figsize=(9,4.6))
    for seed in [6,80,100]:
        ax.plot(rounds, fpr_mat[SEEDS.index(seed)],marker='o',markersize=3,label=f'Seed {seed}')
    others=np.array([fpr_mat[i] for i,s in enumerate(SEEDS) if s not in [6,80,100]])
    ax.plot(rounds,others.mean(axis=0),linestyle='--',linewidth=2,label='Média das outras 12')
    ax.set(xlabel='Rodada',ylabel='FPR de validação (%)',title='E1-A · FPR ao longo do treino (exploratório)')
    ax.set_xticks(rounds[::2]);ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(args.out_dir/'04_fpr_validacao.png',dpi=170);plt.close(fig)

    print(json.dumps(stats,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
