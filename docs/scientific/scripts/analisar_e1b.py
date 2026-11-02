#!/usr/bin/env python3
"""Consolida os 15 resultados pareados do E1-B.

Exemplos na raiz do repositório:
  uv run python docs/scientific/scripts/analisar_e1b.py \
      --input models/e1/comparacoes --output docs/scientific/e1/e1_assets
  uv run python docs/scientific/scripts/analisar_e1b.py \
      --input comparacoes.zip --output docs/scientific/e1/e1_assets

Bootstrap: reamostra as 15 seeds completas, mantendo quatro rodadas por seed.
"""
import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEEDS = [*range(1, 11), 20, 40, 42, 80, 100]
ROUNDS = [1, 3, 10, 20]
RATE_COLS = ['evasao_pre', 'evasao_adversarial', 'evasao_convencional',
             'fpr_adversarial', 'fpr_convencional', 'f1_adversarial',
             'f1_convencional', 'recall_adversarial', 'recall_convencional']


def read_records(source: Path):
    if source.is_file() and source.suffix == '.zip':
        with zipfile.ZipFile(source) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f'ZIP inválido: {bad}')
            paths = [name for name in z.namelist() if re.search(r'/seed-\d+/resultados\.json$', name)]
            if len(paths) != len(SEEDS):
                raise ValueError(f'Esperados 15 resultados.json; encontrados {len(paths)}')
            pairs = [(path, json.loads(z.read(path)), pd.read_csv(io.BytesIO(z.read(path[:-4] + 'csv')))) for path in paths]
    elif source.is_dir():
        paths = sorted(source.glob('seed-*/resultados.json'))
        pairs = [(str(path), json.loads(path.read_text()), pd.read_csv(path.with_suffix('.csv'))) for path in paths]
    else:
        raise FileNotFoundError(source)
    records = []
    for path, doc, csv in pairs:
        seed = int(re.search(r'seed-(\d+)', path).group(1))
        if seed != doc['seed'] or doc['rounds'] != ROUNDS or len(doc['results']) != 4:
            raise ValueError(f'Metadados inconsistentes: {path}')
        if doc.get('evaluation') != 'validação offline; teste reservado não utilizado':
            raise ValueError(f'Protocolo de avaliação inesperado: {path}')
        for rec in doc['results']:
            selected = csv[csv.rodada == rec['rodada']]
            if len(selected) != 1 or rec['seed'] != seed:
                raise ValueError(f'Rodada/seed não conferem: {path}')
            for key, value in rec.items():
                if not np.isclose(value, selected.iloc[0][key], atol=1e-7):
                    raise ValueError(f'JSON e CSV divergentes em {path}: {key}')
            if not np.isclose((rec['evasao_convencional']-rec['evasao_adversarial'])*100,
                              rec['beneficio_adversarial_pp'], atol=1e-6):
                raise ValueError(f'Benefício divergente: {path}')
            if rec['passos_por_braco'] != 60 or rec['epochs'] != 3:
                raise ValueError(f'Orçamento desigual/inconsistente: {path}')
            records.append(rec)
    df = pd.DataFrame(records).sort_values(['seed', 'rodada']).reset_index(drop=True)
    if len(df) != 60 or sorted(df.seed.unique().tolist()) != SEEDS or sorted(df.rodada.unique().tolist()) != ROUNDS:
        raise ValueError('Incompleta: esperadas 15 seeds, 4 rodadas e 60 pares')
    if df.duplicated(['seed', 'rodada']).any() or df.isna().any().any():
        raise ValueError('Pares duplicados ou métricas ausentes')
    return df


def ci_seed(df, column, rounds=None, reps=40000, seed=2026):
    rng = np.random.default_rng(seed)
    selected = df if rounds is None else df[df.rodada.isin(rounds)]
    arr = np.stack([selected[selected.seed == s].sort_values('rodada')[column].to_numpy() for s in SEEDS])
    ix = rng.integers(0, len(SEEDS), size=(reps, len(SEEDS)))
    means = arr[ix].mean(axis=(1,2))
    return [float(q) for q in np.quantile(means, [.025, .975])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, default=Path('docs/scientific/e1/e1_assets'))
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    df = read_records(args.input)
    df['evasao_pre_pp'] = df.evasao_pre * 100
    df['evasao_adv_pp'] = df.evasao_adversarial * 100
    df['evasao_conv_pp'] = df.evasao_convencional * 100
    df['fpr_adv_pp'] = df.fpr_adversarial * 100
    df['fpr_conv_pp'] = df.fpr_convencional * 100
    df['delta_fpr_pp'] = (df.fpr_adversarial - df.fpr_convencional)*100
    df['delta_f1_pp'] = (df.f1_adversarial - df.f1_convencional)*100
    df['delta_recall_pp'] = (df.recall_adversarial - df.recall_convencional)*100
    df.to_csv(out/'e1b_comparacoes_60.csv', index=False, float_format='%.10f')

    rows=[]
    for r, g in df.groupby('rodada'):
        rows.append(dict(rodada=int(r), n_pares=len(g),
            evasao_pre_media_pct=g.evasao_pre_pp.mean(),
            evasao_adv_media_pct=g.evasao_adv_pp.mean(),
            evasao_conv_media_pct=g.evasao_conv_pp.mean(),
            beneficio_medio_pp=g.beneficio_adversarial_pp.mean(),
            beneficio_mediana_pp=g.beneficio_adversarial_pp.median(),
            beneficio_ic95_seed_min_pp=ci_seed(df, 'beneficio_adversarial_pp',[r])[0],
            beneficio_ic95_seed_max_pp=ci_seed(df, 'beneficio_adversarial_pp',[r])[1],
            vit_adv=int((g.beneficio_adversarial_pp > 1e-6).sum()),
            empate=int((g.beneficio_adversarial_pp.abs()<=1e-6).sum()),
            delta_fpr_medio_pp=g.delta_fpr_pp.mean(),
            delta_f1_medio_pp=g.delta_f1_pp.mean(),
            delta_recall_medio_pp=g.delta_recall_pp.mean()))
    by_round=pd.DataFrame(rows)
    by_round.to_csv(out/'e1b_resumo_por_rodada.csv', index=False, float_format='%.8f')
    by_seed=df.groupby('seed',as_index=False).agg(
        beneficio_medio_pp=('beneficio_adversarial_pp','mean'),
        beneficio_min_pp=('beneficio_adversarial_pp','min'),
        beneficio_max_pp=('beneficio_adversarial_pp','max'),
        delta_fpr_medio_pp=('delta_fpr_pp','mean'),
        delta_f1_medio_pp=('delta_f1_pp','mean'),
        delta_recall_medio_pp=('delta_recall_pp','mean'))
    by_seed.to_csv(out/'e1b_resumo_por_seed.csv', index=False, float_format='%.8f')
    summary={
        'experiment':'E1-B', 'n_seeds':15, 'n_rounds_per_seed':4, 'n_paired_comparisons':60,
        'rounds':ROUNDS,
        'benefit_mean_pp':float(df.beneficio_adversarial_pp.mean()),
        'benefit_median_pp':float(df.beneficio_adversarial_pp.median()),
        'benefit_seed_cluster_bootstrap_95_pp':ci_seed(df,'beneficio_adversarial_pp'),
        'adv_wins':int((df.beneficio_adversarial_pp>1e-6).sum()),
        'ties':int((df.beneficio_adversarial_pp.abs()<=1e-6).sum()),
        'conv_wins':int((df.beneficio_adversarial_pp < -1e-6).sum()),
        'mean_evasion_pre_pct':float(df.evasao_pre_pp.mean()),
        'mean_evasion_adv_pct':float(df.evasao_adv_pp.mean()),
        'mean_evasion_conv_pct':float(df.evasao_conv_pp.mean()),
        'delta_fpr_adv_minus_conv_pp':float(df.delta_fpr_pp.mean()),
        'delta_fpr_seed_cluster_bootstrap_95_pp':ci_seed(df,'delta_fpr_pp'),
        'delta_f1_adv_minus_conv_pp':float(df.delta_f1_pp.mean()),
        'delta_f1_seed_cluster_bootstrap_95_pp':ci_seed(df,'delta_f1_pp'),
        'delta_recall_adv_minus_conv_pp':float(df.delta_recall_pp.mean()),
        'adv_higher_fpr':int((df.delta_fpr_pp>1e-9).sum()),
        'conv_higher_fpr':int((df.delta_fpr_pp<-1e-9).sum()),
        'adv_lower_f1':int((df.delta_f1_pp<-1e-9).sum()),
        'conv_lower_f1':int((df.delta_f1_pp>1e-9).sum()),
        'seed_mean_benefit_all_positive':bool((by_seed.beneficio_medio_pp>0).all()),
        'note':'Bootstrap descritivo com seed como unidade; rodadas dentro da seed não são independentes. Somente validação offline do mesmo A_r.'
    }
    (out/'e1b_estatisticas.json').write_text(json.dumps(summary,ensure_ascii=False, indent=2)+'\n',encoding='utf-8')

    plt.rcParams.update({'font.size':10, 'axes.titlesize':12})
    fig, ax=plt.subplots(figsize=(8.3,4.5), dpi=150)
    x=np.arange(4)
    for i,col in enumerate(['evasao_pre_media_pct','evasao_adv_media_pct','evasao_conv_media_pct']):
        ax.bar(x + (i-1)*.24, by_round[col], .22, label=['Antes','Adversarial','Convencional'][i])
    ax.set_xticks(x, [f'R{r}' for r in ROUNDS]);ax.set_ylabel('Evasão média (%)')
    ax.set_title('E1-B | Evasão em validação, por rodada (n=15 seeds)');ax.legend();ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(out/'05_e1b_evasao_por_rodada.png');plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.3,4.5),dpi=150)
    y=by_round.beneficio_medio_pp.to_numpy(); low=y-by_round.beneficio_ic95_seed_min_pp.to_numpy(); high=by_round.beneficio_ic95_seed_max_pp.to_numpy()-y
    ax.errorbar(np.arange(4), y, yerr=np.stack([low, high]), capsize=6, linestyle='-', marker='o')
    for i,v in enumerate(y):ax.annotate(f'{v:.2f} pp', (i,v), xytext=(0,10),textcoords='offset points',ha='center')
    ax.set_xticks(np.arange(4), [f'R{r}' for r in ROUNDS]);ax.set_ylabel('Convencional − adversarial (pp)');ax.set_ylim(bottom=0)
    ax.set_title('E1-B | Benefício adversarial (IC 95% bootstrap por seed)');ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(out/'06_e1b_beneficio_ic.png');plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.5), dpi=150)
    ax.bar(by_seed.seed.astype(str),by_seed.beneficio_medio_pp)
    ax.set_title('E1-B | Benefício médio nas quatro rodadas, por seed');ax.set_ylabel('Benefício médio (pp)');ax.set_xlabel('Seed');ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(out/'07_e1b_beneficio_por_seed.png');plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.3,4.5),dpi=150)
    rr=df.groupby('rodada')[['fpr_adv_pp','fpr_conv_pp']].mean();x=np.arange(4)
    ax.bar(x-.19,rr.fpr_adv_pp,.37,label='Adversarial');ax.bar(x+.19,rr.fpr_conv_pp,.37,label='Convencional')
    ax.set_xticks(x,[f'R{r}' for r in ROUNDS]);ax.set_ylabel('FPR médio na validação (%)')
    ax.set_title('E1-B | FPR de validação: tratamento e controle');ax.legend();ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(out/'08_e1b_fpr_por_rodada.png');plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.3,4.8),dpi=150)
    for r,g in df.groupby('rodada'):
        ax.scatter(g.delta_fpr_pp, g.beneficio_adversarial_pp,alpha=.75,label=f'R{r}',s=32)
    ax.axvline(0,linestyle='--',alpha=.6);ax.axhline(0,linestyle='--',alpha=.6)
    ax.set_xlabel('Δ FPR = adversarial − convencional (pp)')
    ax.set_ylabel('Benefício = evasão conv. − adv. (pp)')
    ax.set_title('E1-B | Benefício de robustez e custo de FPR');ax.legend();ax.grid(alpha=.23)
    fig.tight_layout();fig.savefig(out/'09_e1b_tradeoff.png');plt.close(fig)

    print('E1-B validado: 15 seeds, 60 pares; benefício médio',round(summary['benefit_mean_pp'],3),'pp')
    print('Saída:',out)

if __name__=='__main__': main()
