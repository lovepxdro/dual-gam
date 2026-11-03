#!/usr/bin/env python3
"""E2-A — transferência temporal retrospectiva, sem executar modelos.

Aceita diretórios de runs da ADArena ou arquivos run_*_seedN.zip.
Todos os percentuais exportados são apresentados em pontos percentuais (0–100).
"""
from __future__ import annotations

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

EXPECTED_SEEDS = {1,2,3,4,5,6,7,8,9,10,20,40,42,80,100}
N_ATTACKERS = 20
N_DEFENDERS = 21
BOOTSTRAPS = 5000


def sources_from_inputs(inputs: list[str]) -> list[Path]:
    import glob
    candidates = []
    for spec in inputs:
        paths = [Path(p) for p in glob.glob(spec)] or [Path(spec)]
        for path in paths:
            if path.is_file() and path.suffix.lower() == '.zip':
                candidates.append(path)
            elif path.is_dir():
                if (path / 'matriz_checkpoints.json').exists():
                    candidates.append(path)
                else:
                    candidates.extend(sorted(path.glob('run_*_seed*/')))
                    candidates.extend(sorted(path.glob('run_*_seed*.zip')))
            else:
                raise FileNotFoundError(f'Fonte não encontrada ou não suportada: {path}')
    return list(dict.fromkeys(candidates))


def _seed_from_name(path: Path) -> int:
    match = re.search(r'_seed(\d+)', path.name)
    if not match:
        raise ValueError(f'Seed não encontrada no nome: {path}')
    return int(match.group(1))


def read_source(path: Path) -> tuple[int, dict, dict, pd.DataFrame]:
    seed = _seed_from_name(path)
    if path.suffix.lower() == '.zip':
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f'ZIP corrompido: {path} (entrada {bad})')
            def load(suffix):
                matches = [n for n in z.namelist() if n.endswith('/' + suffix)]
                if len(matches) != 1:
                    raise ValueError(f'{path}: esperado 1 arquivo {suffix}, encontrados {len(matches)}')
                return z.read(matches[0])
            matrix = json.loads(load('matriz_checkpoints.json'))
            history = json.loads(load('historico_treino.json'))
            defender_metrics = pd.read_csv(io.BytesIO(load('metrics/defender_metrics.csv')))
    else:
        matrix = json.loads((path/'matriz_checkpoints.json').read_text(encoding='utf-8'))
        history = json.loads((path/'historico_treino.json').read_text(encoding='utf-8'))
        defender_metrics = pd.read_csv(path/'metrics/defender_metrics.csv')
    return seed, matrix, history, defender_metrics


def validate_matrix(seed: int, matrix: dict, history: dict) -> np.ndarray:
    labels_a = list(matrix['linhas_atacantes'])
    labels_d = list(matrix['colunas_defensores'])
    if labels_a != list(range(1, N_ATTACKERS+1)):
        raise ValueError(f'Seed {seed}: índice de Attackers inconsistente')
    if labels_d != list(range(N_DEFENDERS)):
        raise ValueError(f'Seed {seed}: índice de Defenders inconsistente')
    values = np.array(matrix['matriz_evasao'], dtype=float)
    if values.shape != (N_ATTACKERS, N_DEFENDERS):
        raise ValueError(f'Seed {seed}: shape inválido {values.shape}')
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError(f'Seed {seed}: valores devem ser frações entre 0 e 1')
    pre = np.asarray(history['taxa_evasao'], dtype=float)
    if pre.shape != (N_ATTACKERS,):
        raise ValueError(f'Seed {seed}: histórico pré com tamanho inválido')
    for j in range(1, N_ATTACKERS+1):
        if not np.isclose(values[j-1,j-1], pre[j-1], atol=2e-7):
            raise ValueError(f'Seed {seed}: matriz e histórico pré inconsistentes na rodada {j}')
    return values


def make_pairs(seeds: list[tuple[int, np.ndarray]]) -> pd.DataFrame:
    records = []
    for seed, mat in seeds:
        for j in range(1, N_ATTACKERS+1):
            for k in range(0, N_DEFENDERS):
                records.append({
                    'seed': seed, 'atacante': j, 'defensor': k, 'distancia': j-k,
                    'evasao_pct': 100*mat[j-1,k],
                    'evasao_alvo_contemporaneo_pct': 100*mat[j-1,j-1],
                    'atacante_efetivo_5pct': mat[j-1,j-1] >= 0.05,
                    'tipo': ('futuro_nao_adjacente' if 1<=k<=19 and j>=k+2 else
                             'futuro_adjacente' if 1<=k<=19 and j==k+1 else
                             'controle_D0' if k==0 else
                             'historico' if j<=k else 'outro')
                })
    return pd.DataFrame.from_records(records)


def seed_bootstrap_ci(values: pd.Series | np.ndarray, rng: np.random.Generator, B: int=BOOTSTRAPS):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return (float('nan'), float('nan'))
    idx = rng.integers(0, len(a), size=(B, len(a)))
    dist = a[idx].mean(axis=1)
    return tuple(np.percentile(dist, [2.5,97.5]).tolist())


def summarise(df: pd.DataFrame, rng: np.random.Generator):
    main = df.query("tipo == 'futuro_nao_adjacente'").copy()
    main['distancia_grupo'] = pd.cut(main['distancia'],bins=[1,3,6,20],labels=['2–3','4–6','7–19'])
    by_seed = main.groupby('seed').agg(
        n=('evasao_pct','size'), evasao_media_pct=('evasao_pct','mean'),
        evasao_mediana_pct=('evasao_pct','median'),
        pct_evasao_abaixo_1=('evasao_pct',lambda s: 100*(s<1).mean()),
        pct_evasao_5_ou_mais=('evasao_pct',lambda s: 100*(s>=5).mean()),
        pct_atacante_efetivo_5=('atacante_efetivo_5pct',lambda s:100*s.mean())
    ).reset_index()
    by_k = main.groupby(['seed','defensor']).agg(
        n=('evasao_pct','size'), evasao_media_pct=('evasao_pct','mean'),
        evasao_mediana_pct=('evasao_pct','median'),
        pct_abaixo_1=('evasao_pct',lambda s:100*(s<1).mean())
    ).reset_index()
    by_gap = main.groupby(['seed','distancia_grupo'],observed=True).agg(
        n=('evasao_pct','size'),evasao_media_pct=('evasao_pct','mean'),
        pct_abaixo_1=('evasao_pct',lambda s:100*(s<1).mean())
    ).reset_index()
    eff = main[main['atacante_efetivo_5pct']].copy()
    per_seed_eff = eff.groupby('seed').agg(
        n=('evasao_pct','size'),ev_future=('evasao_pct','mean'),
        ev_target=('evasao_alvo_contemporaneo_pct','mean'),
        pct_abaixo_1=('evasao_pct',lambda s:100*(s<1).mean())
    ).reset_index()
    paired=[]
    for seed, sub in df.groupby('seed'):
        d4=sub[(sub.defensor==4)&(sub.atacante>=7)].sort_values('atacante')
        d5=sub[(sub.defensor==5)&(sub.atacante>=7)].sort_values('atacante')
        assert d4.atacante.tolist() == d5.atacante.tolist() == list(range(7,21))
        paired.append({'seed':seed,'media_D4_pct':d4.evasao_pct.mean(),
                       'media_D5_pct':d5.evasao_pct.mean(),
                       'queda_D4_para_D5_pp':(d4.evasao_pct.to_numpy()-d5.evasao_pct.to_numpy()).mean()})
    paired=pd.DataFrame(paired)
    # Comparações descritivas utilizam seeds como unidades para CI, não pares dependentes.
    mean_k=by_k.groupby('defensor').agg(
        evasao_media_pct=('evasao_media_pct','mean'),n_pares_por_seed=('n','first'),
        pct_abaixo_1=('pct_abaixo_1','mean')
    ).reset_index()
    mean_k[['ci95_inf','ci95_sup']] = [seed_bootstrap_ci(by_k.loc[by_k.defensor==k,'evasao_media_pct'],rng) for k in mean_k.defensor]
    mean_gap=by_gap.groupby('distancia_grupo',observed=True).agg(
        evasao_media_pct=('evasao_media_pct','mean'), pct_abaixo_1=('pct_abaixo_1','mean'),
        n_pares_por_seed=('n','first')
    ).reset_index()
    mean_gap[['ci95_inf','ci95_sup']]=[seed_bootstrap_ci(by_gap.loc[by_gap.distancia_grupo==g,'evasao_media_pct'],rng) for g in mean_gap.distancia_grupo]
    overall_per_seed=by_seed['evasao_media_pct']
    paired_ci=seed_bootstrap_ci(paired['queda_D4_para_D5_pp'],rng)
    summary = {
        'design': 'E2-A: 15 seeds; future nonadjacent A_j x D_k with 1<=k<=19 and j>=k+2',
        'seeds':sorted(main.seed.unique().tolist()),
        'n_pares_todos_futuros':int((df.tipo.isin(['futuro_adjacente','futuro_nao_adjacente'])).sum()),
        'n_adjacentes':int((df.tipo=='futuro_adjacente').sum()),
        'n_principais':len(main),
        'evasao_futura_media_pct':float(overall_per_seed.mean()),
        'evasao_futura_media_bootstrap_ci95_pct':seed_bootstrap_ci(overall_per_seed,rng),
        'evasao_mediana_todos_pares_pct':float(main.evasao_pct.median()),
        'pct_pares_evasao_abaixo_1':float((main.evasao_pct<1).mean()*100),
        'pct_pares_evasao_maior_igual_5':float((main.evasao_pct>=5).mean()*100),
        'n_pares_atacantes_efetivos':len(eff),
        'pct_pares_atacantes_efetivos':len(eff)/len(main)*100,
        'evasao_futura_media_ataques_efetivos_pct':float(eff.evasao_pct.mean()),
        'evasao_contemporanea_media_ataques_efetivos_pct':float(eff.evasao_alvo_contemporaneo_pct.mean()),
        'pct_ataques_efetivos_com_evasao_futura_abaixo_1':float((eff.evasao_pct<1).mean()*100),
        'efeito_pareado_D4_vs_D5_mesmos_A7_A20_queda_media_pp':float(paired.queda_D4_para_D5_pp.mean()),
        'efeito_pareado_D4_vs_D5_ci95_pp':paired_ci,
        'seeds_melhora_D4_para_D5':int((paired.queda_D4_para_D5_pp>0).sum()),
        'limites':['As seeds também alteram splits do dataset','Attackers futuros pertencem à mesma trajetória',
                   'A matriz reutiliza validação/ruído do E0','D20 não possui Attacker posterior na matriz']
    }
    return main, eff, by_seed, by_k,mean_k,by_gap,mean_gap,per_seed_eff,paired, summary


def charts(df:pd.DataFrame, main: pd.DataFrame, by_k:pd.DataFrame,mean_k:pd.DataFrame,
           mean_gap:pd.DataFrame,per_seed_eff:pd.DataFrame,paired:pd.DataFrame,fig_dir:Path):
    fig_dir.mkdir(parents=True, exist_ok=True)
    # heatmap masked past + diagonal + adjacent
    piv = df.pivot_table(index='atacante',columns='defensor',values='evasao_pct').reindex(index=range(1,21),columns=range(1,20))
    h=piv.to_numpy();msk=np.array([[j<k+2 for k in range(1,20)] for j in range(1,21)])
    h=np.ma.array(h,mask=msk)
    fig,ax=plt.subplots(figsize=(11,7));im=ax.imshow(h,origin='lower',interpolation='nearest',aspect='auto',cmap='viridis',vmin=0,vmax=40)
    ax.set_xticks(range(19),labels=[f'D{i}' for i in range(1,20)],rotation=45)
    ax.set_yticks(range(20),labels=[f'A{i}' for i in range(1,21)])
    ax.set_xlabel('Checkpoint do Defender');ax.set_ylabel('Checkpoint do Attacker')
    ax.set_title('E2-A — Evasão média entre seeds, apenas Attackers futuros não adjacentes')
    fig.colorbar(im,ax=ax,label='Evasão média (%)');fig.tight_layout();fig.savefig(fig_dir/'01_heatmap_transferencia.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5));ax.plot(mean_k.defensor,mean_k.evasao_media_pct,marker='o',label='Média entre seeds')
    ax.fill_between(mean_k.defensor,mean_k.ci95_inf,mean_k.ci95_sup,alpha=.23,label='Bootstrap por seed — IC 95%')
    ax.set_xticks(range(1,20));ax.set_ylabel('Evasão futura média (%)');ax.set_xlabel('Defender Dk');ax.set_title('Transferência temporal ao longo do treinamento')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(fig_dir/'02_curva_por_defender.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));pos=np.arange(len(mean_gap));ax.bar(pos,mean_gap.evasao_media_pct)
    ax.errorbar(pos,mean_gap.evasao_media_pct,yerr=[mean_gap.evasao_media_pct-mean_gap.ci95_inf,mean_gap.ci95_sup-mean_gap.evasao_media_pct],capsize=4,fmt='none',color='black')
    ax.set_xticks(pos,labels=mean_gap.distancia_grupo.astype(str));ax.set_xlabel('Distância temporal j − k (rodadas)');ax.set_ylabel('Evasão futura média (%)')
    ax.set_title('Evasão por distância temporal — composição de checkpoints variável');ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(fig_dir/'03_distancia_temporal.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));means=main.groupby('seed').evasao_pct.mean().sort_index()
    ax.bar([str(s) for s in means.index],means.to_numpy());ax.set_xlabel('Seed');ax.set_ylabel('Evasão futura média (%)');ax.set_title('Heterogeneidade entre seeds (171 pares por seed)')
    ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(fig_dir/'04_distribuicao_seeds.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));p=paired.sort_values('seed');xs=np.arange(len(p));w=.42
    ax.bar(xs-w/2,p.media_D4_pct,w,label='D4');ax.bar(xs+w/2,p.media_D5_pct,w,label='D5');ax.set_xticks(xs,labels=p.seed.astype(str))
    ax.set_xlabel('Seed');ax.set_ylabel('Evasão média (%)');ax.set_title('Comparação pareada: D4 vs D5 contra os mesmos A7–A20')
    ax.legend();ax.grid(axis='y',alpha=.25);fig.tight_layout();fig.savefig(fig_dir/'05_d4_d5_pareado.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(per_seed_eff));width=.4;p=per_seed_eff.sort_values('seed')
    ax.bar(x-width/2,p.ev_future,width,label='Contra Defender futuro anterior Dk');ax.bar(x+width/2,p.ev_target,width,label='Contra alvo contemporâneo D(j−1)')
    ax.set_xticks(x,labels=p.seed.astype(str));ax.set_xlabel('Seed');ax.set_ylabel('Evasão média (%)');ax.set_title('Subconjunto condicionado a Attackers efetivos (≥5% no alvo)')
    ax.grid(axis='y',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(fig_dir/'06_ataques_efetivos.png',dpi=170);plt.close(fig)


def run(inputs: list[str],output:Path, enforce_seeds:bool=True):
    sources = sources_from_inputs(inputs)
    if not sources:
        raise ValueError('Nenhuma fonte de matrizes encontrada')
    seeds=[];info=[]
    for path in sources:
        seed,mat,history,metrics=read_source(path)
        if seed in {s for s,_ in seeds}:
            raise ValueError(f'Seed duplicada: {seed}')
        values=validate_matrix(seed,mat,history)
        if len(metrics)!=20 or not {'rodada','fpr','f1','recall'}.issubset(metrics.columns):
            raise ValueError(f'Seed {seed}: métricas convencionais incompletas')
        seeds.append((seed,values));info.append((seed,metrics))
    seeds.sort(key=lambda item:item[0]);actual={s for s,_ in seeds}
    if enforce_seeds and actual != EXPECTED_SEEDS:
        raise ValueError(f'Seeds inesperadas. Faltando={sorted(EXPECTED_SEEDS-actual)}, extras={sorted(actual-EXPECTED_SEEDS)}')
    df=make_pairs(seeds)
    assert len(df)==len(seeds)*420
    rng=np.random.default_rng(2026)
    main,eff,by_seed,by_k,mean_k,by_gap,mean_gap,per_seed_eff,paired,summary=summarise(df,rng)
    data=output/'data'; figs=output/'figures';data.mkdir(parents=True,exist_ok=True)
    tables={'todos_pares':df, 'pares_futuros_nao_adjacentes':main,
            'resumo_por_seed':by_seed,'por_seed_e_defender':by_k,
            'resumo_por_defender':mean_k,'por_seed_e_distancia':by_gap,
            'resumo_por_distancia':mean_gap,'atacantes_efetivos_por_seed':per_seed_eff,
            'comparacao_pareada_D4_D5':paired}
    conventional=pd.concat([v.assign(seed=k) for k,v in info],ignore_index=True)
    tables['metricas_convencionais_contexto']=conventional
    for name,table in tables.items():
        table.to_csv(data/(name+'.csv'),index=False,float_format='%.8f')
    (data/'resumo_estatistico.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    charts(df,main,by_k,mean_k,mean_gap,per_seed_eff,paired,figs)
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary


def main():
    parser=argparse.ArgumentParser(description='E2-A: análise de transferência temporal de ADArena')
    parser.add_argument('--input',nargs='+',required=True,help='ZIPs, runs ou diretório contendo ambos')
    parser.add_argument('--output',type=Path,default=Path('docs/scientific/e2/e2_assets'))
    parser.add_argument('--allow-subset',action='store_true',help='Para testes piloto apenas; padrão exige 15 seeds')
    args=parser.parse_args()
    run(args.input,args.output,not args.allow_subset)


if __name__=='__main__':
    main()
