"""E3: analise reproduzivel das 15x15 transferencias cross-seed.

Uso:
 uv run python docs/scientific/scripts/e3_analise.py \\
   --input models/e3/campanha/resultados.csv \\
   --output-dir docs/scientific/e3/e3_assets

Os intervalos exploratorios fazem bootstrap independente de sementes para os
indices de origem/destino, com exclusao das diagonais reais. Os 210 pares
cross-seed NAO sao tratados como 210 sementes independentes.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def pct(x): return 100.0 * x

def analisar(input_csv:Path, output_dir:Path, n_boot:int=12000, bootstrap_seed:int=20261117):
    df=pd.read_csv(input_csv)
    seeds=sorted(int(x) for x in df.seed_origem.unique())
    if len(seeds) != 15: raise ValueError('Esperadas 15 seeds')
    labels=['destino','controle_mesma_trajetoria','efetividade_na_origem']
    checkpoints=['A3','A20']
    if len(df)!=1350 or not set(labels)==set(df.avaliacao): raise ValueError('Contagem ou labels invalidos')
    key=['seed_origem','seed_destino','checkpoint_atacante','avaliacao']
    if df.duplicated(key).any(): raise ValueError('Duplicatas de chaves')
    if len(df[df.avaliacao=='destino'])!=450: raise ValueError('Matriz incompleta')
    if (df.n<1).any() or (df.n>5000).any(): raise ValueError('N invalido')
    if not np.allclose(df.evasao,df.evasoes/df.n,atol=1e-9): raise ValueError('Inconsistencia evasao vs contagem')
    if not np.allclose(df.evasao_limpa,1-df.n_detectadas_limpas/df.n,atol=1e-9): raise ValueError('Inconsistencia base limpa')
    if df.epsilon.nunique()!=1 or df.threshold.nunique()!=1:raise ValueError('Parametros divergentes')
    wide=df.pivot(index=['seed_origem','seed_destino','checkpoint_atacante'],columns='avaliacao',values=['evasao','evasao_limpa','sucesso_condicional','n','base_test_ids_sha256'])
    wide.columns=['_'.join(col) for col in wide.columns]
    wide=wide.reset_index()
    for c in wide.columns:
        if c.startswith(('evasao_','sucesso_condicional_')): wide[c]=pd.to_numeric(wide[c])
    if (wide.n_destino!=wide.n_controle_mesma_trajetoria).any() or (wide.n_destino!=wide.n_efetividade_na_origem).any():raise ValueError('Controles com N divergente')
    if (wide.base_test_ids_sha256_destino!=wide.base_test_ids_sha256_controle_mesma_trajetoria).any() or (wide.base_test_ids_sha256_destino!=wide.base_test_ids_sha256_efetividade_na_origem).any():raise ValueError('Controles sem base compartilhada')
    wide['cross']=wide.seed_origem!=wide.seed_destino
    wide['transferencia_pp']=pct(wide.evasao_destino)
    wide['controle_mesma_seed_pp']=pct(wide.evasao_controle_mesma_trajetoria)
    wide['referencia_origem_pp']=pct(wide.evasao_efetividade_na_origem)
    wide['limpa_destino_pp']=pct(wide.evasao_limpa_destino)
    wide['dif_dest_vs_own_d20_pp']=wide.transferencia_pp-wide.controle_mesma_seed_pp
    figdir=output_dir/'figures'; datadir=output_dir/'data'
    figdir.mkdir(parents=True,exist_ok=True); datadir.mkdir(parents=True,exist_ok=True)
    wide.to_csv(datadir/'e3_pares_consolidados.csv',index=False)
    dest=wide[wide.cross]
    summary=[]
    for ch in checkpoints:
        sub=dest[dest.checkpoint_atacante==ch]
        eff=sub[sub.referencia_origem_pp>=5]
        diag=wide[(~wide.cross)&(wide.checkpoint_atacante==ch)]
        summary.append({'checkpoint':ch,'n_cross':len(sub),'media_cross_pp':sub.transferencia_pp.mean(),
            'mediana_cross_pp':sub.transferencia_pp.median(),'max_cross_pp':sub.transferencia_pp.max(),
            'n_cross_ge1':int((sub.transferencia_pp>=1).sum()),'n_cross_ge5':int((sub.transferencia_pp>=5).sum()),
            'n_efetivos_ref_ge5':len(eff),'origens_efetivas':int(eff.seed_origem.nunique()),
            'media_cross_efetivos_pp':eff.transferencia_pp.mean(),
            'media_referencia_origem_efetivos_pp':eff.referencia_origem_pp.mean(),
            'media_cross_own_d20_diff_pp':sub.dif_dest_vs_own_d20_pp.mean(),
            'media_diag_pp':diag.transferencia_pp.mean()})
    pd.DataFrame(summary).to_csv(datadir/'e3_resumo_por_checkpoint.csv',index=False)
    bysrc=dest.groupby(['checkpoint_atacante','seed_origem']).agg(transferencia_media_pp=('transferencia_pp','mean'),
        referencia_origem_media_pp=('referencia_origem_pp','mean')).reset_index()
    bydst=dest.groupby(['checkpoint_atacante','seed_destino']).agg(transferencia_media_pp=('transferencia_pp','mean'),
        referencia_origem_media_pp=('referencia_origem_pp','mean')).reset_index()
    bysrc.to_csv(datadir/'e3_resumo_por_origem.csv',index=False)
    bydst.to_csv(datadir/'e3_resumo_por_destino.csv',index=False)
    rng=np.random.default_rng(bootstrap_seed); N=len(seeds)
    mats={ch:wide[wide.checkpoint_atacante==ch].pivot(index='seed_origem',columns='seed_destino',values='transferencia_pp').loc[seeds,seeds].to_numpy(float) for ch in checkpoints}
    own=wide[wide.checkpoint_atacante=='A3'].pivot(index='seed_origem',columns='seed_destino',values='controle_mesma_seed_pp').loc[seeds,seeds].to_numpy(float)
    samples=np.empty((n_boot,4))
    ss=np.array(seeds)
    for it in range(n_boot):
        a=rng.integers(0,N,N);b=rng.integers(0,N,N)
        off=ss[a,None]!=ss[None,b]
        A=mats['A3'][np.ix_(a,b)][off]
        B=mats['A20'][np.ix_(a,b)][off]
        C=own[np.ix_(a,b)][off]
        samples[it]=[A.mean(),B.mean(),(A-B).mean(),(A-C).mean()]
    ci={name:np.percentile(samples[:,i],[2.5,97.5]).tolist() for i,name in enumerate(['A3_cross_pp','A20_cross_pp','A3_minus_A20_pp','A3_cross_minus_own_D20_pp'])}
    paired=wide[wide.cross].pivot(index=['seed_origem','seed_destino'],columns='checkpoint_atacante',values='transferencia_pp')
    result={'seeds':seeds,'n_aval':len(df),'n_pares':len(wide),'n_cross_por_checkpoint':210,
        'n_efetivos_ref_ge5':{r['checkpoint']:r['n_efetivos_ref_ge5'] for r in summary},
        'A3_supera_A20_pares':int((paired.A3>paired.A20).sum()),
        'A3_empata_A20_pares':int((paired.A3==paired.A20).sum()),
        'A3_inferior_A20_pares':int((paired.A3<paired.A20).sum()),
        'bootstrap_two_way_cluster_95_ci_pp':ci,'bootstrap_n':n_boot,'bootstrap_seed':bootstrap_seed,
        'summary':summary}
    (datadir/'e3_estatisticas.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    # Fig 1 & 2 heatmaps; omit stars/ranks. Shared color limits for comparison.
    for ch,num in [('A3','01'),('A20','02')]:
        fig,ax=plt.subplots(figsize=(9,7))
        mat=mats[ch]; img=ax.imshow(mat,vmin=0,vmax=13,interpolation='nearest',aspect='auto')
        ax.set_xticks(np.arange(N),[str(k) for k in seeds]);ax.set_yticks(np.arange(N),[str(k) for k in seeds]);
        ax.set_xlabel('Defender D20 — seed de destino');ax.set_ylabel(f'Attacker {ch} — seed de origem')
        ax.set_title(f'E3 — evasão (%) de {ch} contra D20 (inclui diagonal)')
        fig.colorbar(img,ax=ax,label='Taxa de evasão (%)')
        fig.tight_layout();fig.savefig(figdir/f'{num}_heatmap_{ch.lower()}.png',dpi=170);plt.close(fig)
    # Fig 3 A3 and A20 average per Defender
    fig,ax=plt.subplots(figsize=(10,4.8))
    xx=np.arange(N);w=.38
    for ch,xshift in [('A3',-.19),('A20',.19)]:
        vals=bydst[bydst.checkpoint_atacante==ch].sort_values('seed_destino').transferencia_media_pp.values
        ax.bar(xx+xshift,vals,width=w,label=ch)
    ax.set_xticks(xx,[str(x) for x in seeds]);ax.set_xlabel('D20 — seed de destino');ax.set_ylabel('Evasão média cross-seed (%)');ax.legend();ax.set_title('Vulnerabilidade média por Defender (14 origens externas)')
    fig.tight_layout();fig.savefig(figdir/'03_destinos_comparados.png',dpi=170);plt.close(fig)
    # Fig 4 distributions (off only) provide log-like but simple range and inset? use boxplot
    fig,ax=plt.subplots(figsize=(7,4.6))
    v=[dest[dest.checkpoint_atacante==ch].transferencia_pp.to_numpy(float) for ch in checkpoints]
    ax.boxplot(v,tick_labels=checkpoints,showmeans=True,showfliers=True)
    ax.set_ylabel('Evasão cross-seed (%)');ax.set_xlabel('Checkpoint do Attacker');ax.set_title('Distribuição dos 210 pares cross-seed por checkpoint')
    fig.tight_layout();fig.savefig(figdir/'04_distribuicao_evasao.png',dpi=170);plt.close(fig)
    # Fig 5 own ref vs target, points with effective red? use scatter default and x
    fig,ax=plt.subplots(figsize=(7,5))
    for ch in checkpoints:
        sub=dest[dest.checkpoint_atacante==ch];ax.scatter(sub.referencia_origem_pp,sub.transferencia_pp,s=15,alpha=.6,label=ch)
    ax.axvline(5,linestyle='--',linewidth=1,label='Efetividade ≥ 5%')
    ax.set_xlabel('Evasão contra Defender de referência (D2 ou D19) (%)')
    ax.set_ylabel('Evasão contra D20 externo (%)');ax.set_title('Transferência vs efetividade na origem');ax.legend()
    fig.tight_layout();fig.savefig(figdir/'05_efetividade_e_transferencia.png',dpi=170);plt.close(fig)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('models/e3/campanha/resultados.csv'))
    p.add_argument('--output-dir',type=Path,default=Path('docs/scientific/e3/e3_assets'))
    p.add_argument('--bootstrap',type=int,default=12000)
    a=p.parse_args();analisar(a.input,a.output_dir,a.bootstrap)
if __name__=='__main__':main()
