"""Consolida E2-B com bootstrap por seed original e figuras reproduzíveis.

Exemplo:
  python analisar_e2b.py --campaign campanha.zip --pilot seed42_1042.zip \
    --pilot seed42_1043.zip --pilot seed42_1044.zip \
    --output docs/scientific/e2/e2_assets/data/e2b

Caminhos de entrada aceitam ZIP ou diretório com resultados.json; os pesos .pth
não são lidos. Requer pandas, numpy, matplotlib. Não altera checkpoints.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def load_records(path: Path):
    if path.is_file() and path.suffix.lower() == '.zip':
        with ZipFile(path) as z:
            bad = z.testzip()
            if bad:
                raise ValueError(f'ZIP corrompido: {path} / {bad}')
            names = sorted(n for n in z.namelist() if n.endswith('resultados.json'))
            if not names:
                raise ValueError(f'Sem resultados.json: {path}')
            return [(f'{path.name}!{n}', json.loads(z.read(n))) for n in names]
    if path.is_dir():
        names = sorted(path.rglob('resultados.json'))
        if not names:
            raise ValueError(f'Sem resultados.json: {path}')
        return [(str(n), json.loads(n.read_text(encoding='utf-8'))) for n in names]
    raise FileNotFoundError(path)


def convert(name: str, d: dict) -> dict:
    m = {(x['atacante'], x['defender']): x for x in d['metrics']}
    ev = lambda a, b: float(m[a,b]['evasao'])*100.0
    b = d['budgets']; integ = d['integrity']
    if d['initialization'] != 'a20' or b['epochs'] != 3 or b['n_ddos_treino'] != 89609 or b['n_val_ddos'] != 5000:
        raise ValueError(f'Parâmetros divergentes em {name}')
    r = dict(seed=int(d['seed_e0']), attacker_seed=int(d['attacker_seed']), source=name,
             historical_d19=ev('A20_historico','D19'), historical_d20=ev('A20_historico','D20'),
             baseline_clean_d19=ev('limpo','D19'),baseline_clean_d20=ev('limpo','D20'),
             trained_d19_on_d19=ev('novo_alvo_D19','D19'),
             transferred_d19_to_d20=ev('novo_alvo_D19','D20'),
             trained_d20_on_d20=ev('novo_alvo_D20','D20'),
             trained_d20_to_d19=ev('novo_alvo_D20','D19'),
             hash19=integ['d19_sha256'],hash20=integ['d20_sha256'])
    for target in ['D19','D20']:
        for ep in [1,2,3]:
            vals=[x for x in d['curves'][target] if x['defender']==target and x['epoch']==ep]
            if len(vals) != 1:
                raise ValueError(f'Curva incompleta em {name}, {target}, epoch {ep}')
            r[f'epoch{ep}_{target}']=float(vals[0]['evasao'])*100
        measured=r['trained_d19_on_d19'] if target=='D19' else r['trained_d20_on_d20']
        if not np.isclose(r[f'epoch3_{target}'],measured,atol=1e-5):
            raise ValueError(f'Epoch final difere do resumo em {name}: {target}')
    for t in ['D19','D20']:
        if not np.isclose(integ[f'a20_{t.lower()}_evasao']*100,r[f'historical_{t.lower()}'],atol=1e-5):
            raise ValueError(f'Baseline histórico inconsistente em {name}')
    return r


def run(args):
    rows=[]
    for p in [Path(args.campaign)]+[Path(x) for x in args.pilot]:
        for name,d in load_records(p): rows.append(convert(name,d))
    df=pd.DataFrame(rows).sort_values(['seed','attacker_seed']).reset_index(drop=True)
    if df.duplicated(['seed','attacker_seed']).any():
        raise ValueError('Combinação de seed/attacker_seed duplicada')
    expected={1,2,3,4,5,6,7,8,9,10,20,40,42,80,100}
    if len(df)!=45 or set(df.seed)!=expected or not (df.groupby('seed').size()==3).all():
        raise ValueError('Coleta incompleta: esperado 15 seeds × 3 replicações = 45')
    for c in ['hash19','hash20','historical_d19','historical_d20']:
        if df.groupby('seed')[c].nunique().max() != 1:
            raise ValueError('Os checkpoints e controles devem coincidir dentro de cada seed: '+c)
    numeric=['historical_d19','historical_d20','trained_d19_on_d19','transferred_d19_to_d20',
             'trained_d20_on_d20','trained_d20_to_d19','baseline_clean_d20']
    g=df.groupby('seed')[numeric].mean().reset_index()
    boot=g.copy()
    boot['adv_minus_transfer']=g.trained_d20_on_d20-g.transferred_d19_to_d20
    boot['target_minus_transfer']=g.trained_d19_on_d19-g.transferred_d19_to_d20
    rng=np.random.default_rng(2026)
    pick=rng.integers(0,15,size=(20000,15)); table=boot.drop(columns=['seed']); draws=table.to_numpy()[pick].mean(axis=1); bounds=np.percentile(draws,[2.5,97.5],axis=0)
    intervals={key:dict(media_pct=float(table[key].mean()),IC95_bootstrap_seeds=[float(bounds[0,i]),float(bounds[1,i])]) for i,key in enumerate(table.columns)}
    t=df.trained_d19_on_d19>=5
    summary=dict(n_seeds=15,n_replicacoes=45,reps_per_seed=3,
                 n_d19_effective_5pct=int(t.sum()),
                 n_seeds_d19_effective=int(df.groupby('seed').trained_d19_on_d19.max().ge(5).sum()),
                 n_direct_d20_effective_5pct=int((df.trained_d20_on_d20>=5).sum()),
                 n_seeds_direct_d20_effective=int(df.groupby('seed').trained_d20_on_d20.max().ge(5).sum()),
                 n_transfer_ge_1pct=int((df.transferred_d19_to_d20>=1).sum()),
                 max_transfer_pct=float(df.transferred_d19_to_d20.max()),
                 max_direct_pct=float(df.trained_d20_on_d20.max()),
                 median_direct_pct=float(df.trained_d20_on_d20.median()),
                 effective_d19_mean_on_target_pct=float(df.loc[t,'trained_d19_on_d19'].mean()),
                 effective_d19_mean_transfer_pct=float(df.loc[t,'transferred_d19_to_d20'].mean()),bootstrap=intervals)
    dest=Path(args.output);dest.mkdir(parents=True,exist_ok=True)
    df.to_csv(dest/'replicacoes.csv',index=False,float_format='%.8f');g.to_csv(dest/'resumo_por_seed.csv',index=False,float_format='%.8f')
    (dest/'resumo_estatistico.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figdir=Path(args.figures);figdir.mkdir(parents=True,exist_ok=True)
        xs=np.arange(15);w=.28
        fig,ax=plt.subplots(figsize=(11,5.5));ax.bar(xs-w,g.trained_d19_on_d19,w,label='Novo A→D19');ax.bar(xs,g.transferred_d19_to_d20,w,label='A(D19)→D20');ax.bar(xs+w,g.trained_d20_on_d20,w,label='Novo A(D20)→D20');ax.set_xticks(xs,[str(x) for x in g.seed]);ax.set_xlabel('Seed E0');ax.set_ylabel('Evasão média (%)');ax.set_title('E2-B | Evasão por seed e condição');ax.legend(fontsize=8);ax.grid(axis='y',alpha=.18);fig.tight_layout();fig.savefig(figdir/'07_e2b_por_seed.png',dpi=155);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5));ax.boxplot([df.trained_d19_on_d19,df.transferred_d19_to_d20,df.trained_d20_on_d20],tick_labels=['A(D19)→D19','A(D19)→D20','A(D20)→D20'],showmeans=True);ax.set_ylabel('Evasão (%)');ax.set_title('E2-B | Distribuição das 45 replicações');ax.grid(axis='y',alpha=.18);fig.tight_layout();fig.savefig(figdir/'08_e2b_distribuicao.png',dpi=155);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,5));epochs=np.array([1,2,3]);ax.plot(epochs,[df[f'epoch{x}_D19'].mean() for x in epochs],'o-',label='A(D19)→D19');ax.plot(epochs,[df[f'epoch{x}_D20'].mean() for x in epochs],'s-',label='A(D20)→D20');ax.set_xticks(epochs);ax.set_xlabel('Epoch');ax.set_ylabel('Evasão média (%)');ax.set_title('E2-B | Evolução por epoch');ax.legend();ax.grid(alpha=.18);fig.tight_layout();fig.savefig(figdir/'09_e2b_evolucao.png',dpi=155);plt.close(fig)
    print('Concluído:',len(df),'replicações;',df.seed.nunique(),'seeds')
    print('Transferência média (%):',round(df.transferred_d19_to_d20.mean(),3))
    print('Direcionado a D20 (%):',round(df.trained_d20_on_d20.mean(),3))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--campaign',required=True,help='ZIP ou diretório com as 42 replicações')
    ap.add_argument('--pilot',action='append',default=[],help='ZIP/diretório de cada piloto seed 42; repetir 3 vezes')
    ap.add_argument('--output',required=True,help='Diretório de saída de CSVs e estatísticas')
    ap.add_argument('--figures',help='Diretório de saída dos três gráficos')
    run(ap.parse_args())
