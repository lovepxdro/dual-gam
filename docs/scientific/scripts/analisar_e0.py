#!/usr/bin/env python3
"""E0 / ADArena: consolida artefatos .zip em CSVs, figuras e relatório Markdown.

Exemplo:
    python scripts/analisar_e0.py --source '/caminho/run_*seed*.zip' --out .

Dependências: numpy, pandas, matplotlib.
Não abre ou executa checkpoints; somente lê JSON e CSV das execuções.
"""
from __future__ import annotations
import argparse, copy, glob, json, math, pathlib, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXPECTED_SEEDS = [*range(1,11),20,40,42,80,100]

def read_json(z, suffix):
    names=[k for k in z.namelist() if k.endswith('/'+suffix)]
    if len(names)!=1: raise ValueError(f'{suffix} não encontrado univocamente no ZIP')
    return json.loads(z.read(names[0]))

def percent(x): return 100*float(x)
def fmt(x, dec=2): return f'{x:.{dec}f}'.replace('.',',')
def stat(x):
    x=np.asarray(x,dtype=float)
    return {'mean':x.mean(),'sd':x.std(ddof=1), 'median':np.median(x),'min':x.min(),'max':x.max()}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source', required=True, help='glob para ZIPs de runs, ex.: /mnt/data/run_*seed*.zip')
    ap.add_argument('--out', default='.', help='diretório de saída, relativo a raiz do repositório')
    args=ap.parse_args()
    out=pathlib.Path(args.out); docroot=out/'docs/scientific'; assets=docroot/'e0_assets'
    assets.mkdir(parents=True,exist_ok=True)
    rows=[]; rounds=[]; historical=[]; matrix=[]; allconfigs=[]
    files=sorted(glob.glob(args.source))
    for path in files:
        with zipfile.ZipFile(path) as z:
            bad=z.testzip()
            if bad: raise ValueError(f'Arquivo ZIP inválido {path}: {bad}')
            s=read_json(z,'summary.json'); h=read_json(z,'historico_treino.json'); m=read_json(z,'matriz_checkpoints.json');c=read_json(z,'config_execucao.json')
            seed=int(s['seed'])
            if seed!=int(c['seed']): raise ValueError(f'Seed divergente em {path}')
            if s['n_rodadas']!=20: raise ValueError(f'Quantidade de rodadas inesperada: seed {seed}')
            for key in ['taxa_evasao_pre_adaptacao','taxa_evasao_pos_adaptacao','fpr_defensor_validacao','fnr_defensor_validacao','f1_defensor_validacao']:
                if len(h[key])!=20: raise ValueError(f'Histórico inválido seed {seed} {key}')
            cm=np.asarray(s['teste_final']['confusion_matrix'],dtype=float)
            tn,fp,fn,tp=cm[0,0],cm[0,1],cm[1,0],cm[1,1]
            # Matthews correlation coefficient (TN FP FN TP; rótulo positivo = ataque)
            denominator=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
            mcc=(tp*tn-fp*fn)/denominator if denominator else np.nan
            summary={'seed':seed, 'run_id':s['run_id'],
                     **{f'{k}_teste':float(v) for k,v in s['teste_final'].items() if k!='confusion_matrix'},
                     'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'mcc_teste':mcc,
                     'evasao_pre_r1':float(h['taxa_evasao_pre_adaptacao'][0]),
                     'evasao_pre_r20':float(h['taxa_evasao_pre_adaptacao'][-1]),
                     'evasao_pos_r20':float(h['taxa_evasao_pos_adaptacao'][-1]),
                     'evasao_historica_media_d20':float(s['robustez_acumulada']['evasao_media_d_final']),
                     'evasao_historica_max_d20':float(s['robustez_acumulada']['pior_evasao_d_final']),
                     'pior_atacante_d20':int(s['robustez_acumulada']['pior_atacante_d_final']),
                     'fpr_val_r18':float(h['fpr_defensor_validacao'][17]),
                     'fpr_val_r19':float(h['fpr_defensor_validacao'][18]),
                     'fpr_val_r20':float(h['fpr_defensor_validacao'][19])}
            pre=np.asarray(h['taxa_evasao_pre_adaptacao']);post=np.asarray(h['taxa_evasao_pos_adaptacao'])
            summary['evasao_pre_media_20r']=float(pre.mean());summary['evasao_pos_media_20r']=float(post.mean());summary['ganho_medio_pp_20r']=percent((pre-post).mean())
            rows.append(summary)
            for j in range(20):
                rounds.append({'seed':seed,'rodada':j+1,'evasao_pre':float(pre[j]),'evasao_pos':float(post[j]),
                               'ganho_pp':percent(pre[j]-post[j]),
                               'fpr_validacao':float(h['fpr_defensor_validacao'][j]),
                               'fnr_validacao':float(h['fnr_defensor_validacao'][j]),
                               'recall_validacao':float(h['recall_defensor_validacao'][j]),
                               'f1_validacao':float(h['f1_defensor_validacao'][j])})
            mdata=np.asarray(m['matriz_evasao'],dtype=float)
            if mdata.shape!=(20,21):raise ValueError(f'Matriz inesperada {seed} {mdata.shape}')
            for i in range(20):
                historical.append({'seed':seed,'atacante':i+1,'evasao_contra_d20':mdata[i,20]})
                for j in range(21):
                    matrix.append({'seed':seed,'atacante':i+1,'defensor':j,'evasao':mdata[i,j]})
            # Compara parâmetros efetivos; ignorar identidade e horário (não ignorar outras diferenças)
            norm=copy.deepcopy(c)
            for key in ('run_id','created_at','seed'):norm.pop(key,None)
            allconfigs.append((seed,norm))
    seeds=sorted(x['seed'] for x in rows)
    if seeds!=EXPECTED_SEEDS:raise ValueError(f'Seeds observadas {seeds}; esperadas {EXPECTED_SEEDS}')
    base=allconfigs[0][1]
    for seed,c in allconfigs[1:]:
        if c!=base: raise ValueError(f'Configurações divergentes para seed {seed}')

    runs=pd.DataFrame(rows).sort_values('seed').reset_index(drop=True)
    rr=pd.DataFrame(rounds).sort_values(['seed','rodada']).reset_index(drop=True)
    hist=pd.DataFrame(historical).sort_values(['seed','atacante']).reset_index(drop=True)
    mat=pd.DataFrame(matrix).sort_values(['seed','atacante','defensor']).reset_index(drop=True)
    runs.to_csv(assets/'resumo_por_seed.csv',index=False,float_format='%.9g')
    rr.to_csv(assets/'metricas_por_rodada.csv',index=False,float_format='%.9g')
    hist.to_csv(assets/'evasao_historica_d20.csv',index=False,float_format='%.9g')
    mat.to_csv(assets/'matriz_checkpoints_longa.csv',index=False,float_format='%.9g')

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.figsize':(9.4,4.6)})
    axdata=rr.pivot(index='rodada',columns='seed',values='evasao_pre')*100
    med=axdata.median(axis=1); avg=axdata.mean(axis=1);lo=axdata.quantile(.25,axis=1);hi=axdata.quantile(.75,axis=1)
    fig,ax=plt.subplots()
    for seed in seeds:ax.plot(axdata.index,axdata[seed],alpha=.19,linewidth=.9)
    ax.plot(axdata.index,avg,linewidth=2.6,label='Média (15 seeds)')
    ax.plot(axdata.index,med,linewidth=2.1,linestyle='--',label='Mediana')
    ax.fill_between(axdata.index,lo,hi,alpha=.21,label='Intervalo interquartil (P25–P75)')
    ax.set(xlabel='Rodada adversarial',ylabel='Evasão pré-adaptação (%)',title='E0 — Evolução da evasão antes da adaptação')
    ax.set_xticks([1,2,3,4,5,8,10,12,15,18,20]);ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(fontsize=9,loc='upper right')
    fig.tight_layout();fig.savefig(assets/'01_evasao_pre_evolucao.png',dpi=180);plt.close(fig)

    pre=rr.groupby('rodada')['evasao_pre'].mean()*100
    post=rr.groupby('rodada')['evasao_pos'].mean()*100
    fig,ax=plt.subplots()
    ax.plot(pre.index,pre.values,linewidth=2.2,marker='.',label='Pré-adaptação (média)')
    ax.plot(post.index,post.values,linewidth=2.2,marker='.',label='Pós-adaptação (média)')
    ax.set(xlabel='Rodada adversarial',ylabel='Evasão média (%)',title='E0 — Adaptação por rodada (15 seeds)')
    ax.set_xticks(range(1,21));ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(assets/'02_pre_pos_evolucao.png',dpi=180);plt.close(fig)

    fprmatrix=rr.pivot(index='rodada',columns='seed',values='fpr_validacao')*100
    fig,ax=plt.subplots(figsize=(9.8,4.9))
    for seed in seeds:
        if seed in (6,80,100):continue
        ax.plot(fprmatrix.index,fprmatrix[seed],linewidth=.9,alpha=.27)
    for seed in (6,80,100):
        ax.plot(fprmatrix.index,fprmatrix[seed],linewidth=2,marker='.',markersize=3,label=f'Seed {seed}')
    ax.set(xlabel='Rodada adversarial',ylabel='FPR na validação (%)',title='E0 — Quedas tardias de FPR em três execuções')
    ax.set_xticks(range(1,21));ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(assets/'03_fpr_validacao_rodadas.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.5,4.4))
    ax.bar([str(s) for s in seeds],runs['fpr_teste']*100)
    ax.axhline(runs['fpr_teste'].mean()*100,linestyle='--',linewidth=1.5,label=f'Média: {runs["fpr_teste"].mean()*100:.2f}%')
    ax.set(xlabel='Seed',ylabel='FPR no teste reservado (%)',title='E0 — FPR final por execução')
    ax.grid(axis='y',alpha=.18);ax.legend();fig.tight_layout();fig.savefig(assets/'04_fpr_teste_seeds.png',dpi=180);plt.close(fig)

    a3=mat[mat.atacante==3].pivot(index='defensor',columns='seed',values='evasao')*100
    a3_view=a3.loc[3:20]
    fig,ax=plt.subplots(figsize=(9.5,4.7))
    for seed in seeds:ax.plot(a3_view.index,a3_view[seed],alpha=.2,linewidth=.9)
    ax.plot(a3_view.index,a3_view.mean(axis=1),linewidth=2.7,marker='.',label='Média de A3 nas 15 seeds')
    ax.plot(a3_view.index,a3_view.median(axis=1),linewidth=2,linestyle='--',label='Mediana')
    ax.set(xlabel='Checkpoint do Defender (D3–D20)',ylabel='Evasão de A3 (%)',title='E0 — Robustez histórica contra A3 após a rodada 3')
    ax.set_xticks([3,5,7,9,11,13,15,17,19,20]);ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(assets/'05_a3_ao_longo_defensores.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(9.5,4.5))
    for attacker in (2,3,4,10):
        sub=mat[(mat.atacante==attacker)&(mat.defensor>=4)].groupby('defensor')['evasao'].mean()*100
        ax.plot(sub.index,sub,linewidth=2,label=f'A{attacker}')
    ax.set(xlabel='Checkpoint do Defender',ylabel='Evasão média (%)',title='E0 — Ataques históricos selecionados (média entre seeds)')
    ax.set_xticks(range(4,21,2));ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(assets/'06_attackers_historicos.png',dpi=180);plt.close(fig)

    # Relatório
    p=lambda k:stat(runs[k]*100)
    f1=p('f1_teste'); fpr=p('fpr_teste'); recall=p('recall_teste'); mcc=stat(runs['mcc_teste']);auc=stat(runs['roc_auc_teste']);
    hist_mean=p('evasao_historica_media_d20'); hist_max=p('evasao_historica_max_d20')
    combined=rr.copy();combined['pre_pp']=combined.evasao_pre*100;combined['post_pp']=combined.evasao_pos*100
    n0=len(combined); nbelow=(combined.post_pp<1).sum()
    high=combined[combined.pre_pp>=5]
    highbelow=(high.post_pp<1).sum()
    peaks=combined[(combined.rodada>=10)&(combined.pre_pp>=10)]
    a3d3=a3.loc[3]; a3d20=a3.loc[20]; drift=a3d20-a3d3
    freq=runs.pior_atacante_d20.value_counts().to_dict()
    perrun=[]
    for _,r in runs.iterrows():
        seed=int(r.seed)
        perrun.append(f'| {seed} | `{r.run_id}` | {fmt(r.f1_teste*100)}% | {fmt(r.fpr_teste*100)}% | {fmt(r.recall_teste*100)}% | {fmt(r.mcc_teste,3)} | {fmt(r.evasao_historica_media_d20*100)}% | A{int(r.pior_atacante_d20)} ({fmt(r.evasao_historica_max_d20*100)}%) |')
    byround=[]
    for t,g in rr.groupby('rodada'):
        e=stat(g.evasao_pre*100); po=stat(g.evasao_pos*100); gain=stat(g.ganho_pp)
        byround.append(f'| {t} | {fmt(e["mean"])} ± {fmt(e["sd"])} | {fmt(e["median"])} | {fmt(po["mean"])} | {fmt(gain["mean"])} |')
    grp=[]
    for seed in (6,80,100):
        sr=runs[runs.seed==seed].iloc[0]
        grp.append(f'| {seed} | {fmt(sr.fpr_val_r18*100)}% | {fmt(sr.fpr_val_r19*100)}% | {fmt(sr.fpr_val_r20*100)}% | {fmt(sr.fpr_teste*100)}% |')
    a3tab=[]
    for seed in seeds:
        a3tab.append(f'| {seed} | {fmt(a3.loc[3,seed])}% | {fmt(a3.loc[20,seed])}% | {fmt(a3.loc[20,seed]-a3.loc[3,seed]):s} pp |')
    pre_first=stat(rr[rr.rodada==1].evasao_pre*100)
    pre_four=stat(rr[rr.rodada==4].evasao_pre*100)
    pre_final=stat(rr[rr.rodada==20].evasao_pre*100)
    post_total=stat(rr.evasao_pos*100)
    gain_total=stat(rr.ganho_pp)
    r18fpr=stat(rr[rr.rodada==18].fpr_validacao*100)
    a3ever=stat(a3d20)
    doc=f'''# E0 — Baseline multi-seed

## Estado

**Concluído (análise descritiva).** Coleta: 20 de outubro de 2026; 15 execuções independentes (seeds 1–10, 20, 40, 42, 80 e 100); 300 rodadas adversariais. O documento substitui a etapa inicial de planejamento (seeds 42, 43 e 44) pelos experimentos efetivamente executados.

## 1. Pergunta científica

> O comportamento atual da ADArena é estável quando o mesmo experimento é repetido com diferentes seeds?

## 2. Hipótese

> Mantendo dataset, arquitetura e hiperparâmetros constantes, seeds diferentes devem produzir variação quantitativa nas métricas, mas preservar o comportamento qualitativo de adaptação do Defender.

**Critério interpretativo**: observar capacidade inicial de evasão, redução da evasão após adaptação, variabilidade entre seeds e desempenho no teste reservado. Não é exigida convergência monotônica. O E0 não pretende prever o resultado a partir do valor numérico da seed.

## 3. Objetivo

Estabelecer a baseline multi-seed da linha ADArena v2.7; caracterizar a evolução temporal da evasão e os resultados finais, quantificar a variabilidade e identificar limitações que possam orientar E1–E3 sem modificar os modelos neste experimento.

## 4. Metodologia e variáveis

- **Variável alterada:** seed (1–10, 20, 40, 42, 80, 100). As partições treino/validação/teste e a inicialização dependem da seed; portanto, o E0 mede a variabilidade do **pipeline completo**, e não somente da inicialização.
- **Controles verificados nos `config_execucao.json`:** demais campos idênticos entre as 15 execuções após excluir identificador de run, data de criação e seed. Todos no modo `TRAIN`, com o mesmo dataset, componentes e hiperparâmetros.
- **Dataset:** CIC-IDS2017, `data/DDoS-Friday-no-metadata.parquet`, 221.264 amostras, 77 features; benignas 93.250 e DDoS 128.014. Split por execução: treino 154.884, validação 22.127 e teste 44.253, sem interseção reportada.
- **Modelos e protocolo:** `adarena.perturbation_attacker`, `adarena.binary_mlp_defender`, `adarena.adversarial_training`; threshold 0,5; epsilon 0,3; pré-treino de 5 epochs; 20 rodadas de 3 epochs; 5.000 amostras adversariais de validação por rodada.
- **Coleta:** 15 ZIPs contendo `summary.json`, `historico_treino.json`, `matriz_checkpoints.json`, configuração, gráficos, métricas e checkpoints. O pipeline foi executado localmente pelo pesquisador; esta análise reproduz cálculos a partir dos artefatos exportados, sem retreinar modelos.
- **Estatística:** médias, desvios padrões amostrais (n=15, `ddof=1`), medianas, mínimos e máximos. Para curvas por rodada, são calculados valores agregados entre 15 runs na mesma rodada. **Não tratar as 300 rodadas como amostras independentes** para inferência estatística.
- **Ganho de adaptação:** `evasao_pre - evasao_pos`, em pontos percentuais; taxas de evasão são frações antes de converter para %. **MCC** calculado a partir da matriz de confusão final (`TN`, `FP`, `FN`, `TP`) — não foi uma métrica exportada diretamente no `summary.json`.

## 5. Execuções e desempenho final

| Seed | Run ID | F1 teste | FPR teste | Recall teste | MCC teste | Evasão média histórica × D20 | Pior ataque histórico × D20 |
|---:|---|---:|---:|---:|---:|---:|---|
{chr(10).join(perrun)}

### 5.1 Síntese quantitativa

| Métrica | Média ± DP | Mediana | Mínimo–máximo |
|---|---:|---:|---:|
| F1 no teste | {fmt(f1['mean'])}% ± {fmt(f1['sd'])} pp | {fmt(f1['median'])}% | {fmt(f1['min'])}%–{fmt(f1['max'])}% |
| Recall no teste | {fmt(recall['mean'])}% ± {fmt(recall['sd'])} pp | {fmt(recall['median'])}% | {fmt(recall['min'])}%–{fmt(recall['max'])}% |
| FPR no teste | {fmt(fpr['mean'])}% ± {fmt(fpr['sd'])} pp | {fmt(fpr['median'])}% | {fmt(fpr['min'])}%–{fmt(fpr['max'])}% |
| MCC no teste | {fmt(mcc['mean'],3)} ± {fmt(mcc['sd'],3)} | {fmt(mcc['median'],3)} | {fmt(mcc['min'],3)}–{fmt(mcc['max'],3)} |
| ROC-AUC no teste | {fmt(auc['mean'],5)} ± {fmt(auc['sd'],5)} | {fmt(auc['median'],5)} | {fmt(auc['min'],5)}–{fmt(auc['max'],5)} |
| Evasão média histórica contra D20 | {fmt(hist_mean['mean'])}% ± {fmt(hist_mean['sd'])} pp | {fmt(hist_mean['median'])}% | {fmt(hist_mean['min'])}%–{fmt(hist_mean['max'])}% |
| Maior evasão histórica contra D20 | {fmt(hist_max['mean'])}% ± {fmt(hist_max['sd'])} pp | {fmt(hist_max['median'])}% | {fmt(hist_max['min'])}%–{fmt(hist_max['max'])}% |

**Leitura:** desempenho convencional geralmente elevado, mas FPR e evasão histórica apresentam dispersão mais relevante que recall. Valores de FPR representam **teste reservado**, enquanto as séries temporais de FPR (a seguir) representam **validação**; essas medidas não devem ser misturadas numa mesma curva.

![FPR no teste por seed](e0_assets/04_fpr_teste_seeds.png)

## 6. Evolução da evasão ao longo das rodadas

![Evasão pré-adaptação ao longo das rodadas](e0_assets/01_evasao_pre_evolucao.png)

![Média pré e pós-adaptação por rodada](e0_assets/02_pre_pos_evolucao.png)

| Rodada | Pré: média ± DP (%) | Pré: mediana (%) | Pós: média (%) | Ganho: média (pp) |
|---:|---:|---:|---:|---:|
{chr(10).join(byround)}

- **Primeira rodada:** evasão pré-adaptação média {fmt(pre_first['mean'])}% (DP {fmt(pre_first['sd'])} pp). O atacante encontrou evasão em todas as 15 seeds e a adaptação reduziu a evasão significativamente na rodada inicial.
- **Quarta rodada:** evasão pré-adaptação média {fmt(pre_four['mean'])}%; **vigésima:** {fmt(pre_final['mean'])}% (mediana {fmt(pre_final['median'])}%). A média final é influenciada por picos nas seeds 42 e 100. Isso mostra declínio agregado entre início e fim, **não** uma queda contínua.
- **Oscilações persistentes:** foram observadas {len(peaks)} rodadas com evasão pré-adaptação ≥10% entre as rodadas 10–20, distribuídas por {peaks.seed.nunique()} seeds. Há recuperação temporária da capacidade de evasão pelo Attacker mesmo após períodos com valores próximos de zero.
- **Resposta imediata do Defender:** evasão pós-adaptação média de {fmt(post_total['mean'],3)}% nas 300 comparações; {nbelow}/{n0} rodadas ({fmt(100*nbelow/n0)}%) terminaram com evasão pós-adaptação abaixo de 1%. O único valor ≥1% ocorreu na seed 8, rodada 1 (1,26%).
- **Ataques com evasão pré ≥5%:** {highbelow}/{len(high)} ficaram abaixo de 1% após adaptação, reforçando que o padrão de adaptação aparece principalmente quando existe uma evasão material a reduzir.
- **Ganho médio por rodada e seed, agregado descritivamente:** {fmt(gain_total['mean'])} pp. Em rodadas com evasão pré já muito próxima de zero, o ganho se aproxima de zero ou pode ser levemente negativo; isso não invalida o efeito observado nos picos.

**Interpretação temporal:** a curva sugere redução acentuada nas rodadas iniciais, seguida de oscilações/episódios de evasão, sem lei de crescimento determinística ou convergência monotônica. É possível descrever o desempenho típico ao longo dessas 20 rodadas por média, mediana e dispersão empíricas; não é adequado extrapolar mecanicamente para mais rodadas ou prever a partir do inteiro da seed.

## 7. Diferenças no FPR: efeito persistente ou tardio?

Na avaliação final do teste reservado, as seeds 6, 80 e 100 apresentaram FPR de {fmt(float(runs.loc[runs.seed==6,'fpr_teste'].iloc[0])*100)}%, {fmt(float(runs.loc[runs.seed==80,'fpr_teste'].iloc[0])*100)}% e {fmt(float(runs.loc[runs.seed==100,'fpr_teste'].iloc[0])*100)}%, respectivamente. As outras 12 ficaram entre {fmt(runs[~runs.seed.isin([6,80,100])].fpr_teste.min()*100)}% e {fmt(runs[~runs.seed.isin([6,80,100])].fpr_teste.max()*100)}%.

Uma inspeção das séries de validação muda a interpretação: na rodada 18, o FPR médio entre as 15 execuções era {fmt(r18fpr['mean'])}% ± {fmt(r18fpr['sd'])} pp. As três seeds destacadas **ainda tinham valores próximos aos demais** nessa rodada.

| Seed | FPR validação r18 | FPR validação r19 | FPR validação r20 | FPR teste final |
|---:|---:|---:|---:|---:|
{chr(10).join(grp)}

![FPR na validação por rodada](e0_assets/03_fpr_validacao_rodadas.png)

**Resultado observado:** a melhora aparece no último checkpoint (seeds 6 e 80) e nas duas últimas rodadas (seed 100), sem perda expressiva de recall nessas transições. **Interpretação:** esses resultados não demonstram a existência de um regime de baixo FPR persistentemente estável; são pontos finais favoráveis de trajetórias que anteriormente apresentavam FPR semelhante ao das outras seeds. Não é possível inferir só pelas curvas se a causa é composição das partições, dinâmica de atualização, oscilações do classificador ou outro fator. Não selecionar seeds a posteriori como se fossem arquiteturas superiores.

## 8. Recorrência do A3 e retenção de robustez histórica

A avaliação cruzada compara `A1..A20` contra `D0..D20` **dentro de cada execução**. O atacante A3 foi o pior caso contra D20 em **{freq.get(3,0)}/15 seeds**; A2 foi o pior nas outras {freq.get(2,0)}.

![Evasão de A3 contra os checkpoints de Defender](e0_assets/05_a3_ao_longo_defensores.png)

![Evasão histórica de ataques selecionados](e0_assets/06_attackers_historicos.png)

Para o mesmo checkpoint de atacante A3, comparando as avaliações de cada matriz:

- Evasão de A3 contra D3: média **{fmt(a3d3.mean())}%** entre seeds;
- Evasão de A3 contra D20: média **{fmt(a3d20.mean())}%**, mediana {fmt(a3d20.median())}%, máximo {fmt(a3d20.max())}% (seed 1);
- A diferença A3×D20 − A3×D3 foi positiva em **{(drift>1e-6).sum()}/15 seeds**, média **{fmt(drift.mean())} pp**. Na seed 10, houve pequena redução ({fmt(drift.loc[10])} pp).

| Seed | Evasão A3×D3 | Evasão A3×D20 | Diferença |
|---:|---:|---:|---:|
{chr(10).join(a3tab)}

**Resultado observado:** a capacidade de detectar o A3 não se manteve igualmente forte ao longo da sequência de checkpoints, apesar de o Defender apresentar evasão pós-adaptação muito baixa em rodadas individuais. **Hipótese interpretativa:** possível perda de robustez contra um ataque antigo durante o treinamento continuado, compatível com esquecimento parcial. **Não comprovado neste E0:** o mecanismo de esquecimento, causalidade específica ou desempenho perante atacantes de outras seeds. Isso justifica uma investigação direcionada em E1/E2, não uma modificação imediata do modelo.

## 9. Confronto com a hipótese

| Critério | Evidência dos artefatos | Julgamento |
|---|---|---|
| Attacker encontra evasão | Evasão pré inicial em todas as 15 seeds; média {fmt(pre_first['mean'])}% | Sustentado |
| Defender adapta nas mesmas rodadas | {nbelow}/{n0} evasões pós abaixo de 1%; {highbelow}/{len(high)} quando pré ≥5% | Sustentado para adaptação imediata |
| Intensidade varia quantitativamente | FPR de {fmt(fpr['min'])}%–{fmt(fpr['max'])}%; evasão histórica máxima de {fmt(hist_max['min'])}%–{fmt(hist_max['max'])}% | Sustentado; variação relevante |
| Convergência monotônica | Picos de evasão durante todo o ciclo | Não observada nem exigida |
| Robustez a ataques históricos preservada | A3×D20 > A3×D3 em {(drift>1e-6).sum()}/15 | Não sustentado como estabilidade temporal forte |

**Síntese:** a hipótese é **parcialmente sustentada**. O comportamento qualitativo central — surgimento de evasão e adaptação imediata — foi reproduzido em todas as seeds. Entretanto, a consistência da resposta imediata não equivale a estabilidade temporal plena: o FPR final depende fortemente do checkpoint e algumas resistências históricas enfraquecem com o treinamento.

## 10. Limitações e validade

1. **Split variável:** seed altera partições e inicialização; impossível atribuir toda a dispersão à inicialização isoladamente. Diferentes composições de teste restringem comparações causais.
2. **Seleção intencional de seeds:** não são uma amostra formalmente aleatória do espaço de seeds; estatísticas são descritivas para as 15 execuções, não garantias universais.
3. **Dependência temporal:** 20 rodadas de uma mesma seed compartilham estado e treino. Não usar 300 como n independente.
4. **Avaliação histórica intra-seed:** D20 contra Attackers históricos da própria trajetória, sem testes cross-seed ou ataques desconhecidos; generalização não foi examinada.
5. **Uma execução por seed:** não avalia determinismo ou reprodutibilidade bit a bit para seeds repetidas.
6. **Métricas e domínio:** apenas um dataset e família DDoS; alta performance em dataset de features não comprova eficácia sobre pacotes transmitidos ou plausibilidade física das perturbações.
7. **Análise pós-hoc:** picos A3 e baixo FPR de 6/80/100 foram identificados nos dados. Não interpretar suas comparações como testes confirmatórios sem pré-registro ou validação adicional.
8. **Horizonte fixo:** 20 rodadas; tendência empírica não estabelece taxa assintótica de crescimento nem garante projeção para períodos mais longos.

## 11. Conclusão e decisão

**Conclusão do E0:** a configuração atual da ADArena oferece uma baseline **adequada para estudar adaptação imediata em condições controladas**, com resposta forte do Defender aos ataques gerados durante as mesmas rodadas e com desempenho convencional alto. Porém, sua estabilidade deve ser descrita como **parcial**, não total: o FPR exibe alterações tardias e a robustez frente ao checkpoint histórico A3 pode diminuir em defensores posteriores.

**Decisão:** concluir E0 sem novo treinamento complementar obrigatório e usar a distribuição empírica (média, DP, mediana, mínimo/máximo por seed e curvas por rodada) como referência nas comparações futuras. Ao comparar melhorias, manter as mesmas sementes, documentar o split, reportar métricas convencionais **e** robustez histórica, e evitar selecionar apenas o checkpoint com melhor FPR sem critério prévio de seleção.

## 12. Próximos passos derivados

- **E1 (adaptação):** investigar retenção da resposta a Attackers históricos, especialmente A3 ao longo de D3–D20; comparar adaptação imediata com resistência acumulada.
- **E2 (generalização):** avaliar Attackers ou checkpoints separados da adaptação quando formalmente não observados pelo Defender.
- **E3 (cross-seed):** estudar transferência entre treinamentos independentes com seeds distintas.
- **Complemento opcional, não pré-requisito do E0:** fixar as partições e variar só a aleatoriedade dos modelos para estudar separadamente a variação do FPR; repetir alguma seed para testar reprodutibilidade; avaliar seleção de checkpoint por validação com regra definida antes de consultar teste.

## 13. Artefatos reproduzíveis

Na pasta [`e0_assets`](e0_assets/):

- `resumo_por_seed.csv` — uma linha por seed (15);
- `metricas_por_rodada.csv` — 20 rodadas × 15 seeds (300);
- `evasao_historica_d20.csv` — 20 ataques × 15 seeds (300);
- `matriz_checkpoints_longa.csv` — 20 ataques × 21 Defenders × 15 seeds (6.300);
- `01_*.png` a `06_*.png` — visualizações;
- `scripts/analisar_e0.py` (na raiz do pacote) — reconstrução dos dados e relatório a partir dos ZIPs, sem carregar os checkpoints `.pth`.

**Para reproduzir a análise com os ZIPs em diretório próprio:**

```bash
python scripts/analisar_e0.py --source 'models/e0/exports/run_*seed*.zip' --out .
```

Os arquivos `.zip` de execução originais não estão incluídos neste pacote de relatório devido ao tamanho; devem ser preservados separadamente como dados brutos do experimento.
'''
    (docroot/'e0.md').write_text(doc,encoding='utf-8')
    print('Seeds:',seeds)
    print('Média f1',f1,'fpr',fpr,'mcc',mcc)
    print('300 avaliações; pos<1%',nbelow,'pre>5%',len(high),'pre>5% and pos<1%',highbelow)
    print('A3 D3',a3d3.mean(),'D20',a3d20.mean(),'delta mean',drift.mean(),'positive',(drift>0).sum())
    print('Arquivo:',docroot/'e0.md')

if __name__=='__main__':main()
