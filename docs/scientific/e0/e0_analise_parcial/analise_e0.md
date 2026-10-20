# E0 — Consolidação descritiva das 15 seeds

Análise dos artefatos enviados pelo usuário. Não representa validação independente do código, nem resultados em outro dataset.

## Protocolo

- Seeds: 1–10, 20, 40, 42, 80 e 100 (15 execuções).
- Dataset: `DDoS-Friday-no-metadata.parquet` (221.264 linhas, 77 features).
- Configurações conferidas: diferem apenas em seed, `run_id` e horário de criação.
- Dispositivo CPU; 5 epochs de pré-treino do Defender; 20 rodadas; 3 epochs por rodada; 5.000 amostras adversariais de validação por rodada; epsilon 0,3 e limiar 0,5.
- Um split pseudoaleatório é realizado por seed; portanto, resultados entre seeds variam em inicialização **e** composição das partições.
- Avaliações pré e pós-adaptação usam o mesmo conjunto adversarial dentro de uma mesma rodada.

## Resultados agregados (teste reservado e histórico)

| Métrica | Média ± desvio padrão entre seeds | Mediana | Mínimo–máximo |
|---|---:|---:|---:|
| f1 | 98,87% ± 0,40 pp | 98,69% | 98,53%–99,68% |
| recall | 99,95% ± 0,02 pp | 99,95% | 99,92%–99,97% |
| fpr | 3,08% ± 1,13 pp | 3,57% | 0,79%–4,04% |
| evasao_media_d_final | 0,29% ± 0,24 pp | 0,24% | 0,05%–0,81% |
| pior_evasao_d_final | 3,55% ± 3,29 pp | 2,78% | 0,10%–11,98% |

## Evolução do treinamento

- Evasão pré-adaptação na 1ª rodada: média 73,92% (desvio padrão 2,49 pp).
- Evasão pré-adaptação na 20ª rodada: média 2,32%, mediana 0,04%. A média é elevada principalmente pelas seeds 42 e 100.
- Evasão pós-adaptação nas 300 rodadas: média 0,05%; 299/300 observações abaixo de 1%.
- Ganho médio pré−pós-adaptação: 13,48 pontos percentuais, agregado sobre todas as rodadas e seeds,
- Em 13/15 execuções, A3 foi o pior atacante histórico contra D20; A2 nas outras duas.
- A taxa pré-adaptação oscila ao longo das rodadas. Não há evidência de queda monotônica nem de convergência irreversível do atacante.

## Resultados por seed (teste reservado)

| Seed | F1 | FPR | Evasão histórica média contra D20 | Pior evasão histórica contra D20 |
|---:|---:|---:|---:|---:|
| 1 | 98,69% | 3,59% | 0,81% | 11,98% |
| 2 | 98,90% | 2,99% | 0,24% | 2,78% |
| 3 | 98,71% | 3,55% | 0,70% | 8,46% |
| 4 | 98,72% | 3,50% | 0,08% | 0,66% |
| 5 | 98,66% | 3,68% | 0,07% | 1,08% |
| 6 | 99,67% | 0,80% | 0,23% | 3,30% |
| 7 | 98,62% | 3,77% | 0,10% | 1,44% |
| 8 | 98,53% | 4,04% | 0,58% | 6,28% |
| 9 | 98,68% | 3,61% | 0,05% | 0,62% |
| 10 | 98,69% | 3,57% | 0,07% | 0,10% |
| 20 | 98,68% | 3,57% | 0,10% | 1,34% |
| 40 | 98,64% | 3,68% | 0,31% | 2,26% |
| 42 | 98,61% | 3,79% | 0,24% | 3,92% |
| 80 | 99,68% | 0,79% | 0,34% | 4,48% |
| 100 | 99,53% | 1,24% | 0,38% | 4,60% |

## Observações

- O FPR final formou dois grupos observados: seeds 6, 80 e 100 (0,79%–1,24%) e as outras 12 (2,99%–4,04%). Isso sugere regimes de desempenho a investigar, não uma causa demonstrada.
- A seed 100 teve evasão pré-adaptação de 15,66% na rodada 20, reduzida para 0,02% após adaptação; F1 final 99,53%, FPR 1,24%.
- A média histórica de evasão contra D20 é baixa, mas A3 persiste como principal ponto de atenção; isso não demonstra generalização para atacantes desconhecidos.
- Resultados exibidos como 0,0% no terminal podem ser pequenos valores positivos nos JSONs.

## Limitações metodológicas

1. As seeds foram selecionadas de forma intencional, não por amostragem aleatória formal. Evitar inferências causais sobre o valor numérico da seed.
2. O split varia com a seed; não separa variação do treinamento de variação nas partições de dados.
3. Comparações com D20 usam apenas atacantes históricos da **mesma** execução (não avaliam transferência entre seeds ou ataques inéditos).
4. Há uma execução por seed: não comprova reprodutibilidade bit a bit de uma mesma seed.
5. Não se está medindo complexidade assintótica nem custo de escalabilidade.

## Arquivos de apoio

- `resumo_por_seed.csv`: métricas finais de cada execução.
- `metricas_por_rodada.csv`: séries completas (300 linhas).
- `evasao_historica_d20.csv`: matriz (15 seeds × 20 ataques).
- `01_evasao_por_rodada.png`, `02_fpr_por_seed.png`, `03_ataques_historicos.png`: figuras.

**Status:** análise descritiva consolidada; interpretação final e próximo desenho experimental devem ser discutidos antes de fechar `docs/scientific/e0.md`.
