# E5-A3 — Ablação de capacidade: análise preliminar

## Escopo e integridade

Foram concluídos 6 treinamentos do Attacker A2, correspondentes às seeds 20 e 42 e réplicas 1042–1044. O comparador persistido assinalou `comparabilidade: aprovada` entre A0, A1 e A2; os 6 registros do A2 têm mesma assinatura do conjunto comum de teste (5.000 ataques DDoS), mesmo orçamento ε = 0,3, 3 épocas, perda por logits, saídas iniciais idênticas ao A0 (`max_abs_diff = 0`) e invariância a `x` (`max_abs_diff = 0`).

**Atenção:** a igualdade de SHA do teste só foi verificada aqui diretamente entre os 6 resultados A2; a comparabilidade com A0/A1 se apoia nos arquivos de comparação fornecidos, não em nova reexecução de inferência.

## Resultados principais — médias das seis comparações

| Métrica | A0 original | A1 condicional | A2 controle de capacidade |
|---|---:|---:|---:|
| Parâmetros | 64.013 | 104.538 | 104.482 |
| ASR condicional no alvo | 3.795% | 12.395% | 4.022% |
| ASR condicional de transferência | 0.774% | 1.997% | 1.100% |
| Diversidade L2 | 4.870 | 4.354 | 4.062 |
| Tempo médio de treinamento | 10.30 s | 14.61 s | 14.45 s |

A vantagem média de A1 sobre A2 foi **8.373 pontos percentuais**, observada nas **6/6 comparações pareadas** (amplitude 5.122 a 13.025 pp). A1 alcançou aproximadamente 3.08 vezes o ASR médio de A2.

A2 teve vantagem média sobre A0 de apenas **0.227 pp**, vencendo **3/6** pares. Não houve ganho uniforme associado exclusivamente ao aumento do número de parâmetros.

## Resultado por seed

| Seed | A0 alvo | A1 alvo | A2 alvo | A1 − A2 | A2 − A0 |
|---|---:|---:|---:|---:|---:|
| 20 | 2.434% | 10.311% | 4.155% | +6.156 pp | +1.721 pp |
| 42 | 5.155% | 14.479% | 3.888% | +10.591 pp | -1.267 pp |

## Interpretação

- Nas condições do piloto, o condicionamento de A1 está associado a maior eficácia do que o controle A2 de capacidade quase equivalente; a maior capacidade isolada não reproduziu o ganho.
- O ASR de transferência melhorou em A1, mas os valores absolutos permaneceram baixos.
- A diversidade L2 de A1 excedeu a de A2, mas permaneceu abaixo da de A0; maior eficácia não implica maior diversidade.
- Não se pode atribuir causalidade exclusiva ao condicionamento: A1 e A2 têm arquitetura interna diferente, apesar de quase igual contagem de parâmetros. Existem apenas dois Defenders independentes; réplicas do mesmo Defender não são 6 observações independentes para inferência populacional.

## Próxima etapa sugerida: E5-B

Ampliar para as 15 seeds originais, com as três arquiteturas e três réplicas, usando perda baseada em logits, D20 congelado e controles pareados (135 treinamentos no desenho integral). É possível reutilizar os 18 treinamentos A0/A1/A2 do piloto apenas se o protocolo final preservar exatamente as mesmas definições de conjunto de teste, ruído e normalização; do contrário, tratá-los como piloto independente.

**Atenção metodológica:** não exigir uma interseção comum entre testes das 15 seeds. Comparações A0/A1/A2 devem usar o mesmo teste de origem *dentro de cada seed*; transferências para outros Defenders devem usar amostras compatíveis com cada par de splits e scalers, reportando os tamanhos efetivos dos conjuntos.
