# E5-A2 — Análise comparativa com E5-A

## Escopo e controles

- E5-A: BCE aplicada às probabilidades após `Sigmoid`.
- E5-A2: BCEWithLogitsLoss aplicada à saída anterior à `Sigmoid`.
- 12 treinamentos em cada versão: seeds E0 20 e 42, três réplicas (1042, 1043, 1044), braços A0 e A1.
- Checkpoints A20 como inicialização; D20 congelado; três épocas; batch 512; epsilon 0,3; validação 4096 e teste 5000.
- Configurações comuns idênticas; mesmo hash das amostras de teste e mesmo ruído de avaliação para pares correspondentes. As 4998 amostras corretamente detectadas por D20 no conjunto de teste são o denominador de ASR em todos os resultados.

## Resultados (médias de seis execuções por arquitetura)

| Indicador | A0: E5-A | A0: E5-A2 | A1: E5-A | A1: E5-A2 |
|---|---:|---:|---:|---:|
| ASR no Defender-alvo | 0.627% | 3.795% | 0.650% | 12.395% |
| Transferência (20↔42) | 0% | 0.774% | 0% | 1.997% |
| Diversidade L2 entre perturbações | 7.421 | 4.870 | 6.239 | 4.354 |
| Duração treino | 7.94s | 10.30s | 11.60s | 14.61s |

A1 superou A0 em todas as seis combinações de seed e réplica no E5-A2, com vantagem média de 8.600 pontos percentuais. No E5-A, a vantagem média fora apenas 0.023 pp.

## Diagnóstico de saturação

Nos diagnósticos pré-treinamento, em um mesmo batch, a proporção média de probabilidades iguais a 1 em precisão de máquina foi de 65.10%. A norma L2 média do gradiente passou de 7.479 para 66.601 (razão média por execução de 9.74×). Isso apoia a hipótese de que a formulação anterior prejudicava a otimização, mas não demonstra que todos os gradientes anteriores fossem exatamente nulos.

## Interpretação e limites

A mudança de loss elevou a evasão de A0 e A1, além de permitir transferência não nula. Sob loss estável, A1 apresentou ganho em todas as comparações pareadas. Ainda assim, essa superioridade não pode ser atribuída exclusivamente ao condicionamento por x: A1 tem mais parâmetros (104538 vs. 64013), e são apenas duas seeds base. A diversidade L2 média global é menor após a correção; diversidade entre pares eficazes deve ser interpretada com cuidado, pois o subconjunto de pares eficazes mudou de tamanho entre execuções.

As perturbações são feitas em espaço normalizado e não foi demonstrada validade física de fluxos de rede reconstruídos.

## Próxima decisão científica recomendada

Adicionar uma **ablação de capacidade** (mesmo porte de parâmetros, sem dependência efetiva de x) antes de atribuir as diferenças ao condicionamento e antes de ampliar para 15 seeds. Outra possibilidade é ampliar primeiro como resultado **de comparação de arquiteturas**, declarando explicitamente esse fator de confusão.
