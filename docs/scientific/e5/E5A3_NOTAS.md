# E5-A3 — Ablação de capacidade do Attacker

**Situação:** implementação pronta; campanha real pendente. Esta etapa não substitui nem modifica E5-A2.

## Pergunta

A vantagem de A1(x,z) sobre A0(z) no E5-A2 se deve ao acesso à amostra x ou apenas a uma arquitetura com maior número de parâmetros?

## Braços

| Braço | Entrada do gerador | Estrutura | Parâmetros (77 features, z=32) | Origem dos resultados |
|---|---|---|---:|---|
| A0 | z | Gerador original | 64.013 | E5-A2 (reutilizado) |
| A1 | x e z | A0 + ramo residual 128→128 | 104.538 | E5-A2 (reutilizado) |
| A2 | z | A0 + ramo residual 153→153 | 104.482 | E5-A3 (novo treinamento) |

A diferença entre A1 e A2 é de **56 parâmetros (0,054%)**. A2 recebe x apenas como argumento da API; os valores de x **não entram no cálculo da perturbação**. Todos os parâmetros de A2 participam de caminhos computacionais efetivos, diferentemente de uma alternativa com x artificialmente zerado.

## Controles e metodologia

- Mesmo checkpoint A20 de origem para A0, A1 e A2; mesma saída antes do treino.
- D20 congelado, scaler do E0, epsilon=0,3, limiar=0,5.
- Loss `BCEWithLogitsLoss` idêntica ao E5-A2.
- Seeds 20 e 42 × réplicas 1042, 1043, 1044: **6 novos treinamentos A2**.
- Três épocas e batch 512, treino DDoS completo; validação com 4.096 DDoS e teste comum com 5.000 DDoS.
- Mesmos índices de treino/validação/teste, mesma ordem de batches e ruído entre as réplicas; Adam novo com LR e betas do E0.
- Inferência em D20 de origem e no D20 da outra seed; ASR condicional à detecção correta no tráfego limpo, diversidade L2, tempo.
- Auditoria automática recusa incompatibilidade de preflight, orçamento, amostras de teste, semente do ruído ou configuração do otimizador.
- Checkpoints E5-A2 originais **não são alterados**.

## Análises previstas

1. **A1 − A2** (pontos percentuais de ASR): indício de benefício associado ao condicionamento por x, com capacidade próxima.
2. **A2 − A0**: ganho associado à expansão do gerador sem x.
3. **A1 − A0**: referência já medida no E5-A2.
4. Transferência, diversidade e consistência por seed/réplica.

Comparar **seis pares com apenas duas seeds independentes**. Não realizar testes de significância tratando os seis pares como seis experimentos independentes. O ajuste quase exato do número de parâmetros não iguala completamente a capacidade funcional: a largura interna de A2 (153) difere de A1 (128). A avaliação continua no espaço de features normalizadas e não demonstra plausibilidade do tráfego gerado.

## Artefatos esperados

- `models/e5/a3_capacidade/config_e5a3.json`
- `models/e5/a3_capacidade/seed_<seed>/rep_<rep>/A2/{resultado.json,atacante_final.pth}`
- `models/e5/a3_capacidade/resultados_e5a3.csv`
- `models/e5/a3_capacidade/resumo_e5a3.json`
- `models/e5/a3_comparacao/{comparacao_e5a3_pareada.csv,resumo_comparacao_e5a3.json}`

Os scripts foram testados com dados sintéticos; a campanha real ainda precisa ser executada no ambiente do projeto.
