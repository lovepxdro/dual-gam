# E5 — Registro de avanço: preflight E5-P e execução E5-A

**Atualizado:** 19/11/2026.

## E5-P: concluído

O arquivo de preflight fornecido pelo pesquisador (`models/e5/preflight/e5_preflight.json`) reporta `status=aprovado` para seeds 20 e 42. Nas duas: 77 features, noise_dim 32, epsilon 0.3, D20 congelado, gradientes no A1, e erro máximo inicial A0 versus A1 igual a zero. O preflight verifica o isolamento dos splits a partir dos registros persistidos do E0, sem reconstruí-los.

**Não houve treinamento nem avaliação adversarial no E5-P.**

## E5-A: implementação pronta; execução pendente

Execução: `docs/scientific/scripts/e5_train.py`. Cada uma das seeds 20 e 42 terá três repetições (1042, 1043, 1044) pareadas A0/A1, totalizando 12 treinamentos. Ambas as arquiteturas começam com o backbone A20 da seed respectiva. D20 é congelado. Adam reinicializado. Três epochs em todos os DDoS de treino, batch 512, epsilon 0.3, lr_atacante e betas da execução E0.

O script reconstrói as duas divisões estratificadas, deduplicação e scaler E0, com validação contra a auditoria e estatísticas do scaler; falha se houver divergências. Para diagnóstico, as curvas por epoch usam somente validação. O teste é executado uma vez, após o último epoch pré-definido. Origem e destino usam a mesma interseção do conjunto de teste das duas seeds e os mesmos vetores de ruído em braços pareados. Transferência recompõe as features brutas antes de aplicar o scaler do destino.

Critérios para revisar o piloto: execução completa, nenhum erro de integridade, ASR condicional no alvo e no destino, diversidade de perturbações e diversidade entre pares eficazes, magnitude, parâmetros, tempo, variabilidade entre repetições. Ganho de A1 não pode ser atribuído exclusivamente ao condicionamento devido à maior capacidade.

**Resultados científicos do E5-A ainda não existem neste documento.** Não confundir os dados de testes unitários/smoke com resultados experimentais.
