# E5-B — Campanha ampliada de Attacker A0 × A1 × A2

**Estado:** implementação e testes sintéticos concluídos. A campanha real ainda **não** foi executada neste pacote.

## Pergunta de pesquisa

O ganho observado para o Attacker condicional A1(x,z), após a correção da perda por logits e a ablação de capacidade, mantém-se nas **15 seeds** do E0? Esse ganho também aumenta a transferência adversarial para outros Defenders?

## Escopo e hipóteses

- **A0**: gerador histórico G(z), 64.013 parâmetros.
- **A1**: gerador condicional G(x,z), 104.538 parâmetros.
- **A2**: gerador z-only com ramo residual de capacidade próxima à A1, 104.482 parâmetros.
- Todos partem do **A20** histórico correspondente à seed e treinam contra o **D20 congelado** da mesma seed. As três arquiteturas mantêm BCEWithLogitsLoss, 3 épocas, batch de 512 e ε=0,3.
- H1: A1 supera A2 em ASR condicional no Defender-alvo, em média nas seeds.
- H2: A1 supera A0 e A2 em transferência adversarial, em média nas seeds.
- H3: a melhora de ASR não exige necessariamente maior diversidade média das perturbações.

## Controle e tamanho

15 seeds × 3 réplicas × 3 arquiteturas = **135 treinamentos**. Se os arquivos originais de E5-A2/E5-A3 estiverem completos, **18 checkpoints** (seeds 20,42 × 3 réplicas × 3 braços) são copiados após verificação da procedência. Restam **117 treinamentos novos**. Se a reutilização falhar, não ignorar a falha; corrigir fontes ou executar os 135 do zero em pasta nova.

**Observação:** reaproveitamos apenas **pesos finais de treinamento**, jamais ASRs anteriores. O novo protocolo reavalia todos os Attackers em amostras de teste comparáveis.

## Particionamento e avaliação

1. **Treinamento:** split E0 individual da seed; DDoS do treino, scaler persistido, base A20 e D20 inalterados. Validação somente para curvas; não seleciona checkpoint. Ruído e ordem de batches idênticos para os três braços numa réplica.
2. **Evasão no alvo:** até 5.000 fluxos DDoS do **teste da própria seed**, selecionados deterministicamente. As três arquiteturas usam os mesmos índices e ruído de teste para cada seed/réplica. ASR = sucessos dentre os fluxos que o D20 detecta corretamente antes da perturbação.
3. **Transferência:** para cada par de seeds `(origem,destino)`, avaliar até 1.024 DDoS pertencentes à **interseção entre os dois testes E0** (test/test). Isso evita avaliar registros presentes no treino de qualquer um dos dois Defenders. Amostragem idêntica nas duas direções do mesmo par, mesmo ruído por réplica e arquitetura. Perturbação é criada no espaço normalizado da origem e convertida para o scaler do destino, sem recalibrar o Defender.
4. **Completude:** 15 seeds, 45 pares seed/réplica, 135 avaliações no alvo e **1.890 avaliações de transferência dirigidas** (`15 × 14 × 3 × 3`). São **105 pares não dirigidos**, mas **210 direções** distintas.

## Inferência e limitações

- A comparação principal é **pareada por seed/réplica**. Para resumos inferenciais, primeiro calcular a média das três réplicas de cada seed; relatar distribuição de **15 valores por seed**. Não tratar 1.890 avaliações de transferência como 1.890 observações independentes.
- A média de transferência é uma média por origem, sobre seus 14 Defenders de destino; as interseções test/test podem variar em composição entre pares.
- Intervalo bootstrap entre seeds pode descrever a variabilidade da campanha, **não** a generalização para datasets externos.
- **Não** comparar diretamente o ASR absoluto deste E5-B com E5-A2/A3: amostragem de teste e protocolo ampliado não são os mesmos.
- Perturbações em 77 features padronizadas podem não ser fisicamente realizáveis como fluxos válidos. E5-B mede desempenho do ataque **no espaço de features**, não eficácia em redes reais.
- Similaridade de parâmetros entre A1/A2 não equivale a arquiteturas idênticas; ganhos apoiam a hipótese de condicionamento, mas não provam causalidade isolada.

## Evidência gerada

- `config_e5b.json`: parâmetros da campanha e hash do preflight E5-P expandido.
- `seed_*/rep_*/A*/treinamento.json` e `atacante_final.pth`: resultados e checkpoints; cada execução é retomável.
- `seed_*/rep_*/A*/avaliacao_alvo.json`: ASR, magnitudes e diversidade no D20 da origem.
- `seed_*/rep_*/A*/transferencia_para_*.json`: métricas em cada D20 de destino.
- `metricas_alvo.csv`, `metricas_transferencia.csv`, `resumo_e5b.json`.
- `b_analise/comparacao_pareada_seed_replica.csv`, `resumo_por_seed.csv`, `analise_e5b.json`.

Nenhum desses resultados é antecipado aqui. O relatório científico final deverá ser atualizado **após** a execução e a análise.

- O orçamento ε=0,3 é definido no **espaço normalizado da origem**; ao aplicar a perturbação em outro scaler, o deslocamento nas coordenadas normalizadas de destino pode não respeitar esse mesmo limite. Esse aspecto deve constar na discussão de validade e comparação com E3.
