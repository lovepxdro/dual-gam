# Resultados históricos da ADArena v1.7

> **Documento histórico.** Este arquivo preserva os resultados, interpretações e limitações observados durante a linha **v1.7** da ADArena. Ele não descreve o estado atual da arquitetura e **não deve ser usado como referência para métricas, fluxos ou capacidades da versão atual**. Desde essa execução, a arquitetura foi refatorada e vários componentes e protocolos foram modificados. Para o estado atual do projeto, consulte `architecture.md`, `flow.md` e `usage.md`.

## Objetivo deste documento

A v1.7 marcou a consolidação da primeira linha arquitetural da ADArena como ambiente experimental para pesquisa em defesa adaptativa baseada em aprendizado adversarial.

Este documento registra o que aquela versão permitiu observar sobre:

- vulnerabilidade adversarial do Defensor;
- adaptação após exposição às variantes geradas pelo Atacante;
- retenção de robustez ao longo dos checkpoints;
- diferença entre evasão matemática e tradução para parâmetros de rede;
- limitações da transferência entre feature-space e tráfego produzido;
- lacunas arquiteturais que motivaram as versões posteriores.

Os números abaixo pertencem à execução de referência utilizada na época e devem ser interpretados apenas dentro daquele contexto experimental.

---

## Contexto experimental da v1.7

A execução utilizava:

- seed fixa;
- separação entre treino, validação e teste;
- avaliação antes e depois da adaptação do Defensor;
- checkpoints históricos por rodada;
- matriz cruzada Atacante × Defensor;
- persistência de métricas e artefatos;
- Translator para converter vetores adversariais em parâmetros de rede;
- Sender para materialização ou simulação desses parâmetros.

Naquele momento, o experimento era organizado em quatro perguntas principais:

1. Um detector com desempenho convencional elevado continua vulnerável a variantes adversariais?
2. A exposição a essas variantes reduz a capacidade de evasão contra o Defensor atualizado?
3. A adaptação produz robustez que se estende a variantes não utilizadas diretamente no treinamento?
4. A evasão observada no espaço de features permanece quando a variante é materializada e observada na rede?

---

## Desempenho convencional do Defensor

Antes do ciclo adversarial, o Defensor apresentou aproximadamente:

```text
Accuracy:  99.90%
Precision: 99.91%
Recall:    99.91%
F1:        99.91%
FNR:       0.09%
```

Esse resultado funcionava como uma verificação básica: o problema experimental não era ensinar o modelo a reconhecer DDoS convencional, mas estudar sua robustez diante de variantes adversariais construídas especificamente para reduzir a detecção.

---

## H1 — Vulnerabilidade adversarial

Na primeira rodada, o cenário de referência foi:

```text
A1 × D0
```

A taxa de evasão observada foi:

```text
74.9%
```

Dentro daquela execução, isso mostrou que desempenho convencional elevado e robustez adversarial não eram propriedades equivalentes:

```text
bom desempenho convencional
            ≠
robustez adversarial
```

A interpretação da época era restrita ao ambiente experimental: o Defensor reconhecia DDoS convencional com alta precisão, mas ainda possuía regiões de decisão exploráveis pelo Atacante.

---

## H2 — Adaptação do Defensor

A v1.7 passou a medir explicitamente a mesma estratégia antes e depois da adaptação do Defensor.

Alguns exemplos registrados foram:

```text
A1 × D0 → 74.9%
A1 × D1 → 0.3%

A2 × D1 → 42.9%
A2 × D2 → 0.1%

A3 × D2 → 38.4%
A3 × D3 → ~0%
```

Em uma rodada posterior também foi observado:

```text
A16 × D15 → 29.0%
A16 × D16 → ~0%
```

O comportamento era compatível com o ciclo:

```text
Atacante encontra uma evasão
          ↓
Defensor é exposto a ela
          ↓
Defensor é atualizado
          ↓
evasão diminui
```

Isso fornecia evidência de adaptação às variantes apresentadas ao Defensor, mas não era suficiente para demonstrar generalização para variantes realmente reservadas.

---

## Dinâmica não monotônica

A execução também mostrou que a evolução adversarial não era simplesmente monotônica.

Entre algumas rodadas, a evasão pré-adaptação permaneceu praticamente nula:

```text
A10 × D9  ≈ 0%
A11 × D10 ≈ 0%
A12 × D11 ≈ 0%
A13 × D12 ≈ 0%
A14 × D13 ≈ 0%
A15 × D14 ≈ 0%
```

Posteriormente surgiu novamente uma evasão significativa:

```text
A16 × D15 ≈ 29%
```

Na época, os dados não permitiam concluir se isso representava uma nova região adversarial, variação de treinamento ou outro fenômeno. O resultado foi registrado principalmente como motivação para preservar e comparar checkpoints históricos.

---

## Matriz de checkpoints e robustez acumulada

Uma das mudanças centrais da v1.7 foi a avaliação cruzada:

```text
A1..A20 × D0..D20
```

Para o Defensor final `D20`, foram registrados aproximadamente:

```text
evasão média contra atacantes históricos: 0.46%

pior caso:
A3 × D20 → 7.68%
```

A maior parte dos atacantes históricos apresentou evasão próxima de zero contra o Defensor final.

Esse comportamento sugeria retenção de parte da robustez adquirida ao longo do processo. Entretanto, a matriz não demonstrava generalização adversarial em sentido forte, porque os atacantes históricos pertenciam à mesma dinâmica que produziu o Defensor final.

---

## H3 — Generalização permanecia em aberto

A pergunta central era se um Defensor adaptativo responderia melhor também a variantes adversariais que não participaram de sua adaptação.

A infraestrutura da v1.7 ainda não possuía um protocolo experimental completo para essa comparação.

O desenho considerado adequado na época era comparar, a partir de condições equivalentes:

```text
Defensor baseline
vs.
Defensor adaptativo
```

contra três grupos:

```text
ataques convencionais

variantes utilizadas na adaptação

variantes adversariais reservadas
```

Por isso, H3 não era considerada confirmada pela execução registrada neste documento.

---

## Desempenho convencional após adaptação

No conjunto de teste reservado, o Defensor final apresentou aproximadamente:

```text
Accuracy:  99.67%
Precision: 99.50%
Recall:    99.94%
F1:        99.72%
FPR:       0.69%
FNR:       0.06%
ROC-AUC:   0.9998
```

A matriz de confusão registrada foi:

```text
                 Pred. Benigno   Pred. DDoS
Real Benigno          18521          129
Real DDoS                16        25587
```

Naquela execução, o processo de adaptação não produziu um colapso evidente do desempenho convencional do Defensor.

---

## Evasão matemática e tradução válida

A v1.7 também tornou explícita uma diferença importante entre o espaço de features e a materialização em rede.

No dry-run de demonstração `A1 × D0`, em três ciclos de 20 vetores, foram registrados:

```text
60 vetores gerados
46 evasões matemáticas
≈ 76.7%

6 traduções consideradas válidas
= 10% dos vetores gerados
```

Isso mostrou que:

```text
evasão matemática
        ≠
tradução válida
```

Uma perturbação podia enganar o Defensor e ainda assim produzir uma combinação de features que o Translator rejeitava segundo suas regras de consistência.

---

## Checkpoint final

O dry-run também foi executado com os modelos finais:

```text
A20 × D20
```

Nos três ciclos registrados:

```text
60 vetores gerados
0 evasões
0 traduções
0 execuções
```

Esse comportamento era consistente com a avaliação final do treinamento, na qual o Defensor final já havia se adaptado ao Atacante final.

---

## H4 — Transferência entre feature-space e rede

A quarta hipótese investigava se as propriedades observadas no espaço de features permaneciam após a materialização do tráfego.

Na execução de rede controlada utilizada como referência foram registrados, em 10 ciclos:

```text
1000 vetores gerados
756 evasões matemáticas
≈ 75.6%

109 traduções válidas
≈ 10.9% dos vetores gerados
```

Todos os parâmetros encaminhados ao Sender concluíram sua execução, mas surgiu um problema de fidelidade entre o tráfego solicitado e o efetivamente produzido.

Em alguns casos, o Translator solicitava aproximadamente:

```text
20.000–25.000 PPS
```

enquanto o Sender produzia aproximadamente:

```text
22–24 PPS
```

O resultado reforçou a existência de um gap entre:

```text
vetor adversarial
      ↓
tradução válida
      ↓
AttackParams
      ↓
tráfego efetivamente produzido
```

Portanto, uma tradução válida não implicava que o tráfego resultante preservava as propriedades representadas pelo vetor adversarial original.

---

## Lacuna arquitetural observada na v1.7

Naquele momento, a arquitetura conseguia percorrer:

```text
vetor adversarial
      ↓
Translator
      ↓
Sender
      ↓
rede
```

mas ainda não possuía o caminho completo de retorno:

```text
rede
 ↓
captura
 ↓
extração de features
 ↓
vetor reconstruído
 ↓
Defensor
```

Sem esse retorno, não era possível medir de forma independente se o tráfego realmente produzido continuava apresentando as características adversariais esperadas.

Essa limitação foi uma das principais motivações para a evolução arquitetural posterior da ADArena.

> **Nota histórica:** versões posteriores passaram a introduzir abstrações explícitas de captura, extração e inferência sobre tráfego observado. Portanto, a ausência descrita nesta seção representa o estado da **v1.7**, e não o estado atual do projeto.

---

## Duas frentes identificadas durante a v1.7

A análise daquela versão ajudou a separar duas frentes complementares.

### Defesa adaptativa

A frente científica buscava entender em que medida a exposição contínua a variantes adversariais poderia tornar um mecanismo defensivo mais robusto diante de mudanças no comportamento ofensivo.

O problema deixou de ser apenas "fazer duas IAs competirem" e passou a ser visto como o estudo da rigidez de uma defesa diante de um adversário que muda.

### ADArena como ferramenta experimental

A segunda frente passou a tratar a ADArena como infraestrutura para estudar o problema anterior.

Os critérios considerados importantes incluíam:

- modularidade;
- reprodutibilidade;
- observabilidade;
- substituição de atacantes e defensores;
- isolamento entre experimentos;
- fidelidade entre feature-space e rede;
- comparação de estratégias sob condições equivalentes.

Essa separação continuou orientando as refatorações posteriores da arquitetura.

---

## O que a v1.7 permitia afirmar

Dentro das condições daquela execução, havia evidência de que:

- um Defensor com desempenho convencional muito alto ainda podia ser vulnerável a variantes adversariais;
- a exposição às variantes geradas pelo Atacante reduzia fortemente sua evasão contra o Defensor atualizado;
- o desempenho convencional permanecia alto após o ciclo adversarial;
- o Defensor final apresentava baixa evasão média contra a maior parte dos atacantes históricos;
- havia comportamentos não triviais entre checkpoints que justificavam investigação adicional;
- evasão matemática e tradução válida eram medidas diferentes;
- uma tradução válida não garantia fidelidade entre parâmetros desejados e tráfego efetivamente produzido.

A v1.7 **não demonstrava** que:

- o Defensor generalizava melhor contra variantes realmente reservadas;
- os resultados eram estáveis entre múltiplas seeds;
- as perturbações traduzidas preservavam suas características na rede;
- o tráfego capturado continuava evadindo o Defensor após reconstrução das features;
- o ciclo constituía uma defesa adaptativa online;
- a arquitetura era genérica para diferentes famílias de ataque.

---

## Importância histórica da v1.7

A principal contribuição da v1.7 para a evolução do projeto não foi apenas uma métrica específica.

Ela tornou explícitas diferenças que passaram a orientar a arquitetura posterior:

```text
acurácia convencional
        ≠
robustez adversarial

evasão matemática
        ≠
tradução válida

tradução válida
        ≠
tráfego reproduzido
```

Também deixou claro que a ADArena precisava evoluir de uma implementação centrada em modelos e em um Controller para uma ferramenta experimental com componentes substituíveis, protocolos explícitos, rastreabilidade das execuções e um caminho completo entre feature-space e rede.

Os valores deste documento são preservados justamente para registrar essa etapa da evolução do projeto. Eles não devem ser comparados diretamente com execuções posteriores sem considerar as mudanças de arquitetura, protocolo, dataset, checkpoints e implementação.
