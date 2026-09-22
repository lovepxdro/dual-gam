# Fluxos de execução da ADArena

Este documento descreve os fluxos formais e os níveis de validação da linha atual da **ADArena — Adaptive Defense Arena**.

A plataforma possui três modos públicos principais:

1. **TRAIN** — treinamento experimental;
2. **SIMULATE** — validação feature-space → backend exclusivamente em `dry-run`;
3. **OBSERVE** — observação passiva rede → features → Defender → controle em `dry-run`.

Além dos modos públicos, a suíte utiliza testes sintéticos/fakes e smoke tests. Eles são estratégias de validação, não novos modos da aplicação.

> Para a visão estrutural dos componentes, consulte `architecture.md`. Para operação da CLI/TUI e significado dos parâmetros, consulte `usage.md`.

---

## 1. Entrada da ferramenta

Existem duas formas de construir uma execução.

### 1.1 TUI

```text
usuário
   ↓
TUI Builder
   ↓
seleção de modo
   ↓
seleção de componentes via Registry
   ↓
parâmetros
   ↓
ExperimentConfig
```

O `ExperimentConfig` pode então ser:

```text
validado
salvo como TOML
executado
```

### 1.2 CLI / TOML

```text
TOML
 ↓
Config Loader
 ↓
ExperimentConfig
```

As duas entradas convergem para a mesma camada:

```text
TUI Builder ──► ExperimentConfig ──┐
                                   │
TOML ─────────► Config Loader ─────┤
                                   ▼
                            application.py
                                   ↓
                           ExperimentRunner
                                   ↓
                               Protocol
```

A TUI não implementa uma segunda versão do experimento.

---

## 2. Ciclo geral

```text
ExperimentConfig
      ↓
validação semântica
      ↓
preflight de recursos externos
      ↓
ExperimentRunner
      ↓
criação do run
      ↓
preparação dependente do modo
      ↓
ExperimentProtocol resolvido pelo Registry
      ↓
execução do protocolo
      ↓
ProtocolResult
      ↓
métricas + artefatos + snapshot
```

A validação semântica verifica, entre outros:

- tipo de cada componente;
- compatibilidade de representações;
- regras do modo;
- valores gerais da configuração.

O preflight verifica recursos externos conhecidos antes do início da execução, como:

- dataset;
- checkpoints;
- preprocessador persistido;
- interface de captura.

---

## 3. Registry e seleção de componentes

A interface seleciona componentes registrados, e não classes concretas hardcoded.

```text
ComponentRegistry
      ↓
ComponentSpec
      ├── component_id
      ├── kind
      ├── input_representation
      ├── output_representation
      └── factory
```

O Core verifica compatibilidade entre produtor e consumidor.

Exemplo:

```text
Dataset
output = FLOW_FEATURES
       ↓
Attacker
input  = FLOW_FEATURES
output = FLOW_FEATURES
       ↓
Defender
input  = FLOW_FEATURES
```

No caminho de rede:

```text
Attacker
   ↓
Renderer
   ↓
NetworkBackend
```

e:

```text
Capture
   ↓
Extractor
   ↓
Defender
```

Isso separa duas ideias:

```text
arquitetura genérica
        ≠
componentes concretos universais
```

O núcleo está preparado para novos componentes, mas os built-ins atuais continuam concentrados no caso experimental de IDS baseado em flow features/CIC-IDS2017.

---

## 4. Preparação dependente do modo

```text
TRAIN
  ↓
Dataset
  ↓
Preprocessor
  ↓
split + fit do scaler


SIMULATE
  ↓
Dataset
  ↓
Preprocessor persistido
  ↓
split sem novo fit
  ↓
checkpoints persistidos


OBSERVE
  ↓
sem Dataset
sem Attacker
sem split
  ↓
Preprocessor persistido
  ↓
Defender persistido
```

`OBSERVE` não precisa criar dados artificiais para validar o caminho defensivo.

---

# 5. TRAIN

O treinamento é implementado pelo protocolo adversarial atual.

```text
Dataset
   ↓
Preprocessor
   ↓
train / validation / test
   ↓
pré-treino do Defender
   ↓
D0
   ↓
┌────────────────────────────────────────────┐
│ rodada n                                   │
│                                            │
│ Attacker An treina contra D(n-1)           │
│        ↓                                   │
│ avaliação pré-adaptação                    │
│        ↓                                   │
│ Defender é atualizado                      │
│        ↓                                   │
│ Dn                                         │
│        ↓                                   │
│ avaliação pós-adaptação                    │
│        ↓                                   │
│ checkpoints An e Dn                        │
└────────────────────────────────────────────┘
   ↓
avaliação cruzada de checkpoints
   ↓
avaliação final
   ↓
artefatos
```

## 5.1 Pré-processamento

Regras importantes:

- scaler ajustado somente no treino;
- validação e teste não participam do `fit`;
- preprocessador persistido junto do run;
- nomes e ordem das features preservados.

## 5.2 Checkpoints históricos

A preservação dos estados permite analisar pares como:

```text
A1 × D0
A2 × D1
An × Dm
```

e estudar adaptação, retenção e robustez sem reduzir a análise ao último par.

## 5.3 Smoke de treinamento

Um smoke usa o **mesmo modo TRAIN**, porém com parâmetros reduzidos.

```text
TRAIN normal
  n_rodadas = N
  epochs = N
  amostras = N

TRAIN smoke
  n_rodadas = pequeno
  epochs = pequeno
  amostras = pequeno
```

Portanto:

```text
SMOKE ≠ ExperimentMode
```

O smoke responde se a integração continua funcional. Ele não substitui uma execução científica.

---

# 6. SIMULATE

`NetworkSimulationProtocol` valida o caminho do espaço de features até o backend.

Ele opera exclusivamente em `dry-run`.

```text
Dataset
   ↓
Preprocessor persistido
   ↓
seleção de amostras de ataque
   ↓
Attacker persistido
   ↓
variantes adversariais
   ↓
Defender persistido
   ↓
probabilidades
   ↓
threshold
   ↓
evasões matemáticas
   ↓
Renderer
   ↓
traduções válidas
   ↓
NetworkBackend
   ↓
dry-run
```

## 6.1 Threshold

Com threshold `τ`:

```text
score >= τ  → ataque
score <  τ  → evasão matemática
```

Somente as evasões matemáticas seguem para a etapa de tradução.

## 6.2 Renderer

A arquitetura preserva a distinção:

```text
evasão matemática
      ≠
tradução válida
```

Uma amostra pode evadir o classificador e ainda não poder ser traduzida de forma consistente para a representação esperada pelo backend.

## 6.3 Backend

A interface pública exige:

```text
dry_run = true
```

Portanto, `SIMULATE` valida:

```text
feature-space
    ↓
Attacker
    ↓
Defender
    ↓
Renderer
    ↓
Backend dry-run
```

sem transformar a plataforma em um caminho público de transmissão ofensiva.

---

# 7. Inferência rede → feature-space

O caminho defensivo básico é:

```text
CaptureBatch
   ↓
Extractor
   ↓
FlowFeatureBatch
   ↓
FeatureSchemaValidator
   ↓
Preprocessor persistido
   ↓
BinaryPredictor
   ↓
Defender
```

## 7.1 Capture

`Capture` produz registros neutros dos pacotes observados.

O restante do pipeline não depende diretamente da implementação concreta de captura.

## 7.2 Extractor

O Extractor reconstrói features de fluxo e retorna:

- matriz `X`;
- nomes das features;
- IDs dos fluxos;
- metadados;
- informação sobre cobertura/reconstrução.

## 7.3 Schema

Antes de inferir, o pipeline verifica compatibilidade com o schema esperado.

```text
features esperadas == features observadas
ordem esperada      == ordem observada
valores              finitos
```

A cobertura estrutural do `BasicFlowExtractor` no schema atual é de 77/77 features esperadas.

Isso significa cobertura estrutural do vetor, não equivalência numérica comprovada com o CICFlowMeter.

---

# 8. Controle defensivo

A resposta é separada da detecção.

```text
Defender
   ↓
DecisionPolicy
   ↓
RuleEnforcer
```

## 8.1 ThresholdDecisionPolicy

Recebe:

```text
flow_id
prediction
score
```

e produz uma decisão.

Conceitualmente:

```text
score < threshold
   ↓
NONE

score >= threshold
   ↓
BLOCK
```

## 8.2 DryRunRuleEnforcer

O enforcer atual registra a ação sem alterar o plano de dados.

```text
success = true
applied = false
dry_run = true
```

Essa separação permite testar a lógica de controle sem depender de enforcement real.

---

# 9. OBSERVE

`NetworkObservationProtocol` fecha o caminho físico defensivo.

```text
tráfego já existente
        ↓
Capture.capture(...)
        ↓
Extractor
        ↓
FeatureSchemaValidator
        ↓
Preprocessor persistido
        ↓
Defender persistido
        ↓
DecisionPolicy
        ↓
DryRunRuleEnforcer
        ↓
ProtocolResult
```

Requisitos:

- Defender com checkpoint;
- preprocessador persistido;
- Capture;
- Extractor.

Não exige:

- Attacker;
- Dataset;
- Renderer;
- NetworkBackend.

O tráfego observado pode ser simplesmente o tráfego benigno já produzido pelo laboratório.

---

# 10. Fluxo físico validado

A integração foi validada sobre a rede Docker do laboratório.

```text
h1 ─┐
h2 ─┤
h3 ─┼──► rede Docker ───► h-target
h4 ─┘                         │
                              ▼
                     veth host-side
                              │
                              ▼
                           Capture
                              ↓
                          Extractor
                              ↓
                         Defender
                              ↓
                      DecisionPolicy
                              ↓
                    DryRunRuleEnforcer
```

Em validações físicas recentes, o pipeline capturou tráfego IP existente, reconstruiu flows e classificou o tráfego benigno sem aplicar regras reais.

A interface `veth` deve ser descoberta novamente quando a topologia Docker for recriada, pois seu nome não é permanente.

---

# 11. NetworkObservationPipeline com backend

Existe também um pipeline que liga uma execução de backend à observação da mesma unidade experimental:

```text
Capture.start()
      ↓
Backend.execute_many(...)
      ↓
Capture.stop()
      ↓
NetworkInferencePipeline
```

Na linha atual ele é particularmente útil para testes sintéticos/fakes do contrato.

Ele não deve ser confundido com `OBSERVE`, cujo objetivo é observar passivamente tráfego já existente.

---

# 12. Testes sintéticos e componentes fake

A suíte automatizada substitui componentes físicos por fakes quando isso torna o teste mais determinístico.

Exemplo:

```text
Renderer
   ↓
FakeNetworkBackend
   ↓
NetworkResult conhecido
```

ou:

```text
Capture fake
   ↓
Extractor
   ↓
Predictor controlado
   ↓
DecisionPolicy
   ↓
DryRunRuleEnforcer
```

Isso permite validar:

- contratos;
- compatibilidade;
- tratamento de erros;
- ramo `NONE`;
- ramo `BLOCK`;
- propagação de resultados.

Esses testes não representam um quarto modo público.

---

# 13. Níveis de validação

A nomenclatura recomendada é:

```text
UNIT / SYNTHETIC TEST
  componentes isolados ou fakes
  objetivo: validar contratos

SMOKE
  execução reduzida
  objetivo: validar integração do software

TRAIN
  treinamento experimental
  objetivo: produzir e avaliar modelos

SIMULATE
  feature-space → backend dry-run
  objetivo: validar tradução/materialização

OBSERVE
  rede física/passiva → modelo → controle dry-run
  objetivo: validar o caminho físico defensivo
```

Uma mesma execução pode ser um smoke e, ao mesmo tempo, usar o modo `TRAIN`.

---

# 14. Reprodutibilidade

Existem duas representações importantes da configuração.

## 14.1 TOML

O TOML representa o experimento solicitado.

Ele pode ser:

```text
escrito manualmente
ou
gerado pelo TUI Builder
```

e reutilizado pela CLI ou pela TUI.

## 14.2 config_execucao.json

Cada run registra um snapshot efetivo em:

```text
config_execucao.json
```

Ele preserva, conforme aplicável:

- modo;
- seed;
- componentes;
- checkpoints;
- parâmetros;
- preprocessador;
- configuração de rede;
- threshold;
- metadados do run.

O objetivo é não depender de estado implícito da máquina para interpretar o resultado depois.

---

# 15. Extensibilidade para outros ataques

A arquitetura foi construída para permitir novos domínios sem alterar o Core toda vez.

O objetivo é:

```text
novo ataque
   ↓
novos Dataset / modelos / adapters
   ↓
mesmo Registry
mesmo ExperimentConfig
mesmo ExperimentRunner
mesmas interfaces
```

Mas isso não significa:

```text
"qualquer dataset pode ser colocado e funcionará automaticamente"
```

A compatibilidade depende das representações declaradas pelos componentes.

Um novo cenário baseado em `FLOW_FEATURES` pode reutilizar parte maior do pipeline atual.

Um domínio diferente pode exigir novas representações e componentes.

Exemplo conceitual:

```text
Dataset de rede
   ↓ FLOW_FEATURES
Attacker
   ↓ FLOW_FEATURES
Defender
```

Outro domínio poderia futuramente usar:

```text
Dataset
   ↓ TEXT / PROMPT / outra representação
Attacker específico
   ↓
Defender específico
```

Essas representações adicionais não fazem parte dos built-ins atuais.

O critério arquitetural de sucesso é que a extensão aconteça principalmente nas bordas — Dataset, modelos, Renderer/Extractor e representações — sem reescrever o orquestrador central.

---

# 16. Relação entre os fluxos

```text
TRAIN
=====
Dataset
   ↓
Attacker ↔ Defender
   ↓
checkpoints


SIMULATE
========
Dataset + checkpoints
   ↓
Attacker
   ↓
Defender
   ↓
Renderer
   ↓
Backend dry-run


OBSERVE
=======
tráfego físico existente
   ↓
Capture
   ↓
Extractor
   ↓
Preprocessor persistido
   ↓
Defender
   ↓
DecisionPolicy
   ↓
DryRunRuleEnforcer
```

Os caminhos são independentes por projeto.

`SIMULATE` não precisa transmitir pacotes para validar tradução.

`OBSERVE` não precisa de Attacker para validar detecção e controle.

`TRAIN` não precisa da rede física para produzir checkpoints.

---

# 17. Estado da Frente 1

A Frente 1 cobre a **construção da plataforma**, não a qualidade final dos modelos.

Estado:

```text
[✓] Component contracts
[✓] Registry
[✓] ExperimentConfig
[✓] ExperimentRunner
[✓] Protocol abstraction

[✓] Dataset adapter
[✓] Renderer
[✓] NetworkBackend
[✓] Capture
[✓] Extractor
[✓] schema validation

[✓] NetworkInferencePipeline
[✓] DecisionPolicy
[✓] RuleEnforcer dry-run
[✓] ControlPipeline
[✓] NetworkObservationProtocol

[✓] TRAIN
[✓] SIMULATE dry-run
[✓] OBSERVE passivo
[✓] validação física em Docker

[✓] CLI
[✓] TUI
[✓] TUI Builder
[✓] seleção de componentes via Registry
[✓] configuração de parâmetros
[✓] salvar/recarregar TOML
[✓] inspeção de runs
[✓] logging por run
[✓] testes automatizados
[✓] documentação de uso e fluxo
```

A plataforma pode continuar recebendo melhorias incrementais, mas essas melhorias não impedem o encerramento da Frente 1.

Não fazem parte do fechamento atual:

- enforcement SDN real;
- aprendizado online;
- integração obrigatória com OVS/OS-Ken;
- suporte pronto e automático a qualquer domínio de ataque;
- qualidade científica final dos modelos.

Esses pontos pertencem a extensões futuras ou às próximas frentes.

---

# 18. Próxima frente

Com a plataforma fechada, a atenção passa para os **modelos e experimentos científicos**.

Questões naturais da próxima etapa:

- substituir/evoluir os modelos atuais;
- estudar o comportamento competitivo Attacker × Defender;
- definir experimentos reproduzíveis;
- testar generalização;
- verificar quanto da arquitetura é reutilizado ao introduzir outro cenário;
- separar resultados de engenharia dos resultados científicos.

---

# 19. Documentação histórica

Documentos em:

```text
docs/history/
```

registram estados anteriores da pesquisa.

Eles não devem ser utilizados como descrição da implementação atual quando houver conflito com `architecture.md`, `flow.md` ou `usage.md`.
