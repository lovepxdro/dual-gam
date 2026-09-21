# Arquitetura da ADArena

Este documento descreve a arquitetura **atual** da **ADArena — Adaptive Defense Arena**.

A ADArena é um ambiente experimental para pesquisa em defesa adaptativa baseada em aprendizado adversarial. A arquitetura é deliberadamente modular: atacantes, defensores, datasets, protocolos, mecanismos de tradução, backends de rede, captura, extração de features e mecanismos de controle podem ser substituídos sem reescrever o restante do sistema.

> **Estado deste documento:** documentação viva da linha atual da arquitetura. Resultados e decisões antigas da linha v1.x são preservados separadamente em `docs/history/`.

---

## 1. Arquitetura de referência

A arquitetura é organizada em três responsabilidades principais: **aprendizado**, **observação da rede** e **controle**.

```text
┌──────────────────────────────────────────────────────────────┐
│                     Camada de aprendizado                    │
│                                                              │
│          Attacker  ◄──────────────►  Defender                │
│                                                              │
│   treino adversarial e avaliação de checkpoints históricos   │
└──────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────────────────────────────┐
│                       Camada de rede                         │
│                                                              │
│  h1-h4 ─────┐                                                │
│              ├────► rede Docker ─────────────► h-target      │
│  h-attack ───┘                                                │
│                         │                                    │
│                         ▼                                    │
│                      Capture                                 │
│                         ↓                                    │
│                      Extractor                               │
└─────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                      Plano de controle                       │
│                                                              │
│  Defender                                                    │
│      ↓                                                       │
│  DecisionPolicy                                              │
│      ↓                                                       │
│  RuleEnforcer                                                │
│      ↓                                                       │
│  ação experimental / plano de dados                          │
└──────────────────────────────────────────────────────────────┘
```

O diagrama é uma **referência conceitual**, não uma obrigação de implementação 1:1. Por exemplo, a responsabilidade de monitoramento pode ser implementada por Scapy, por um controlador SDN ou por outro mecanismo de observação.

Na linha atual, a observação física é realizada diretamente sobre a rede Docker. Um controlador SDN real não é necessário para o fechamento da Frente 1.

### Terminologia

Os nomes `Attacker` e `Defender` são preferidos a “GAN atacante” e “GAN defensora”.

Na implementação atual:

- o `Attacker` é um gerador neural de perturbações adversariais;
- o `Defender` é um classificador binário;
- a arquitetura não pressupõe que componentes futuros usem a mesma técnica;
- `DecisionPolicy` decide a reação experimental;
- `RuleEnforcer` materializa a decisão;
- a implementação atual de enforcement é deliberadamente `dry-run`.

---

## 2. Core experimental

A linha atual da ADArena separa o **Core experimental** dos componentes concretos.

```text
ExperimentConfig
      ↓
ExperimentRunner
      ↓
ExperimentProtocol
      ↓
ComponentRegistry
      ↓
┌──────────────────────────────────────────────────────────────┐
│ Dataset / Preprocessor / Attacker / Defender                 │
│ Renderer / NetworkBackend / Capture / Extractor              │
└──────────────────────────────────────────────────────────────┘
```

Essa separação permite que o Runner conheça apenas os contratos da plataforma, enquanto decisões específicas de treinamento, simulação ou observação ficam encapsuladas em protocolos.

### 2.1 `ComponentRegistry`

Centraliza o registro e a criação de componentes.

Entre os tipos atualmente representados estão:

- `ATTACKER`;
- `DEFENDER`;
- `DATASET`;
- `EXPERIMENT_PROTOCOL`;
- `RENDERER`;
- `NETWORK_BACKEND`;
- `CAPTURE`;
- `EXTRACTOR`.

Cada componente declara também as representações de dados que recebe e produz. Isso permite validar cadeias de componentes antes da execução.

Os protocolos built-in atuais incluem:

```text
adarena.adversarial_training
adarena.network_simulation
adarena.network_observation
```

### 2.2 `ExperimentConfig`

Representa declarativamente uma execução.

A configuração inclui, entre outros elementos:

- atacante e defensor;
- dataset;
- protocolo;
- modo de execução;
- seed e device;
- parâmetros de treinamento;
- configuração de rede;
- checkpoints;
- preprocessador persistido;
- `capture_duration`;
- `packet_limit`;
- threshold de classificação;
- `dry_run`.

Nos modos de treinamento e simulação, `attacker` e `attack_dataset` continuam fazendo parte do caminho experimental.

Em `OBSERVE`, ambos podem ser `None`.

Isso permite representar corretamente uma execução que apenas observa tráfego já existente:

```text
OBSERVE
├── attacker = None
├── attack_dataset = None
├── defender = checkpoint persistido
├── preprocessor_source = persistido
├── capture
└── extractor
```

### 2.3 `ExperimentRunner`

É o orquestrador genérico da execução.

Suas responsabilidades incluem:

- validar a configuração;
- criar o diretório da execução;
- resolver componentes pelo Registry;
- preparar dados quando o protocolo exige dataset;
- reutilizar preprocessador persistido quando necessário;
- salvar `config_execucao.json`;
- construir o `ProtocolContext`;
- delegar a execução ao protocolo;
- receber o `ProtocolResult`;
- atualizar a referência `latest` ao final de uma execução bem-sucedida.

O Runner agora é **mode-aware**.

```text
TRAIN / SIMULATE
      ↓
dataset + preparação experimental

OBSERVE
      ↓
sem dataset
sem split
sem novo fit do scaler
      ↓
Preprocessador.carregar(...)
```

Em `OBSERVE`, os campos de split do `ProtocolContext` são `None`.

O Runner não deve conhecer detalhes internos do Attacker, do Defender ou do protocolo científico.

### 2.4 `ExperimentProtocol`

Encapsula a lógica científica ou operacional de uma execução.

Os protocolos principais da linha atual são:

- `AdversarialTrainingProtocol` — treinamento e avaliação adversarial;
- `NetworkSimulationProtocol` — materialização feature-space → rede em `dry-run`;
- `NetworkObservationProtocol` — observação passiva rede → features → Defender → controle.

---

## 3. Camada de aprendizado

### 3.1 Preprocessor

O preprocessador é responsável por tornar os dados compatíveis com o treinamento e a inferência.

Entre suas responsabilidades estão:

- adotar o schema fornecido pelo Dataset;
- remover duplicatas exatas quando configurado;
- separar treino, validação e teste;
- auditar interseções entre os conjuntos;
- ajustar o `StandardScaler` somente sobre o treino;
- persistir `feature_names`, scaler e demais informações necessárias para reproduzir a transformação.

Durante `SIMULATE` e `OBSERVE`, o preprocessador persistido no treinamento é reutilizado.

O scaler não é reajustado.

### 3.2 Attacker

O Attacker atual recebe amostras reais de DDoS e produz perturbações no espaço de features.

Sua implementação atual é substituível pelo Registry. Portanto, a arquitetura não depende do modelo neural existente para definir o papel de “atacante”.

### 3.3 Defender

O Defender atual recebe vetores normalizados e retorna a probabilidade de DDoS.

O caminho de inferência usa um adapter (`BinaryPredictor`) para evitar que os pipelines de rede dependam diretamente de PyTorch.

### 3.4 Treinamento adversarial

O `AdversarialTrainingProtocol` encapsula o ciclo de aprendizado competitivo e a preservação de checkpoints históricos.

A avaliação cruzada entre checkpoints permite comparar pares `An × Dm` sem depender apenas do estado final do treinamento.

---

## 4. Feature-space → rede

A materialização de uma variante adversarial é separada em duas responsabilidades.

```text
vetor adversarial
      ↓
Renderer
      ↓
parâmetros de execução
      ↓
NetworkBackend
      ↓
resultado
```

### 4.1 Renderer

O `Renderer` converte uma representação no espaço de features para uma representação executável pelo backend.

A implementação atual, `TranslatorRenderer`, adapta o Translator legado e produz parâmetros de rede a partir de features adversariais.

Essa etapa diferencia:

```text
evasão matemática
      ≠
tradução válida
```

Uma amostra pode enganar o Defender e ainda assim não produzir uma combinação consistente de parâmetros de rede.

### 4.2 NetworkBackend

O `NetworkBackend` recebe o payload produzido pelo Renderer.

Na linha atual, `NetworkSimulationProtocol` opera exclusivamente em `dry-run`.

Portanto, o protocolo valida:

```text
features
  ↓
Renderer
  ↓
parâmetros
  ↓
Backend dry-run
```

sem transmitir tráfego ofensivo.

---

## 5. Rede → feature-space

O caminho inverso possui contratos formais na arquitetura.

```text
rede
 ↓
Capture
 ↓
CaptureBatch
 ↓
FlowExtractor
 ↓
FlowFeatureBatch
 ↓
FeatureSchemaValidator
 ↓
Preprocessor
 ↓
BinaryPredictor / Defender
```

### 5.1 Capture

A responsabilidade de `Capture` é observar tráfego e produzir registros neutros de pacotes.

A implementação atual inclui `ScapyPacketCapture` e permite também implementações sintéticas para testes.

### 5.2 Extractor

O `FlowExtractor` transforma pacotes capturados em vetores de fluxo.

A implementação atual, `BasicFlowExtractor`, reconstrói o schema estrutural esperado pelo caminho CICIDS2017 e nunca preenche silenciosamente features desconhecidas.

A compatibilidade estrutural atual é:

```text
77 features esperadas
77 features reconstruíveis
```

Isso significa compatibilidade de schema. Não significa equivalência numérica já comprovada com o CICFlowMeter para todas as features.

### 5.3 Validação de schema

Antes de um vetor reconstruído chegar ao modelo, a ADArena verifica:

- nomes esperados;
- ordem das features;
- features ausentes;
- features extras;
- features não reconstruídas;
- valores não finitos.

A inferência só é realizada quando o schema reconstruído é exatamente compatível com o schema esperado pelo preprocessador/modelo.

### 5.4 `NetworkInferencePipeline`

Orquestra:

```text
CaptureBatch
   ↓
Extractor
   ↓
validação de schema
   ↓
normalização
   ↓
Defender
```

---

## 6. Dois caminhos de observação

A arquitetura possui hoje dois mecanismos relacionados, mas com objetivos diferentes.

### 6.1 `NetworkObservationPipeline`

Esse pipeline pertence ao caminho de simulação controlada.

```text
Capture.start()
      ↓
NetworkBackend
      ↓
Capture.stop()
      ↓
NetworkInferencePipeline
```

Ele foi criado para observar o tráfego correspondente a uma execução de backend dentro da mesma unidade experimental.

Como o backend real da linha atual permanece em `dry-run`, esse caminho é mais útil com componentes sintéticos/fakes ou futuras fontes de execução controlada.

### 6.2 `NetworkObservationProtocol`

O protocolo novo representa a observação **passiva** de tráfego já existente.

```text
tráfego existente
      ↓
Capture.capture(...)
      ↓
Extractor
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

Características importantes:

- não requer Attacker;
- não requer Dataset;
- não requer Renderer;
- não requer NetworkBackend;
- não gera tráfego;
- não altera a rede;
- utiliza checkpoints e preprocessador persistidos;
- persiste métricas e decisões como artefatos experimentais.

---

## 7. Decision / Policy e Rule Enforcer

A separação entre detectar, decidir e executar já está implementada.

```text
Defender
   ↓
DecisionPolicy
   ↓
RuleEnforcer
```

### 7.1 `Decision`

Uma decisão registra:

- `flow_id`;
- predição;
- score;
- ação;
- motivo;
- metadados.

As ações mínimas atuais são:

```text
NONE
BLOCK
```

### 7.2 `ThresholdDecisionPolicy`

A política atual usa um threshold explícito.

```text
score < threshold
      ↓
NONE

score >= threshold
      ↓
BLOCK
```

A predição original do Defender é mantida para auditoria.

A política não modifica diretamente a rede.

### 7.3 `DryRunRuleEnforcer`

O enforcer atual registra a decisão sem aplicar alterações reais.

A semântica é:

```text
success = decisão processada
applied = mudança real aplicada
```

Em `dry-run`:

```text
success = True
applied = False
dry_run = True
```

Isso permite validar a arquitetura de resposta sem introduzir enforcement real no laboratório.

### 7.4 `ControlPipeline`

Combina as saídas do Defender com a política e o enforcer.

```text
NetworkInferenceResult
      ↓
DecisionPolicy
      ↓
Decision[]
      ↓
RuleEnforcer
      ↓
EnforcementResult[]
```

### 7.5 `ControlledObservationPipeline`

Combina inferência e controle:

```text
CaptureBatch
      ↓
NetworkInferencePipeline
      ↓
NetworkInferenceResult
      ↓
ControlPipeline
      ↓
ControlledObservationResult
```

Esse caminho já possui validação sintética para os ramos `NONE` e `BLOCK`.

---

## 8. Camada física atual

A infraestrutura física utilizada na linha atual é uma rede Docker bridge isolada.

```text
h1  ─┐
h2  ─┤
h3  ─┼──► rede Docker 172.20.0.0/24 ───► h-target
h4  ─┘
```

Hosts observados:

```text
h-target  172.20.0.10
h1        172.20.0.21
h2        172.20.0.22
h3        172.20.0.23
h4        172.20.0.24
```

O caminho físico foi validado sobre a interface host-side do `h-target`.

Na execução de fechamento de 21/09/2026:

```text
interface: veth91eb7b2
captura:   180 pacotes IP
duração:   5 s
fluxos:    15
features:  77
```

O resultado foi:

```text
15 fluxos benignos
0 fluxos classificados como ataque
0 decisões BLOCK
0 regras aplicadas
dry_run = true
```

Essa execução usou:

```text
ExperimentRunner
      ↓
ExperimentMode.OBSERVE
      ↓
NetworkObservationProtocol
```

e gerou um run experimental completo.

### 8.1 SDN

Um container antigo `h-control-ofc`/OS-Ken ainda pode existir em ambientes de desenvolvimento, mas ele **não faz parte do caminho necessário para a arquitetura atual**.

A Frente 1 não depende de OVS/OS-Ken.

Integração com um controlador SDN real e enforcement no plano de dados podem ser tratados futuramente como extensão.

---

## 9. Feedback experimental

O fechamento da Frente 1 não transforma a ADArena em um sistema de aprendizado online autônomo.

O sistema possui dois caminhos que permanecem conceitualmente separados:

```text
CAMINHO DE SIMULAÇÃO

Attacker
   ↓
Defender
   ↓
Renderer
   ↓
Backend dry-run
```

e:

```text
CAMINHO DE OBSERVAÇÃO

rede física
   ↓
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

Isso permite estudar separadamente:

- comportamento adversarial no espaço de features;
- traduzibilidade das perturbações;
- reconstrução de features da rede;
- comportamento do Defender sobre tráfego observado;
- decisões defensivas resultantes.

Uma futura extensão poderá ligar materialização real controlada e observação física na mesma execução, mas isso não é requisito da arquitetura atual.

---

## 10. Persistência e reprodutibilidade

Cada execução preserva contexto suficiente para ser interpretada posteriormente.

O snapshot atual registra, entre outros:

- modo de execução;
- seed;
- dataset e parâmetros quando aplicável;
- atacante e checkpoint quando aplicável;
- defensor e checkpoint;
- protocolo;
- Renderer;
- NetworkBackend;
- Capture;
- Extractor;
- preprocessador de origem;
- quantidade de amostras;
- `capture_duration`;
- threshold configurado e efetivo;
- `packet_limit`;
- `dry_run`;
- estado de observação.

Essa informação é armazenada em:

```text
config_execucao.json
```

Em `OBSERVE`, campos que não participam da execução são registrados explicitamente como `null`.

Exemplo conceitual:

```text
dataset = null
attacker = null
mode = observe
```

O `NetworkObservationProtocol` persiste também:

```text
metrics/
├── network_observation.json
└── network_observation_evaluations.json
```

As avaliações incluem:

- `flow_ids`;
- probabilidades;
- predições;
- decisões;
- resultados do enforcement;
- metadados de captura.

---

## 11. Mapeamento da arquitetura de referência

| Bloco | Estado | Implementação / observação |
| --- | --- | --- |
| Attacker | Implementado | Componente registrado e protocolo de treinamento |
| Defender | Implementado | Classificador registrado + `BinaryPredictor` |
| Treinamento competitivo | Implementado | `AdversarialTrainingProtocol` |
| Dataset / preprocessing | Implementado | Dataset adapters + preprocessador persistido |
| Renderer | Implementado | `TranslatorRenderer` |
| NetworkBackend | Implementado | Adapter do Sender; simulação em `dry-run` |
| Capture | Implementado | `ScapyPacketCapture`; validado fisicamente |
| Extractor | Implementado | `BasicFlowExtractor`; schema 77/77 |
| Schema validation | Implementado | `FeatureSchemaValidator` |
| Rede → Defender | Implementado | Pipeline + execução física com modelo persistido |
| Topologia Docker | Implementado | h1-h4 → h-target observados fisicamente |
| Decision / Policy | Implementado | `ThresholdDecisionPolicy` |
| Rule Enforcer | Implementado | `DryRunRuleEnforcer` |
| ControlPipeline | Implementado | decisão + enforcement |
| NetworkObservationProtocol | Implementado | protocolo passivo registrado |
| OBSERVE no Runner | Implementado | sem dataset/splits; preprocessor persistido |
| Feedback rede → experimento | Implementado | resultado físico registrado em `ProtocolResult` e artefatos |
| Enforcement SDN real | Futuro | fora do escopo necessário da Frente 1 |

---

## 12. Critério de conclusão da arquitetura — Frente 1

A Frente 1 está arquiteturalmente concluída.

```text
[✓] contratos de Attacker / Defender / Dataset / Protocol
[✓] Registry + Config + Runner genéricos
[✓] Renderer + NetworkBackend
[✓] Capture + Extractor + inferência
[✓] simulação end-to-end em dry-run
[✓] topologia Docker integrada ao fluxo de observação
[✓] observação física rede → features → Defender
[✓] Decision / Policy mínimo
[✓] Rule Enforcer mínimo em dry-run
[✓] ControlPipeline
[✓] ControlledObservationPipeline
[✓] NetworkObservationProtocol
[✓] OBSERVE via ExperimentRunner
[✓] resultado observado registrado pelo protocolo experimental
```

Validação automatizada atual:

```text
98 testes passando
```

Validação física final:

```text
180 pacotes
15 fluxos
15 benignos
0 BLOCK
0 regras aplicadas
```

Melhorias adicionais podem continuar existindo, mas não bloqueiam o encerramento da Frente 1.

---

## 13. Estrutura relevante do projeto

```text
dual-gam/
├── h-attack/
│   ├── adarena/
│   │   ├── control/
│   │   │   ├── base.py
│   │   │   ├── policy.py
│   │   │   ├── enforcer.py
│   │   │   ├── pipeline.py
│   │   │   └── observation.py
│   │   ├── core/
│   │   │   ├── components.py
│   │   │   ├── config.py
│   │   │   ├── experiment.py
│   │   │   └── registry.py
│   │   ├── datasets/
│   │   ├── network/
│   │   ├── protocols/
│   │   │   ├── adversarial.py
│   │   │   ├── network_simulation.py
│   │   │   └── network_observation.py
│   │   └── builtin.py
│   ├── gan/
│   ├── translator/
│   ├── sender/
│   ├── scripts/
│   └── tests/
├── h-target/
├── docker/
├── docs/
│   ├── architecture.md
│   ├── flow.md
│   ├── usage.md
│   └── history/
├── models/
└── README.md
```

Os diretórios legados continuam existindo enquanto componentes antigos são adaptados para os contratos da ADArena.

---

## 14. Próxima frente

Com a arquitetura da Frente 1 fechada, os próximos trabalhos são principalmente de produto e manutenção da ferramenta:

- CLI para selecionar modos e componentes;
- configuração de execução;
- mensagens de erro;
- logging;
- nomenclatura genérica de logs (`run.log` em vez de `train.log`);
- empacotamento e remoção da dependência operacional de `PYTHONPATH`;
- documentação de uso;
- tratamento explícito de componentes legados.

Esses itens não alteram os contratos arquiteturais centrais já validados.

---

## 15. Documentação histórica

Documentos em `docs/history/` registram estados anteriores da pesquisa e **não devem ser interpretados como descrição da implementação atual**.

Em especial:

```text
docs/history/resultados_v1_7.md
```

preserva resultados e limitações da linha v1.7 para permitir acompanhar a evolução da arquitetura.
