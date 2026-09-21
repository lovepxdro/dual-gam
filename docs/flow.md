# Fluxos de execução da ADArena

Este documento descreve os principais fluxos da linha atual da ADArena.

A arquitetura possui hoje três caminhos formais:

1. **treinamento adversarial**;
2. **simulação feature-space → rede em `dry-run`**;
3. **observação passiva rede → feature-space → Defender → controle**.

Os três caminhos passam pelo mesmo Core experimental, mas cada protocolo recebe apenas os recursos de que realmente precisa.

> Para a visão estrutural dos componentes, consulte `architecture.md`.

---

## 1. Ciclo geral de uma execução

Todas as execuções novas passam pelo mesmo Core.

```text
ExperimentConfig
      ↓
ExperimentRunner
      ↓
validação da configuração
      ↓
criação do run
      ↓
preparação dependente do modo
      ↓
ExperimentProtocol pelo Registry
      ↓
execução do protocolo
      ↓
ProtocolResult
      ↓
artefatos + métricas + snapshot
```

### 1.1 `ExperimentRunner`

O Runner é responsável por preparar a unidade experimental, mas não decide a lógica científica da execução.

Ele:

- valida a configuração;
- cria `models/.../experiments/<run_id>`;
- prepara dados quando o modo exige dataset;
- carrega preprocessador persistido quando o modo exige reutilização;
- salva `config_execucao.json`;
- instancia o protocolo selecionado;
- entrega um `ProtocolContext` ao protocolo;
- recebe um `ProtocolResult`;
- atualiza a referência `latest` ao final de uma execução bem-sucedida.

### 1.2 Preparação dependente do modo

O Runner não força mais dataset em todas as execuções.

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
split sem novo fit do scaler


OBSERVE
  ↓
sem Dataset
sem split
  ↓
Preprocessor persistido
```

Em `OBSERVE`:

```text
X_train = None
X_val   = None
X_test  = None

y_train = None
y_val   = None
y_test  = None

dataset_data = None
```

---

# 2. Treinamento adversarial

O treinamento é implementado por `AdversarialTrainingProtocol`.

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
│ evasão pré-adaptação                       │
│        ↓                                   │
│ Defender aprende com adversariais          │
│        ↓                                   │
│ Dn                                         │
│        ↓                                   │
│ evasão pós-adaptação                       │
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

## 2.1 Pré-processamento

O preprocessador recebe o schema fornecido pelo Dataset e realiza o split experimental.

Regras metodológicas importantes:

- o scaler é ajustado somente sobre o treino;
- validação e teste não participam do `fit`;
- o preprocessador utilizado é persistido junto da execução;
- a ordem e os nomes das features são preservados para uso posterior.

## 2.2 Pré-treino do Defender

Antes do ciclo competitivo, o Defender aprende a distinguir tráfego convencional.

Esse estado inicial é preservado como `D0`.

## 2.3 Rodadas adversariais

Em uma rodada `n`:

```text
An × D(n-1)
      ↓
evasão pré-adaptação
      ↓
Defender é atualizado
      ↓
An × Dn
      ↓
evasão pós-adaptação
```

Essa separação permite medir a reação do Defender à estratégia observada naquela rodada.

## 2.4 Checkpoints históricos

Os checkpoints são preservados para permitir avaliações cruzadas.

```text
A1..An × D0..Dn
```

A matriz resultante é usada para estudar retenção, adaptação e robustez acumulada sem reduzir a análise ao par final.

## 2.5 Artefatos

O protocolo de treinamento produz, entre outros:

```text
checkpoints/
metrics/
plots/
config_execucao.json
historico_treino.json
matriz_checkpoints.json
matriz_checkpoints.csv
summary.json
attack_samples.pt
ddos_samples.pt
preprocessador/
```

---

# 3. Simulação feature-space → rede

O fluxo é implementado por `NetworkSimulationProtocol`.

Ele opera **exclusivamente em `dry-run`** na linha atual.

```text
Dataset real
   ↓
preprocessador persistido do treinamento
   ↓
seleção de amostras DDoS
   ↓
Attacker selecionado
   ↓
variantes adversariais
   ↓
Defender selecionado
   ↓
classificação
   ↓
somente evasões matemáticas
   ↓
Renderer
   ↓
somente traduções válidas
   ↓
NetworkBackend em dry-run
```

## 3.1 Reutilização do preprocessador

Em `SIMULATE`, o Runner carrega o preprocessador indicado por `network.preprocessor_source`.

O objetivo é garantir que a simulação use exatamente:

- o mesmo schema;
- a mesma ordem de features;
- o mesmo scaler do treinamento.

O scaler não é reajustado durante a simulação.

## 3.2 Seleção dos checkpoints

Attacker e Defender possuem `source` explícito na configuração.

Isso permite executar cenários como:

```text
A1 × D0
A2 × D1
An × Dm
```

sem depender apenas dos checkpoints finais.

## 3.3 Geração de variantes

O protocolo seleciona amostras de ataque do conjunto de teste e chama o Attacker em modo de inferência.

As variantes produzidas são avaliadas pelo Defender selecionado.

Com threshold `τ`:

```text
P(DDoS) >= τ  → detectado como ataque
P(DDoS) <  τ  → evasão matemática
```

## 3.4 Renderer

Somente evasões matemáticas seguem para o Renderer.

```text
evasão matemática
      ↓
Renderer
      ↓
tradução válida ou rejeição
```

A arquitetura mantém explicitamente a distinção:

```text
evasão matemática
      ≠
tradução válida
```

## 3.5 Backend

As traduções válidas seguem para o `NetworkBackend`.

Na linha atual:

```text
dry_run = True
```

é obrigatório para esse protocolo.

Portanto, essa etapa valida o caminho de materialização sem transmitir tráfego ofensivo.

## 3.6 Métricas principais

O protocolo registra, entre outras:

- quantidade de amostras selecionadas;
- evasões matemáticas;
- taxa de evasão matemática;
- traduções válidas;
- taxa de tradução sobre o total;
- taxa de tradução condicionada à evasão;
- execuções do backend em `dry-run`;
- threshold utilizado;
- epsilon;
- estado da observação.

## 3.7 Artefatos

```text
network/
├── selected_attack_samples.npy
├── adversarial_samples.npy
└── simulation_summary.json
```

---

# 4. Inferência rede → feature-space

O caminho base de inferência é implementado por `NetworkInferencePipeline`.

```text
CaptureBatch
   ↓
Extractor
   ↓
FlowFeatureBatch
   ↓
FeatureSchemaValidator
   ↓
Preprocessor
   ↓
BinaryPredictor
   ↓
Defender
```

## 4.1 Capture

`Capture` produz uma representação neutra dos pacotes observados.

O restante do pipeline não depende diretamente de Scapy.

## 4.2 Extractor

O Extractor agrupa os registros observados e reconstrói as features necessárias.

A saída inclui:

- matriz `X`;
- nomes das features;
- identificadores dos fluxos;
- features que não puderam ser reconstruídas;
- metadados da extração.

## 4.3 Compatibilidade antes da inferência

O Defender não recebe automaticamente qualquer vetor reconstruído.

Primeiro a arquitetura verifica:

```text
features esperadas == features observadas
ordem esperada      == ordem observada
nenhuma feature ausente
nenhuma feature extra
nenhuma feature não suportada
valores finitos
```

Se o schema não for exatamente compatível, a inferência é interrompida.

## 4.4 Normalização e classificação

Somente após a validação:

```text
features reconstruídas
      ↓
preprocessador persistido
      ↓
features normalizadas
      ↓
Defender
      ↓
probabilidade + classe
```

A cobertura estrutural atual do `BasicFlowExtractor` é:

```text
77 / 77 features esperadas
```

Essa cobertura representa compatibilidade de schema, não uma afirmação de paridade numérica total com o CICFlowMeter.

---

# 5. Controle defensivo

A ADArena separa explicitamente detecção, decisão e resposta.

```text
Defender
   ↓
DecisionPolicy
   ↓
RuleEnforcer
```

## 5.1 `ThresholdDecisionPolicy`

A política recebe:

```text
flow_id
prediction
score
```

e produz uma `Decision`.

Com threshold `τ`:

```text
score < τ
   ↓
NONE

score >= τ
   ↓
BLOCK
```

## 5.2 `DryRunRuleEnforcer`

O enforcer atual nunca altera o plano de dados.

Ele registra o resultado como:

```text
success = True
applied = False
dry_run = True
```

para decisões processadas corretamente.

Isso permite validar o contrato do controle sem exigir enforcement real.

## 5.3 `ControlPipeline`

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

O resultado expõe, entre outros:

```text
n_flows
block_count
applied_count
```

## 5.4 `ControlledObservationPipeline`

Une inferência e controle:

```text
CaptureBatch
      ↓
NetworkInferencePipeline
      ↓
ControlPipeline
      ↓
ControlledObservationResult
```

O caminho foi validado sinteticamente com os dois ramos:

```text
NONE
BLOCK
```

O ramo `BLOCK` sintético continua em `dry-run`, portanto:

```text
applied = False
```

---

# 6. `NetworkObservationPipeline`

O `NetworkObservationPipeline` permanece como caminho ligado a uma execução de backend.

```text
Capture.start()
      ↓
Backend.execute_many(...)
      ↓
Capture.stop()
      ↓
NetworkInferencePipeline.infer(...)
```

A captura começa antes do backend para observar o tráfego correspondente à mesma unidade experimental.

Esse pipeline continua útil para testes sintéticos/fakes e futuras execuções controladas.

Ele não é o mecanismo principal para observar o tráfego benigno já existente no laboratório.

---

# 7. `NetworkObservationProtocol`

A observação física passiva é implementada por `NetworkObservationProtocol`.

```text
tráfego já existente
        ↓
Capture.capture(...)
        ↓
NetworkInferencePipeline
        ↓
Defender
        ↓
ControlPipeline
        ↓
DryRunRuleEnforcer
        ↓
ProtocolResult
```

## 7.1 Requisitos

`OBSERVE` exige:

- Defender com `source`;
- preprocessador persistido;
- `Capture`;
- `Extractor`;
- `dry_run=True`.

Ele **não exige**:

- Attacker;
- Dataset;
- Renderer;
- NetworkBackend.

## 7.2 Preparação pelo Runner

```text
ExperimentRunner
      ↓
ExperimentMode.OBSERVE
      ↓
Preprocessador.carregar(...)
      ↓
ProtocolContext
```

O Runner não carrega dataset, não cria split e não reajusta o scaler.

## 7.3 Captura

O protocolo chama:

```text
Capture.capture(
    duration=capture_duration,
    packet_limit=packet_limit
)
```

Depois entrega o `CaptureBatch` ao pipeline controlado.

## 7.4 Resultado

O protocolo registra:

- quantidade de pacotes capturados;
- fluxos reconstruídos;
- quantidade classificada como ataque;
- quantidade classificada como benigno;
- taxa de ataque observada;
- decisões `BLOCK`;
- regras efetivamente aplicadas;
- threshold;
- duração de captura;
- `packet_limit`;
- estado `dry_run`.

Também preserva:

- `flow_ids`;
- probabilidades;
- predições;
- decisões;
- resultados do enforcer;
- metadados da captura.

---

# 8. Fluxo físico validado

A integração física foi validada no laboratório Docker.

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

A primeira captura física não precisou envolver tráfego malicioso.

Os hosts `h1`–`h4` já produzem tráfego HTTP benigno contra `h-target`.

## 8.1 Smoke físico final

Execução de 21/09/2026:

```text
run:
run_20260921_090931_seed42

interface:
veth91eb7b2

captura:
180 pacotes IP

duração:
5 s

fluxos reconstruídos:
15
```

Classificação:

```text
0 ataque
15 benigno
```

Controle:

```text
0 decisões BLOCK
0 regras aplicadas
dry_run = true
```

O resultado confirma fisicamente:

```text
Docker
  ↓
Capture
  ↓
Extractor
  ↓
77 features
  ↓
Preprocessor persistido
  ↓
Defender persistido
  ↓
DecisionPolicy
  ↓
DryRunRuleEnforcer
```

---

# 9. Artefatos de `OBSERVE`

O run físico gerou:

```text
models/network-observe-smoke/experiments/
└── run_20260921_090931_seed42/
    ├── config_execucao.json
    ├── preprocessador/
    ├── metrics/
    │   ├── network_observation.json
    │   └── network_observation_evaluations.json
    └── logs/
        └── train.log
```

`train.log` é atualmente apenas um nome legado do arquivo de log.

Uma melhoria futura da ferramenta é renomeá-lo para:

```text
run.log
```

para representar igualmente `TRAIN`, `SIMULATE` e `OBSERVE`.

---

# 10. Relação entre os fluxos

A arquitetura atual pode ser resumida assim:

```text
TREINAMENTO
===========

Dataset
   ↓
Attacker ↔ Defender
   ↓
checkpoints


SIMULAÇÃO
=========

Dataset + checkpoints
   ↓
Attacker
   ↓
Defender
   ↓
Renderer
   ↓
Backend dry-run


OBSERVAÇÃO
==========

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

`SIMULATE` não precisa transmitir pacotes para validar o caminho de tradução.

`OBSERVE` não precisa de Attacker para validar o caminho físico de detecção e controle.

---

# 11. Feedback experimental

O fechamento arquitetural não exige aprendizado online.

O que já existe:

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

e:

```text
rede física
    ↓
Capture
    ↓
Extractor
    ↓
Defender
    ↓
controle
```

Esses caminhos permitem estudar separadamente:

- evasão matemática;
- traduzibilidade;
- reconstrução de features;
- comportamento do modelo sobre tráfego físico;
- política de resposta;
- resultado do enforcement.

Uma futura extensão pode conectar uma materialização de rede controlada diretamente à observação física da mesma execução.

Isso não é requisito para o fechamento atual da Frente 1.

---

# 12. Persistência e reprodutibilidade

Cada execução deve preservar contexto suficiente para ser interpretada posteriormente.

O snapshot atual registra, entre outros:

- modo de execução;
- seed;
- dataset e parâmetros quando aplicável;
- Attacker e checkpoint quando aplicável;
- Defender e checkpoint;
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

Em `OBSERVE`, elementos não utilizados aparecem explicitamente como `null`.

O objetivo é separar claramente:

```text
resultado científico
      de
estado implícito da máquina / código
```

---

# 13. Estado da Frente 1

```text
[✓] Attacker / Defender
[✓] Registry + Config + Runner
[✓] Protocol abstraction
[✓] Renderer + NetworkBackend
[✓] Capture + Extractor
[✓] NetworkInferencePipeline
[✓] DecisionPolicy
[✓] RuleEnforcer dry-run
[✓] ControlPipeline
[✓] ControlledObservationPipeline
[✓] Docker network operacional
[✓] captura física passiva
[✓] rede → features
[✓] rede → Defender
[✓] Defender → Policy → Enforcer
[✓] NetworkObservationProtocol
[✓] OBSERVE via ExperimentRunner
[✓] resultado físico registrado em artefatos
```

Validação automatizada atual:

```text
98 testes passando
```

A Frente 1 está concluída.

Enforcement SDN real, aprendizado online e integração com OVS/OS-Ken são extensões futuras, não requisitos pendentes dessa frente.

---

# 14. Próxima frente

Com a arquitetura fechada, o trabalho passa para acabamento da ferramenta:

```text
CLI
 ↓
configuração amigável
 ↓
logging
 ↓
erros e validações
 ↓
empacotamento
 ↓
documentação de uso
```

Pontos específicos:

- comandos para `train`, `simulate` e `observe`;
- listagem de componentes registrados;
- padronização do log para `run.log`;
- melhoria das mensagens de erro;
- redução da dependência de `PYTHONPATH`;
- documentação em `usage.md`;
- tratamento explícito de código legado.

---

# 15. Documentação histórica

Resultados antigos não devem ser usados para descrever o comportamento atual da ADArena.

Documentos como:

```text
docs/history/resultados_v1_7.md
```

existem para registrar a evolução da pesquisa, incluindo limitações e resultados que já foram superados ou alterados por novas versões da arquitetura.
