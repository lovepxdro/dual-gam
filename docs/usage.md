# Como usar

Este documento descreve o uso atual da **ADArena — Adaptive Defense Arena**.

A ADArena possui duas interfaces sobre a mesma camada de aplicação:

- uma **TUI (Terminal User Interface)** para uso interativo;
- uma **CLI** para uso direto, reproduzível e automatizável.

A lógica experimental não é duplicada entre as interfaces. Ambas produzem ou carregam um `ExperimentConfig` e delegam validação e execução para a mesma camada de aplicação e para o `ExperimentRunner`.

> **Segurança:** a ADArena deve ser utilizada somente em ambientes controlados e autorizados. O modo `simulate` exposto pelas interfaces públicas opera exclusivamente em `dry-run`. O modo `observe` realiza captura passiva e o `RuleEnforcer` atual também permanece em `dry-run`.

---

## 1. Preparar o ambiente

A ADArena utiliza Python 3.11 e `uv` para gerenciamento do ambiente.

Na raiz do projeto:

```bash
uv sync
```

Verifique a instalação:

```bash
uv run adarena version
```

Para visualizar os comandos disponíveis:

```bash
uv run adarena --help
```

A CLI atual possui:

```text
adarena
├── version
├── components
├── train
├── simulate
├── observe
├── runs
│   └── show
└── config
    └── validate
```

Executar sem subcomandos abre a interface interativa:

```bash
uv run adarena
```

---

## 2. Interface interativa

A TUI é organizada em quatro áreas principais:

```text
1  Dashboard
2  Executar
3  Runs
4  Componentes
```

Atalhos globais:

```text
1  Dashboard
2  Executar
3  Runs
4  Componentes

r  atualizar
q  sair
```

### 2.1 Dashboard

Apresenta um resumo do ambiente, incluindo:

- quantidade de runs encontrados;
- distribuição dos runs por modo;
- quantidade de componentes registrados;
- quantidade de configurações TOML encontradas.

### 2.2 Executar

A aba `Executar` possui dois caminhos:

```text
Executar
├── Novo experimento
└── Carregar TOML
```

#### Novo experimento

O construtor interativo permite configurar o experimento sem escrever o TOML manualmente.

O fluxo é:

```text
selecionar modo
      ↓
selecionar componentes
      ↓
configurar parâmetros
      ↓
Validar
   ├── Salvar TOML
   └── Executar
```

Os componentes apresentados nos seletores são obtidos do `ComponentRegistry`. Portanto, a interface não mantém uma lista manual de nomes concretos de Dataset, Attacker ou Defender.

Dependendo do modo, a interface apresenta apenas os grupos relevantes.

```text
TRAIN
  Dataset
  Attacker
  Defender
  Protocol
  parâmetros de treinamento

SIMULATE
  Dataset
  Attacker + checkpoint
  Defender + checkpoint
  Protocol
  Renderer
  NetworkBackend
  Capture
  Extractor
  preprocessador persistido
  parâmetros da simulação
  dry-run obrigatório

OBSERVE
  Defender + checkpoint
  Protocol
  Capture
  Extractor
  preprocessador persistido
  parâmetros da captura
```

`OBSERVE` não exige Dataset nem Attacker.

#### Salvar TOML

O experimento montado pela TUI pode ser persistido, por exemplo, em:

```text
configs/experiment.local.toml
```

O arquivo salvo utiliza a mesma estrutura aceita pela CLI. Assim, um experimento pode ser:

```text
montado visualmente
      ↓
salvo em TOML
      ↓
validado
      ↓
reutilizado pela TUI ou CLI
```

Por padrão, o salvamento não sobrescreve silenciosamente um arquivo existente.

#### Carregar TOML

O caminho anterior continua disponível. A TUI lista os arquivos `.toml` presentes em:

```text
configs/
```

Também é possível informar manualmente outro caminho.

As ações são:

```text
Validar
Executar
```

### 2.3 Runs

A aba `Runs` apresenta as execuções encontradas abaixo de `models/`.

Ao selecionar um run, são exibidos:

- modo;
- data de criação;
- seed;
- propósito;
- caminho dos artefatos;
- componentes utilizados;
- quantidade de arquivos em cada grupo de artefatos.

### 2.4 Componentes

A aba `Componentes` apresenta os componentes registrados no `ComponentRegistry`.

Categorias atuais:

```text
attacker
defender
dataset
experiment_protocol
renderer
network_backend
capture
extractor
```

### 2.5 Nível de detalhe

No construtor da TUI existem duas opções de apresentação:

```text
Normal
Detalhada
```

Essa escolha controla quanto do resumo final é mostrado pela TUI.

Ela **não reduz o conteúdo persistido no log do run**.

Na CLI, o equivalente para aumentar o detalhe mostrado no terminal é a opção global:

```bash
uv run adarena -v train --config ...
```

---

## 3. Configuração de um experimento

A configuração possui três grupos conceituais:

```text
Experimento
├── componentes
├── parâmetros gerais
└── parâmetros específicos do protocolo/rede
```

### 3.1 Componentes

Uma seleção de componente contém:

```text
component_id
source
params
```

- `component_id`: identifica o componente no Registry;
- `source`: aponta para um recurso persistido quando necessário, como dataset ou checkpoint;
- `params`: parâmetros específicos daquela implementação.

Isso permite adicionar novos componentes sem transformar cada detalhe em um campo fixo do Core.

### 3.2 Seed

Exemplo:

```toml
seed = 42
```

A `seed` controla fontes de aleatoriedade utilizadas pelo experimento.

Usar a mesma seed, com o mesmo código, dados e configuração, ajuda a reproduzir a divisão dos dados e outras operações pseudoaleatórias.

Alterar a seed é útil quando se deseja verificar se uma conclusão se mantém sob outras inicializações ou divisões.

### 3.3 Device

```toml
device = "cpu"
```

Valores aceitos pelo Core atual:

```text
cpu
cuda
```

### 3.4 Split

```toml
[split]
test_size = 0.2
validation_size = 0.1
```

- `test_size`: fração reservada para teste;
- `validation_size`: fração reservada para validação.

A soma precisa ser menor que `1`.

No treinamento, o scaler é ajustado somente sobre a parcela de treino.

### 3.5 Classification threshold

```toml
classification_threshold = 0.5
```

O threshold transforma a saída probabilística do Defender em decisão binária.

Conceitualmente:

```text
score < threshold   → benigno
score >= threshold  → ataque
```

Esse valor afeta diretamente a relação entre falsos positivos e falsos negativos e deve ser tratado como parte da configuração experimental, não como uma constante invisível.

A configuração de rede pode definir um threshold próprio; quando ausente, utiliza-se o threshold do treinamento como fallback.

### 3.6 Parâmetros do treinamento

Os principais parâmetros atuais são:

| Parâmetro | Significado |
| --- | --- |
| `noise_dim` | dimensão do vetor de ruído utilizado pelo Attacker atual |
| `lr_defensor` | learning rate do Defender |
| `lr_atacante` | learning rate do Attacker |
| `adam_betas` | coeficientes do otimizador Adam |
| `epsilon` | limite da perturbação usado pelo atacante atual |
| `classification_threshold` | limiar de classificação |
| `epochs_pretrain` | epochs do pré-treino do Defender |
| `epochs_por_rodada` | epochs executadas dentro de cada rodada adversarial |
| `n_rodadas` | quantidade de rodadas competitivas |
| `amostras_por_rodada` | amostras usadas em cada rodada |
| `amostras_avaliacao_adversarial` | amostras reservadas para avaliação adversarial |
| `batch_size` | tamanho do batch |

Esses campos pertencem ao protocolo adversarial atual. Futuros protocolos podem introduzir parâmetros diferentes por meio de suas seleções e contratos próprios.

### 3.7 Parâmetros de rede

Os principais campos são:

| Parâmetro | Significado |
| --- | --- |
| `preprocessor_source` | preprocessador persistido que deve ser reutilizado |
| `sample_count` | quantidade de amostras selecionadas para a simulação |
| `packet_limit` | limite opcional de pacotes observados |
| `capture_duration` | duração da captura passiva |
| `classification_threshold` | threshold usado na inferência de rede |
| `dry_run` | impede aplicação/transmissão real no caminho público atual |
| `observe` | indica que a configuração inclui observação pós-backend quando aplicável |

Parâmetros concretos, como interface de captura ou destino de laboratório, pertencem aos `params` do componente correspondente.

---

## 4. Preparar datasets

O caso experimental atual utiliza dados derivados do **CIC-IDS2017** e trabalha com `FLOW_FEATURES`.

Um exemplo local é:

```text
data/DDoS-Friday-no-metadata.parquet
```

O diretório `data/` não é versionado pelo Git.

### 4.1 Dataset novo não significa suporte automático

A arquitetura foi construída para permitir novos `Dataset` adapters e novas representações, mas isso não significa que qualquer arquivo de qualquer ataque já seja aceito automaticamente.

Para um conjunto novo funcionar, é necessário que:

```text
Dataset
  ↓ representação compatível
Attacker
  ↓ representação compatível
Defender
```

e, se houver etapa de rede:

```text
Attacker
  ↓
Renderer
  ↓
NetworkBackend
```

ou:

```text
Capture
  ↓
Extractor
  ↓
Defender
```

O Core valida a compatibilidade das representações declaradas pelos componentes.

### 4.2 Outros ataques

Um novo ataque baseado nas mesmas `FLOW_FEATURES` pode exigir apenas novos adapters/modelos e, quando necessário, um Renderer específico.

Um domínio diferente pode exigir uma nova representação.

Exemplo conceitual:

```text
FLOW_FEATURES
PACKET_RECORDS
TEXT
PROMPT
HTTP_REQUEST
...
```

As duas últimas são apenas exemplos de extensões possíveis; não estão implementadas na linha atual.

O objetivo arquitetural é que a extensão aconteça adicionando componentes e representações, e não reescrevendo `ExperimentRunner`, Registry, CLI ou TUI.

---

## 5. Validar uma configuração

Exemplos versionados:

```text
configs/
├── train.example.toml
├── simulate.example.toml
└── observe.example.toml
```

Configurações locais podem utilizar nomes como:

```text
configs/observe.local.toml
configs/train.smoke.toml
configs/experiment.local.toml
```

Para validar sem executar:

```bash
uv run adarena config validate \
  configs/train.example.toml
```

A validação verifica:

- estrutura do TOML;
- campos conhecidos;
- tipos;
- componentes registrados;
- tipos de componente;
- compatibilidade de representações;
- regras semânticas do modo.

Antes da execução, o **preflight** também verifica recursos externos necessários, como arquivos de dataset, checkpoints, preprocessadores e interfaces de captura.

---

## 6. TRAIN

Use:

```toml
mode = "train"
```

Exemplo pela CLI:

```bash
uv run adarena train \
  --config configs/train.example.toml
```

Fluxo:

```text
Dataset
   ↓
Preprocessor
   ↓
split treino / validação / teste
   ↓
pré-treino do Defender
   ↓
ciclo adversarial
   ↓
checkpoints históricos
   ↓
avaliação cruzada
   ↓
avaliação final
   ↓
artefatos
```

Cada execução recebe um `run_id`, por exemplo:

```text
run_YYYYMMDD_HHMMSS_seed42
```

---

## 7. Smoke test

`smoke` **não é um modo da ADArena**.

É uma forma de executar um fluxo completo com parâmetros reduzidos para responder à pergunta:

```text
"as partes principais continuam integradas?"
```

Exemplo:

```text
configs/train.smoke.toml
```

Um smoke pode reduzir:

- epochs;
- rodadas;
- amostras;
- duração.

O objetivo é validar integração e detectar regressões rapidamente.

Um smoke **não deve ser tratado como resultado científico final**.

---

## 8. SIMULATE — dry-run

Use:

```toml
mode = "simulate"

[network]
dry_run = true
```

Exemplo:

```bash
uv run adarena simulate \
  --config configs/simulate.example.toml
```

Fluxo:

```text
Dataset
   ↓
Preprocessor persistido
   ↓
Attacker persistido
   ↓
variantes adversariais
   ↓
Defender persistido
   ↓
seleção das evasões matemáticas
   ↓
Renderer
   ↓
traduções válidas
   ↓
NetworkBackend
   ↓
dry-run
```

As interfaces públicas rejeitam:

```toml
dry_run = false
```

O objetivo desse modo é testar o caminho:

```text
feature-space → representação materializável → backend
```

sem executar tráfego ofensivo real.

---

## 9. OBSERVE — observação passiva

O modo `observe` classifica fluxos reconstruídos a partir de tráfego que já existe no ambiente.

```text
rede
 ↓
Capture
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
métricas + artefatos
```

Requisitos principais:

- checkpoint do Defender;
- preprocessador persistido;
- Capture;
- Extractor;
- interface acessível quando explicitamente configurada.

Não exige:

- Dataset;
- Attacker;
- Renderer;
- NetworkBackend.

O `RuleEnforcer` atual registra decisões, mas não altera o plano de dados.

---

## 10. Testes sintéticos e componentes fake

A suíte automatizada utiliza componentes fake/sintéticos quando o objetivo é testar contratos e integração sem depender do ambiente físico.

Exemplo conceitual:

```text
Renderer
   ↓
FakeNetworkBackend
   ↓
resultado controlado
```

ou:

```text
Capture fake
   ↓
Extractor
   ↓
Defender fake
   ↓
Policy
   ↓
DryRunRuleEnforcer
```

Isso **não é um modo público da ferramenta**.

É uma estratégia de teste para validar:

- contratos;
- compatibilidade;
- caminhos de sucesso e erro;
- ramo `NONE`;
- ramo `BLOCK`;
- integração entre camadas.

---

## 11. Diferença entre os níveis de validação

```text
UNIT / SYNTHETIC TEST
  componentes isolados ou fake
  objetivo: validar contratos

SMOKE
  execução pequena
  objetivo: validar integração real do software

SIMULATE
  pipeline feature-space → backend em dry-run
  objetivo: validar tradução/materialização sem transmissão

OBSERVE
  rede física/passiva → modelo → controle dry-run
  objetivo: validar o caminho físico defensivo

TRAIN
  treinamento experimental
  objetivo: produzir e avaliar modelos/checkpoints
```

Esses conceitos são complementares e não devem ser tratados como nomes alternativos para a mesma coisa.

---

## 12. Logging

### CLI

Por padrão:

```bash
uv run adarena train \
  --config configs/train.example.toml
```

mostra apenas as informações operacionais mais importantes.

Para logs técnicos detalhados no terminal:

```bash
uv run adarena -v train \
  --config configs/train.example.toml
```

`-v` é uma opção global.

### TUI

A opção:

```text
Normal / Detalhada
```

controla o nível do resumo apresentado na interface.

Cada run mantém um arquivo técnico próprio:

```text
logs/run.log
```

O arquivo é independente do nível de resumo escolhido na TUI.

---

## 13. Inspecionar runs

Listar:

```bash
uv run adarena runs
```

Inspecionar:

```bash
uv run adarena runs show \
  run_YYYYMMDD_HHMMSS_seed42
```

A inspeção apresenta, conforme disponível:

- modo;
- data;
- seed;
- propósito;
- caminho;
- componentes;
- quantidade de artefatos.

A TUI fornece a mesma inspeção pela aba `Runs`.

---

## 14. Estrutura dos runs

Em geral:

```text
<output_dir>/
├── experiments/
│   └── run_YYYYMMDD_HHMMSS_seed42/
│       ├── checkpoints/
│       ├── metrics/
│       ├── plots/
│       ├── logs/
│       │   └── run.log
│       ├── preprocessador/
│       ├── config_execucao.json
│       └── ...
└── latest -> experiments/run_...
```

O conteúdo exato depende do protocolo.

`config_execucao.json` preserva o snapshot efetivo da execução.

O TOML salvo pela TUI representa a **configuração solicitada**; `config_execucao.json` representa a **configuração efetivamente registrada no run**, junto do contexto produzido durante a execução.

---

## 15. Fluxo recomendado

Pela TUI:

```text
1. uv run adarena
       ↓
2. Executar → Novo experimento
       ↓
3. selecionar modo e componentes
       ↓
4. configurar parâmetros
       ↓
5. Validar
       ↓
6. Salvar TOML, se desejar
       ↓
7. Executar
       ↓
8. Runs → inspecionar resultado
```

Pela CLI:

```text
1. preparar/copiar TOML
       ↓
2. adarena config validate
       ↓
3. adarena train | simulate | observe
       ↓
4. adarena runs
       ↓
5. adarena runs show <run_id>
       ↓
6. analisar métricas, logs e artefatos
```

As duas interfaces convergem para:

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

---

## 16. Scripts legados

A linha anterior do projeto utilizava principalmente:

```text
scripts/run.sh
```

para diferentes etapas.

Na linha atual, a interface principal é:

```bash
uv run adarena
```

e os subcomandos da CLI.

Os scripts antigos podem permanecer temporariamente como referência histórica ou compatibilidade, mas não devem ser tratados como a interface principal da ADArena atual.

Documentação histórica é preservada em:

```text
docs/history/
```
