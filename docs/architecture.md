# Arquitetura

Este documento descreve a arquitetura atual da **ADArena**.

A ferramenta é organizada em três camadas principais: **aprendizado adversarial**, **controle** e **rede**.

```text
┌──────────────────────────────────────────────────────────────┐
│                 Camada de aprendizado adversarial            │
│                                                              │
│  Preprocessing → Atacante → Defensor → Trainer              │
│                         │            │                        │
│                         └─ checkpoints / métricas / matriz ──┘│
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                     Camada de controle                        │
│                                                              │
│  Controller → Translator → validação → Sender/Scapy         │
│       │                                      │               │
│       └──────── histórico e métricas por execução ───────────┘│
└───────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────┐
│                    Camada de rede (Docker)                    │
│                                                              │
│  h1-h4 ────────────────────────► h-target                    │
│  h-attack ─── bridge ddos-net ─► h-target                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Camada de aprendizado adversarial

Responsável pelo experimento no espaço de features.

### Preprocessor

Carrega o dataset e prepara os dados para treinamento e avaliação.

Entre suas responsabilidades estão:

- remoção de duplicatas exatas quando necessário;
- divisão em treino, validação e teste;
- auditoria de interseções entre os conjuntos;
- ajuste do `StandardScaler` exclusivamente sobre o treino;
- persistência das informações necessárias para reutilizar o mesmo pré-processamento posteriormente.

### Atacante

Recebe amostras reais de DDoS e produz perturbações limitadas com o objetivo de reduzir a probabilidade de detecção pelo Defensor.

Na implementação atual, esse componente funciona como um **gerador neural de perturbações adversariais**, e não como uma GAN clássica completa.

### Defensor

Classificador binário responsável por distinguir tráfego benigno e DDoS.

O modelo pode ser atualizado ao longo do ciclo adversarial a partir das variantes produzidas pelo Atacante.

### Trainer

Orquestra o treinamento adversarial.

É responsável por:

- pré-treino do Defensor;
- treinamento alternado entre Atacante e Defensor;
- avaliação de evasão antes e depois da adaptação;
- salvamento dos checkpoints;
- avaliação cruzada entre atacantes e defensores históricos;
- geração das métricas utilizadas para análise do experimento.

### Reporting

Persiste os artefatos produzidos durante o treinamento, incluindo:

- métricas do Atacante;
- métricas do Defensor;
- matriz Atacante × Defensor;
- resumo da execução;
- gráficos;
- configurações do experimento.

---

## Camada de controle

Responsável por conectar o espaço de features à execução experimental.

### Controller

Orquestra a execução de ataques a partir dos modelos selecionados.

O fluxo atual é:

```text
Atacante
   ↓
Defensor
   ↓
Translator
   ↓
Sender
```

O Controller:

- seleciona os checkpoints;
- solicita a geração de novas variantes;
- identifica quais vetores evadiram o Defensor;
- envia apenas essas evasões ao Translator;
- encaminha somente traduções válidas ao Sender;
- registra métricas e históricos da execução.

### Translator

Converte features selecionadas do vetor adversarial para parâmetros de rede representados por `AttackParams`.

Entre os parâmetros traduzidos estão:

- PPS;
- tamanho de pacote;
- duração;
- protocolo;
- flags.

O Translator também aplica verificações de consistência e pode rejeitar vetores que sejam matematicamente evasivos, mas não produzam uma tradução considerada válida para a execução.

### Sender

Materializa os `AttackParams` em tráfego de rede utilizando Scapy.

Pode operar em dois modos:

- `dry-run`: simula a execução e não envia pacotes;
- execução real: produz tráfego dentro da rede Docker do laboratório.

O Sender também registra diferenças entre parâmetros solicitados e o tráfego que conseguiu produzir.

### Histórico de execução

Cada `attack` ou `dry-run` gera um arquivo independente em:

```text
/models/experiments/<run_id>/network_runs/
```

Isso permite preservar execuções distintas sem sobrescrever resultados anteriores.

---

## Camada de rede

A camada de rede utiliza Docker para criar um ambiente experimental controlado.

Principais componentes:

- `h-attack`: origem do tráfego ofensivo;
- `h1`–`h4`: hosts auxiliares/legítimos;
- `h-target`: servidor alvo;
- `ddos-net`: bridge Docker compartilhada pela topologia.

Na implementação atual, as métricas do Sender representam o tráfego produzido pelo próprio gerador. A captura e reconstrução independente das features diretamente da rede pertencem às etapas futuras da arquitetura.

---

## Estrutura do projeto

```text
dual-gam/
├── h-attack/
│   ├── gan/
│   │   ├── models.py
│   │   ├── trainer.py
│   │   ├── preprocessing.py
│   │   └── reporting.py
│   ├── translator/
│   │   └── translator.py
│   ├── sender/
│   │   └── sender.py
│   ├── controller/
│   │   └── controller.py
│   ├── tests/
│   │   ├── gan/
│   │   ├── translator/
│   │   ├── sender/
│   │   └── controller/
│   ├── pytest.ini
│   └── main.py
├── h-target/
│   └── server.py
├── docker/
│   ├── Dockerfile.h-attack
│   ├── Dockerfile.h-target
│   └── requirements-attack.txt
├── docs/
│   ├── architecture.md
│   ├── flow.md
│   └── usage.md
├── scripts/
│   └── run.sh
├── results/
├── docker-compose.yml
└── README.md
```

Embora a ferramenta agora seja chamada **ADArena**, o nome do repositório e do diretório raiz pode continuar como `dual-gam` por compatibilidade com links e referências anteriores.
