# Dual-GAM — Arquitetura Adversarial para Detecção de DDoS

Repositório da Iniciação Científica desenvolvida na CESAR School.

Implementação de uma **arquitetura adversarial dual** onde dois modelos de IA competem entre si: uma GAN Atacante que aprende a gerar tráfego DDoS capaz de evadir detecção, e uma GAN Defensora que aprende continuamente a identificar e bloquear esses ataques. O ciclo entre os dois simula a co-evolução real entre atacantes e sistemas de defesa.

Este repositório cobre a **segunda etapa da IC**: implementação da arquitetura de rede completa em Docker, integrando as GANs (desenvolvidas na [etapa inicial](https://colab.research.google.com/drive/1SbbZEeh1o0_U7LG3suet0-27QV8WIV7k?usp=sharing)) com geração de tráfego real via Scapy.

---

## Contexto

A questão central da pesquisa não é "detectar DDoS melhor" — isso já existe e funciona bem. A questão é: **como criar uma defesa que se adapte a ataques que ela nunca viu antes?**

Na etapa inicial, validamos o conceito com um experimento em notebook (Colab): defensor com 99.91% de acurácia em dados estáticos, mas vulnerável a ataques perturbados na primeira rodada (75% de evasão). Após 20 rodadas de co-evolução, o defensor adaptativo convergiu para ~98.3% — pagando um trade-off honesto entre acurácia e robustez.

Nesta etapa, saímos do notebook e avançamos para uma implementação da arquitetura em ambiente Docker, integrando os modelos ao tráfego de rede real.

---

## Arquitetura

```
┌─────────────────────────────────────────────────────┐
│              Camada de rede (Docker)                 │
│                                                      │
│  h1-h4 ──────────────────────────────────────────►  │
│  (hosts legítimos)    bridge ddos-net (s1)   h-target│
│  h-attack ───────────────────────────────────────►  │
│  (fonte do ataque)                                   │
└─────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│              Camada ML — Dual GAN                    │
│                                                      │
│   GAN Atacante  ◄────────────►  GAN Defensora        │
└─────────────────────────────────────────────────────┘
```

### Fluxo de execução

```
[GAN Atacante]   perturba amostras reais de DDoS
      ↓
[GAN Defensora]  avalia probabilidade de detecção
      ↓
[Translator]     desnormaliza vetor → AttackParams (pps, size, flags, dur)
      ↓
[Sender/Scapy]   envia pacotes TCP/UDP reais na bridge Docker
      ↓
[Controller]     registra taxa de evasão → próximo ciclo
```

### Estrutura do projeto

```
dual-gam/
├── h-attack/
│   ├── gan/
│   │   ├── models.py        # Defensor e Atacante (nn.Module)
│   │   ├── trainer.py       # Ciclo co-evolutivo adversarial (20 rodadas)
│   │   └── preprocessing.py # Split treino/teste, normalização e FEATURE_MAP
│   ├── translator/
│   │   └── translator.py    # Vetor GAN (77 features) → AttackParams Scapy
│   ├── sender/
│   │   └── sender.py        # Scapy: UDP flood, SYN flood, TCP flood
│   ├── controller/
│   │   └── controller.py    # Orquestra GAN → Translate → Send → Feedback
│   └── main.py              # Entrypoint: train / attack / dry-run
├── h-target/
│   └── server.py            # Flask: servidor alvo + /metrics
├── docker/
│   ├── Dockerfile.h-attack
│   ├── Dockerfile.h-target
│   └── requirements-attack.txt
├── docker-compose.yml
└── scripts/
    └── run.sh
```

---

## Como usar

> **Pré-requisito:** Docker com permissão de acesso ao socket (root ou grupo docker).

### 1. Baixar o dataset

```bash
mkdir -p data
# Download: https://www.kaggle.com/datasets/dhoogla/cicids2017
# Arquivo: DDoS-Friday-no-metadata.parquet → colocar em data/
```

### 2. Baixar o PyTorch (CPU only — evita timeout de download)

```bash
wget -c https://download.pytorch.org/whl/cpu/torch-2.3.1%2Bcpu-cp311-cp311-linux_x86_64.whl \
     -O docker/torch-2.3.1+cpu-cp311-cp311-linux_x86_64.whl
```

### 3. Treinar os modelos

```bash
chmod +x scripts/run.sh
./scripts/run.sh train ./data/DDoS-Friday-no-metadata.parquet
```

Executa: pré-processamento → pré-treino do Defensor (5 epochs) → 20 rodadas de co-evolução → salva checkpoints por rodada + modelos finais no volume `models`.

### 4. Testar o pipeline sem pacotes reais

```bash
./scripts/run.sh dry-run
```

### 5. Executar ataque real

```bash
./scripts/run.sh attack
```

> Apenas em ambiente de laboratório isolado. O script pede confirmação antes de prosseguir.

### 6. Monitorar o servidor alvo

```bash
# Em outro terminal (não precisa de root):
watch -n1 'curl -s http://localhost:8080/metrics | python3 -m json.tool'
```

### 7. Ver logs / encerrar

```bash
./scripts/run.sh logs h-attack
./scripts/run.sh down
```

---

## Versões

| Versão | Alteração |
|--------|-----------|
| v1.0 | Implementação inicial |
| v1.1 | Correção de data leakage |
| v1.2 | Auditoria do dataset |
| v1.3 | Reprodutibilidade: seed fixa |

---

## Referências

- **Dataset:** CIC-IDS2017 — Universidade de New Brunswick. [Download](https://www.kaggle.com/datasets/dhoogla/cicids2017)
- **Survey:** Alauthman et al. (2026). *Generative Adversarial Networks for Intrusion Detection Systems: A Comprehensive Survey.* Arabian Journal for Science and Engineering. [Link](https://link.springer.com/article/10.1007/s13369-026-11103-6)
- **Experimento inicial:** notebook `teste_dualGam_ic.ipynb` + [post no blog](https://lovepxdro.github.io/sec-lounge/experimentos/experimento-inicial-ic/)
- **Implementação inicial da arquitetura (v1):** [post no blog](https://lovepxdro.github.io/sec-lounge/experimentos/arquitetura-dual-gam/)
