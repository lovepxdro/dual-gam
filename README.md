# ADArena — Ambiente Experimental para Simulação de Ataques e Defesa Adaptativa

> **Nota:** O nome do repositório ainda é 'Dual-GAM', mas a ferramenta agora possui um nome.

Repositório da **Iniciação Científica desenvolvida na CESAR School**.

**ADArena (Adaptive Defense Arena)** é um ambiente experimental para pesquisa em **defesa adaptativa baseada em aprendizado adversarial**. O projeto possui duas frentes complementares:

1. investigar como mecanismos defensivos respondem e se adaptam a variantes adversariais construídas para evadi-los;
2. desenvolver uma arquitetura modular que funcione como **ferramenta de experimentação**, permitindo substituir e comparar estratégias de ataque, modelos defensivos, condições de rede e mecanismos de controle.

DDoS é utilizado como o primeiro caso experimental. O objetivo, portanto, não é construir o melhor detector de DDoS possível, mas utilizar um cenário controlado para estudar **vulnerabilidade adversarial, adaptação, generalização e transferência entre o espaço de features e a rede**.

> **Nota terminológica:** a proposta inicial do projeto descrevia dois modelos adversariais como duas GANs. A implementação atual não corresponde a uma configuração clássica de duas GANs: o componente atacante atua como um **gerador neural de perturbações adversariais**, enquanto o componente defensor é um **classificador binário**. A arquitetura é deliberadamente modular para que ambos possam ser substituídos por outras estratégias no futuro.

Este README fornece uma introdução ao projeto. Para detalhes sobre **arquitetura, fluxo de execução, estrutura do repositório e uso**, consulte o diretório `docs/`.

---

## Versões

| Versão | Alteração |
| ------ | --------- |
| v1.0 | Implementação inicial |
| v1.7 | Consolidação da linha v1.x |

---

## Referências

- **Dataset:** CIC-IDS2017 — Universidade de New Brunswick. [Download](https://www.kaggle.com/datasets/dhoogla/cicids2017)
- **Survey:** Alauthman et al. (2026). *Generative Adversarial Networks for Intrusion Detection Systems: A Comprehensive Survey.* Arabian Journal for Science and Engineering. [Link](https://link.springer.com/article/10.1007/s13369-026-11103-6)
- **Experimento inicial:** notebook `teste_dualGam_ic.ipynb` + [post no blog](https://lovepxdro.github.io/sec-lounge/experimentos/experimento-inicial-ic/)
- **Implementação inicial da arquitetura (v1):** [post no blog](https://lovepxdro.github.io/sec-lounge/experimentos/arquitetura-dual-gam/)
- **Releitura das hipóteses científicas:** [post no blog](https://lovepxdro.github.io/sec-lounge/experimentos/hipotese-cientifica-ic/)
- **Consolidando a arquitetura (v1.7):** [post no blog](https://lovepxdro.github.io/sec-lounge/experimentos/adarena-consolidando-arquitetura/)
