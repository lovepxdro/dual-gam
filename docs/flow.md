# Fluxo de execução

A ADArena possui dois fluxos principais: **treinamento e avaliação adversarial** e **execução em rede**.

Eles são relacionados, mas não formam ainda um ciclo online completamente fechado.

## Treinamento e avaliação adversarial

```text
CIC-IDS2017
     ↓
Preprocessing
     ↓
treino / validação / teste isolados
     ↓
pré-treino do Defensor
     ↓
D0
     ↓
┌──────────────────────────────────────────────┐
│ rodada n                                     │
│                                              │
│ Atacante An gera perturbações contra D(n-1) │
│        ↓                                     │
│ mede evasão pré-adaptação                    │
│        ↓                                     │
│ Defensor é retreinado com adversariais      │
│        ↓                                     │
│ mede evasão pós-adaptação                    │
│        ↓                                     │
│ salva An e Dn                                │
└──────────────────────────────────────────────┘
     ↓
matriz A1..An × D0..Dn
     ↓
métricas + resumo + gráficos
     ↓
teste final reservado
```

### 1. Pré-processamento

O dataset é carregado e separado em três conjuntos:

```text
treino
validação
teste
```

O scaler é ajustado somente sobre o treino.

O conjunto de validação é utilizado durante a avaliação adversarial, enquanto o conjunto de teste permanece reservado para a avaliação convencional final.

### 2. Pré-treino do Defensor

Antes do ciclo adversarial, o Defensor é treinado com dados convencionais.

O primeiro checkpoint é salvo como:

```text
D0
```

### 3. Rodada adversarial

Em cada rodada `n`:

```text
An × D(n-1)
```

o Atacante aprende a gerar perturbações contra o Defensor da rodada anterior.

A taxa de evasão é medida **antes da adaptação**.

Em seguida, o Defensor recebe exemplos adversariais e é atualizado, produzindo:

```text
Dn
```

O mesmo Atacante é avaliado novamente contra o Defensor atualizado:

```text
An × Dn
```

Isso fornece a medida de evasão **pós-adaptação**.

### 4. Checkpoints históricos

Cada rodada preserva os modelos produzidos.

Ao final, a arquitetura pode comparar:

```text
A1..An × D0..Dn
```

A matriz resultante permite observar como diferentes defensores respondem a atacantes de rodadas distintas.

### 5. Resultados

O treinamento produz, entre outros artefatos:

```text
summary.json
attacker_metrics.csv
defender_metrics.csv
matriz_checkpoints.csv
gráficos
checkpoints
```

---

## Execução em rede

Após o treinamento, checkpoints podem ser utilizados no pipeline de rede.

```text
checkpoint selecionado
        ↓
Atacante gera variantes
        ↓
Defensor avalia as variantes
        ↓
somente evasões seguem adiante
        ↓
Translator
        ↓
┌───────────────────────────────┐
│ tradução válida?              │
│                               │
│ não → rejeita e registra      │
│ sim → envia ao Sender         │
└───────────────────────────────┘
        ↓
Sender
        ↓
dry-run ou tráfego real
        ↓
métricas + histórico da execução
```

### 1. Seleção de checkpoint

Os modos atuais permitem selecionar, entre outros cenários:

- checkpoint de demonstração;
- modelos finais da execução;
- pares específicos de Atacante e Defensor.

### 2. Geração e evasão

O Atacante gera novas variantes a partir das amostras DDoS disponíveis.

O Defensor avalia cada vetor.

Somente aqueles classificados incorretamente como benignos são considerados evasões e enviados ao Translator.

### 3. Tradução

O Translator tenta converter o vetor adversarial para `AttackParams`.

A arquitetura diferencia:

```text
evasão matemática
      ≠
tradução válida
```

Uma evasão pode ser rejeitada quando as features não resultam em parâmetros coerentes segundo as regras atuais de tradução.

### 4. Execução

Traduções válidas seguem para o Sender.

Em `dry-run`, nenhum pacote é transmitido.

Na execução real, o Sender produz tráfego dentro da rede Docker controlada.

A arquitetura também distingue:

```text
tradução válida
      ≠
tráfego efetivamente produzido
```

Os parâmetros solicitados e os valores observados pelo Sender são registrados separadamente.

### 5. Persistência

Cada execução possui seu próprio histórico:

```text
/models/experiments/<run_id>/network_runs/
```

Exemplos:

```text
dry_run_demo_<timestamp>.json
dry_run_final_<timestamp>.json
attack_demo_<timestamp>.json
```

Esses arquivos podem ser exportados para `results/`.

---

## Relação entre os dois fluxos

Atualmente:

```text
treinamento
     ↓
checkpoints
     ↓
execução em rede
```

A execução em rede utiliza modelos previamente treinados, mas o tráfego produzido ainda não retorna automaticamente ao ciclo de aprendizado.

O fechamento completo desse caminho exige captura, extração de features e feedback para o Defensor, componentes planejados para etapas futuras.
