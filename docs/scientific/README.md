# Análise científica da ADArena

Este diretório concentra a fase da pesquisa que une **experimentação científica** e **evolução dos modelos**.

A partir da v2.7, a plataforma é considerada suficientemente estável para servir como infraestrutura experimental. O foco deixa de ser expandir a ferramenta e passa a ser responder perguntas científicas, coletar métricas, analisar resultados e modificar o Attacker ou o Defender apenas quando os experimentos indicarem que isso é necessário.

> A arquitetura, os fluxos e o uso atual da plataforma continuam documentados em `../architecture.md`, `../flow.md` e `../usage.md`. Este diretório registra o planejamento, a execução e a análise dos experimentos científicos.

---

## 1. Como esta fase vai funcionar

O trabalho será iterativo:

```text
pergunta científica
      ↓
hipótese
      ↓
experimento
      ↓
métricas
      ↓
análise
      ↓
limitação identificada?
   ┌──────────────┴──────────────┐
  não                           sim
   ↓                             ↓
próxima pergunta          modificar o modelo
                                ↓
                         novo experimento
```

A intenção não é implementar primeiro todos os modelos planejados e só depois avaliá-los. A evolução dos modelos deve acontecer em resposta aos resultados obtidos.

Cada experimento terá um arquivo próprio, por exemplo:

```text
docs/scientific/
├── README.md
├── e0.md
├── e1.md
├── e2.md
└── ...
```

O mesmo arquivo será atualizado durante toda a vida do experimento. Ele começa com a pergunta, hipótese e método; depois recebe as execuções, métricas, análise e conclusão.

---

## 2. Estado inicial dos modelos

A linha atual utiliza:

```text
Attacker
└── gerador neural de perturbações adversariais

Defender
└── classificador binário
```

O sistema possui um **jogo adversarial** entre os dois componentes, mas isso não significa que existam atualmente duas GANs completas.

O Attacker atual trabalha principalmente perturbando amostras existentes. O Defender atual é discriminativo.

Uma direção futura é evoluir os componentes para abordagens mais generativas, mas essas mudanças devem ser introduzidas separadamente para permitir comparação.

Uma sequência possível é:

```text
M0  Attacker atual       + Defender atual
 ↓
M1  Attacker generativo  + Defender atual
 ↓
M2  Attacker atual       + Defender generativo
 ↓
M3  Attacker generativo  + Defender generativo
```

Essa ordem não é obrigatória. Os resultados devem decidir qual mudança é prioritária.

---

## 3. Métricas de interesse

As métricas serão organizadas em cinco grupos.

### 3.1 Desempenho convencional do Defender

```text
Accuracy
Precision
Recall
F1
MCC
ROC-AUC
FPR
FNR
```

Essas métricas mostram o desempenho convencional do detector, mas não medem sozinhas robustez adversarial.

### 3.2 Robustez adversarial

```text
taxa de evasão pré-adaptação
taxa de evasão pós-adaptação
robust detection rate = 1 - taxa de evasão
ganho de adaptação
```

O ganho de adaptação será tratado inicialmente como:

```text
ganho_adaptacao
=
evasao_pre - evasao_pos
```

### 3.3 Generalização

Exemplos de avaliações:

```text
Attacker histórico × Defender histórico
Attacker de outra seed × Defender
Attacker não observado durante adaptação × Defender
nova família de ataque × Defender
novo dataset × Defender
novo mecanismo gerador × Defender
```

### 3.4 Viabilidade das variantes

Quando houver tradução para uma representação de rede:

```text
magnitude da perturbação
features alteradas
taxa de traduções válidas
taxa de traduções válidas dado que houve evasão
```

É importante preservar a distinção:

```text
evasão matemática
        ≠
tradução válida
        ≠
comportamento observado na rede
```

### 3.5 Estabilidade

Em experimentos com repetições:

```text
média
desvio padrão
mínimo
máximo
variação entre seeds
```

---

## 4. Primeira leva de experimentos

A primeira leva caracteriza o comportamento atual antes de mudanças importantes nos modelos.

| ID | Pergunta | Principal variação | Objetivo |
| --- | --- | --- | --- |
| **E0** | O comportamento atual é estável entre execuções? | seed | estabelecer baseline multi-seed |
| **E1** | O Defender realmente se adapta ao Attacker? | análise pré/pós-adaptação | quantificar adaptação |
| **E2** | A robustez permanece contra Attackers não observados diretamente? | checkpoints/Attackers separados | avaliar generalização adversarial |
| **E3** | A robustez transfere entre treinamentos independentes? | seeds diferentes | avaliar generalização cross-seed |
| **E4** | O Defender generaliza para outra distribuição? | dataset/família de ataque | avaliar distribution shift |
| **E5** | Um Attacker mais generativo melhora diversidade ou transferência? | novo Attacker | evoluir o modelo atacante |
| **E6** | Uma defesa generativa melhora robustez contra variantes não vistas? | novo Defender | evoluir o modelo defensivo |
| **E7** | Como os modelos evoluídos interagem entre si? | Attacker + Defender novos | estudar competição avançada |
| **E8** | Informação de deception/honeypot altera adaptação ou generalização? | nova fonte de informação | estudar deception |

A tabela é um planejamento inicial. Os resultados podem mudar a ordem, dividir experimentos ou criar novos IDs.

---

## 5. E0 — baseline multi-seed

O primeiro experimento não modifica o modelo.

A pergunta é:

> O comportamento observado com a configuração atual se mantém quando o experimento é repetido com inicializações diferentes?

O padrão observado até agora é:

```text
Attacker encontra evasão
        ↓
Defender é exposto às variantes
        ↓
evasão diminui após adaptação
```

A primeira coleta utilizará, inicialmente:

```text
seed 42
seed 43
seed 44
```

mantendo os demais parâmetros constantes.

O detalhamento e os resultados ficam em `e0.md`.

---

## 6. O que significa "ataque inédito"

A expressão é ambígua. A pesquisa deve sempre explicitar qual definição está sendo usada.

Uma hierarquia inicial é:

```text
Nível 1  amostra nunca vista
Nível 2  checkpoint de Attacker não visto
Nível 3  Attacker treinado com outra seed
Nível 4  nova variante/família de ataque
Nível 5  novo dataset ou nova distribuição
Nível 6  variante produzida por outro mecanismo gerador
```

Assim, conclusões devem ser específicas.

Evitar:

```text
"o Defender detecta ataques inéditos"
```

Preferir:

```text
"o Defender generalizou para Attackers treinados
independentemente com outras seeds"
```

ou:

```text
"o Defender manteve desempenho em uma família de ataque
não utilizada na adaptação"
```

---

## 7. Evolução do Attacker

A evolução do Attacker deve responder a uma pergunta científica, e não apenas ao objetivo de usar uma GAN.

Uma hipótese futura possível é:

> Um Attacker generativo produz variantes adversariais mais diversas e com maior capacidade de transferência para Defenders não observados do que o perturbador atual?

A comparação deverá ser controlada:

```text
Attacker atual
     vs
Attacker generativo
```

mantendo o restante do experimento constante.

---

## 8. Evolução do Defender

O Defender atual é um classificador binário.

Uma direção futura é estudar se modelagem generativa melhora a generalização adversarial.

Exemplo de pergunta:

> Uma defesa baseada em modelagem generativa apresenta maior robustez contra variantes não vistas do que o classificador discriminativo atual?

Novamente, a comparação deve isolar a mudança:

```text
Defender atual
     vs
Defender generativo
```

antes de substituir simultaneamente Attacker e Defender.

---

## 9. Honeypot e deception

Honeypot/deception permanece como hipótese de pesquisa, não como funcionalidade obrigatória da arquitetura.

Uma pergunta possível é:

> Informação proveniente de deception melhora a capacidade do Defender de se adaptar a variantes futuras?

Isso pode levar a uma comparação:

```text
Defender sem informação de deception
                vs
Defender com informação de deception
```

Uma etapa posterior pode investigar como feedback de deception afeta o próprio Attacker.

Mudanças na infraestrutura só devem acontecer se forem necessárias para responder a uma pergunta experimental concreta.

---

## 10. Regra para mudanças nos modelos

A partir desta fase, uma alteração relevante deve seguir:

```text
resultado observado
      ↓
limitação identificada
      ↓
hipótese de melhoria
      ↓
modificação
      ↓
comparação controlada
```

Isso evita alterar várias partes ao mesmo tempo e perder a capacidade de explicar a causa de uma melhora ou piora.

---

## 11. Estrutura dos arquivos experimentais

Cada `eX.md` deve conter:

```text
# Ex — Nome

## Estado
## Pergunta científica
## Hipótese
## Objetivo
## Variáveis
## Configuração
## Procedimento
## Métricas
## Execuções
## Resultados
## Análise
## Limitações
## Conclusão
## Próximos passos
```

O documento deve separar claramente:

```text
resultado observado
        ≠
interpretação
        ≠
decisão de projeto
```

---

## 12. Relação com a plataforma

A v2.7 é o baseline de infraestrutura desta fase.

Regra geral:

```text
não alterar a arquitetura
        ↓
a menos que
        ↓
um experimento revele uma limitação real
que impeça a investigação
```

Correções de bugs continuam possíveis, mas o trabalho principal agora é:

```text
coletar
medir
comparar
interpretar
modificar modelos
repetir
```

---

## 13. Objetivo da primeira leva

A primeira leva deve responder progressivamente:

```text
o comportamento atual é estável?
        ↓
o Defender realmente se adapta?
        ↓
essa adaptação generaliza?
        ↓
onde o sistema falha?
        ↓
qual modificação de modelo faz sentido testar?
```

Os experimentos seguintes serão definidos e priorizados a partir dessas respostas.
