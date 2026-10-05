# Evolução técnica da ADArena — v1.7 → v2.7

> **Documento histórico.** Este arquivo registra as principais decisões técnicas tomadas durante a evolução da ADArena entre a linha **v1.7** e a linha **v2.7**. Ele não substitui a documentação atual da plataforma.
>
> Para o estado atual da implementação, consulte:
>
> - `../architecture.md` — arquitetura e responsabilidades dos componentes;
> - `../flow.md` — fluxos de execução;
> - `../usage.md` — uso da CLI, TUI e configurações.
>
> O objetivo aqui é registrar **por que a ferramenta mudou, como eu queria que ela funcionasse e quais decisões foram usadas para chegar ao desenho atual**.

---

## 1. Ponto de partida: a v1.7

A v1.7 foi a primeira versão em que a ADArena já conseguia executar um ciclo experimental relevante de aprendizado adversarial.

Naquela linha, o projeto já possuía:

- um Attacker capaz de gerar perturbações em amostras de DDoS;
- um Defender binário;
- pré-treino do Defender;
- rodadas competitivas Attacker × Defender;
- checkpoints históricos;
- avaliação antes e depois da adaptação;
- matriz cruzada entre checkpoints;
- persistência de métricas e gráficos;
- um `Translator` para converter vetores adversariais em parâmetros de rede;
- um `Sender` para representar ou executar esses parâmetros.

A v1.7 foi importante porque mostrou que o experimento tinha valor, mas também expôs limitações arquiteturais.

O fluxo era fortemente centrado no experimento específico:

```text
Dataset
   ↓
Attacker ↔ Defender
   ↓
checkpoints
   ↓
Translator
   ↓
Sender
```

Esse desenho permitia testar o caso de uso existente, mas ainda havia bastante conhecimento concreto espalhado pela aplicação: nomes de modelos, diretórios, formato de dataset, lógica de treinamento e detalhes de rede apareciam próximos demais uns dos outros.

Também existia uma lacuna importante no caminho de rede.

A ferramenta conseguia partir do vetor adversarial em direção à rede:

```text
vetor adversarial
      ↓
Translator
      ↓
Sender
      ↓
rede
```

mas ainda não existia um caminho genérico de retorno:

```text
rede
 ↓
captura
 ↓
extração de features
 ↓
vetor reconstruído
 ↓
Defender
```

Além disso, os experimentos da v1.7 deixaram claro que três conceitos diferentes não podiam ser tratados como equivalentes:

```text
evasão matemática
        ≠
tradução válida
        ≠
comportamento efetivamente observado na rede
```

Essa constatação foi uma das principais motivações para a linha v2.

---

## 2. Como eu queria que a ferramenta funcionasse

A partir da v1.7, a intenção deixou de ser apenas melhorar o código do experimento existente.

A meta passou a ser transformar a ADArena em uma **plataforma experimental reutilizável**.

Eu queria que fosse possível montar um experimento mais próximo deste processo:

```text
escolher modo
      ↓
escolher Dataset
      ↓
escolher Attacker
      ↓
escolher Defender
      ↓
escolher Protocol
      ↓
configurar parâmetros
      ↓
validar
      ↓
executar
      ↓
inspecionar resultados
```

sem precisar alterar o Core sempre que um novo modelo, dataset ou protocolo fosse adicionado.

Alguns objetivos passaram a orientar as mudanças:

- separar o mecanismo da plataforma da lógica científica de um experimento;
- permitir substituição de componentes;
- representar explicitamente a compatibilidade entre componentes;
- separar treino, simulação e observação;
- preservar configurações e artefatos de cada execução;
- permitir reprodução por configuração;
- ter uma CLI apropriada para automação;
- ter uma TUI apropriada para uso interativo;
- reduzir dependência de caminhos internos conhecidos pelo usuário;
- manter o caminho de rede seguro e controlado durante o desenvolvimento;
- não obrigar a arquitetura a usar aprendizado online ou SDN real para ser considerada funcional.

Em resumo, a mudança principal foi:

```text
antes:
"executar este experimento"

depois:
"descrever um experimento e deixar a ADArena executá-lo"
```

---

## 3. Separar plataforma e experimento

Uma das primeiras decisões da linha v2 foi tornar explícita a diferença entre:

```text
ADArena
   =
infraestrutura experimental
```

e:

```text
Attacker + Defender + Dataset + protocolo
   =
um experimento concreto
```

Na linha anterior, essas duas responsabilidades ainda estavam muito próximas.

A linha v2 introduziu um Core que conhece contratos e configurações, mas tenta não conhecer detalhes internos dos modelos.

Isso resultou na estrutura conceitual:

```text
ExperimentConfig
      ↓
ExperimentRunner
      ↓
ExperimentProtocol
      ↓
ComponentRegistry
      ↓
componentes concretos
```

O objetivo dessa mudança não foi tornar qualquer experimento automaticamente compatível com a ADArena.

O objetivo foi permitir que uma nova implementação pudesse ser integrada **como componente**, sem obrigar a reescrever a infraestrutura comum.

---

## 4. Evolução da linha v2

A evolução foi feita de forma incremental.

De maneira resumida:

| Versão | Principal mudança |
| --- | --- |
| v2.0 | Core, contratos, Registry e representações |
| v2.1 | Renderer e NetworkBackend |
| v2.2 | Capture, Extractor e caminho rede → modelo |
| v2.3 | DecisionPolicy, RuleEnforcer e controle |
| v2.4 | OBSERVE, ExperimentRunner e fechamento do caminho arquitetural |
| v2.5 | CLI, TOML, preflight, runs e logging |
| v2.6 | TUI e execução interativa |
| v2.7 | Builder, seleção de recursos, persistência de TOML e polimento da interface |

Essa divisão foi útil porque permitiu validar cada etapa antes de aumentar novamente o escopo.

---

## 5. Componentes, Registry e contratos

### Problema

Na implementação anterior, partes do sistema ainda dependiam diretamente de classes concretas ou de convenções específicas do experimento.

Isso dificultava responder perguntas simples como:

- quais Attackers estão disponíveis?;
- qual Dataset pode ser usado?;
- este Renderer aceita a representação produzida pelo Attacker?;
- posso trocar o Defender sem alterar o Runner?

### Decisão

Foi introduzido o `ComponentRegistry`.

Cada componente passa a ser identificado por um `component_id` e classificado por um tipo, como:

```text
ATTACKER
DEFENDER
DATASET
EXPERIMENT_PROTOCOL
RENDERER
NETWORK_BACKEND
CAPTURE
EXTRACTOR
```

O Registry centraliza a descoberta e criação das implementações.

Com isso, em vez de a interface ou o Runner conhecerem classes concretas, eles trabalham com identificadores registrados.

### Representações

Outro ponto importante foi declarar as representações de entrada e saída dos componentes.

Entre as representações usadas atualmente estão conceitos como:

```text
FLOW_FEATURES
PACKET_SEQUENCE
PACKET_RECORDS
RAW_PACKETS
ATTACK_PARAMS
BINARY_CLASSIFICATION
NETWORK_RESULT
```

A intenção é evitar uma composição aparentemente válida que só falharia muito depois durante a execução.

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

O Core pode verificar esse tipo de compatibilidade sem precisar conhecer a arquitetura interna do modelo.

### Limite dessa decisão

A modularidade não significa suporte automático a qualquer domínio.

Um novo dataset, modelo ou ataque ainda precisa implementar os contratos e trabalhar com uma representação compatível.

A diferença é que essa extensão deve acontecer principalmente pela adição de componentes, e não pela reescrita da plataforma.

---

## 6. Configuração declarativa

### Problema

Na linha anterior, parte do contexto do experimento dependia de argumentos, scripts, defaults ou caminhos definidos diretamente no fluxo de execução.

Isso tornava a reprodução menos clara.

### Decisão

A execução passou a ser representada por um `ExperimentConfig`.

A configuração descreve, entre outros elementos:

- modo;
- seed;
- device;
- Dataset;
- Attacker;
- Defender;
- Protocol;
- parâmetros de treinamento;
- componentes de rede;
- checkpoints;
- preprocessador;
- threshold;
- limites e duração de captura;
- parâmetros específicos dos componentes.

O formato TOML foi adotado como representação persistente dessa configuração.

O fluxo passou a ser:

```text
TOML
 ↓
Config Loader
 ↓
ExperimentConfig
 ↓
validação
 ↓
execução
```

A TUI também produz o mesmo `ExperimentConfig`, então não existe uma segunda implementação do experimento específica para a interface.

---

## 7. Validação e preflight

A configuração declarativa permitiu separar dois tipos de erro.

### Validação semântica

A primeira etapa verifica se a configuração faz sentido internamente.

Por exemplo:

- componente registrado;
- categoria correta;
- representação compatível;
- campos válidos;
- regras específicas do modo.

### Preflight

A segunda etapa verifica recursos externos antes de iniciar a execução.

Exemplos:

```text
arquivo de dataset existe?
checkpoint existe?
preprocessador existe?
interface de captura existe?
```

Essa separação reduz a chance de criar um run e só depois descobrir que um recurso básico estava ausente.

---

## 8. ExperimentRunner e protocolos

### Problema

O fluxo anterior concentrava muita lógica científica e operacional em controladores específicos.

Isso tornava difícil distinguir:

```text
"o que toda execução precisa fazer"
```

de:

```text
"o que este experimento específico precisa fazer"
```

### Decisão

O `ExperimentRunner` passou a ser o orquestrador genérico.

Ele é responsável por tarefas como:

- validar a configuração;
- criar o diretório do run;
- preparar os recursos necessários;
- construir o contexto;
- resolver o protocolo;
- executar o protocolo;
- persistir contexto;
- retornar um resultado;
- atualizar a referência da execução mais recente.

A lógica científica fica dentro de `ExperimentProtocol`.

Na linha atual existem protocolos separados para:

```text
treinamento adversarial
simulação de rede
observação de rede
```

Assim, o Runner não precisa saber como funciona internamente o ciclo Attacker × Defender.

Da mesma forma, ele não precisa conhecer os detalhes do protocolo de observação.

Essa foi uma das mudanças mais importantes para que a ADArena deixasse de ser um único experimento encapsulado em uma aplicação.

---

## 9. Separação entre TRAIN, SIMULATE e OBSERVE

Na linha anterior, treino, geração adversarial e rede ainda eram vistos com frequência como partes de um fluxo principal.

Na linha v2, optei por separar três modos públicos.

### TRAIN

Responsável pelo experimento de treinamento.

```text
Dataset
   ↓
Preprocessor
   ↓
Attacker ↔ Defender
   ↓
checkpoints
   ↓
métricas e artefatos
```

Ele não precisa da rede física.

### SIMULATE

Responsável por validar o caminho:

```text
feature-space
     ↓
Attacker
     ↓
Defender
     ↓
Renderer
     ↓
NetworkBackend
```

A interface pública atual mantém esse modo exclusivamente em `dry-run`.

Assim é possível testar:

- evasão matemática;
- tradução das features;
- validade da representação;
- integração com o backend;

sem transformar a validação de software em geração real de tráfego.

### OBSERVE

Responsável pelo caminho inverso:

```text
tráfego existente
      ↓
Capture
      ↓
Extractor
      ↓
Preprocessor
      ↓
Defender
      ↓
DecisionPolicy
      ↓
RuleEnforcer
```

Esse modo não precisa de Dataset nem Attacker.

A separação foi deliberada.

Não existe necessidade arquitetural de forçar um grande pipeline em que todas as etapas aconteçam sempre juntas.

---

## 10. Do Translator/Sender para Renderer/NetworkBackend

A v1.7 já possuía `Translator` e `Sender`.

A linha v2 preservou a ideia, mas a reorganizou em contratos mais genéricos.

### Renderer

O `Renderer` recebe a representação produzida no espaço do modelo e tenta transformá-la em uma representação utilizável pelo caminho de rede.

A mudança de nome também ajuda a deixar claro que:

```text
evasão no modelo
```

não significa automaticamente:

```text
representação válida para rede
```

### NetworkBackend

O `NetworkBackend` recebe a representação produzida pelo Renderer.

A separação permite trocar a lógica de tradução sem acoplar o modelo à implementação do backend.

Na versão atual, o caminho público permanece em `dry-run`.

Isso foi mantido propositalmente durante a Frente 1 porque o objetivo dessa etapa era validar a arquitetura e a integração entre as camadas.

---

## 11. O caminho inverso: Capture e Extractor

Uma das principais lacunas da v1.7 era a ausência do retorno rede → modelo.

Para fechar essa lacuna foram criados dois contratos.

### Capture

Responsável por observar tráfego já existente.

Ele não deve decidir como as features serão construídas.

### Extractor

Responsável por reconstruir a representação esperada pelo modelo.

No caso atual:

```text
pacotes
   ↓
fluxos
   ↓
FLOW_FEATURES
```

O `BasicFlowExtractor` consegue reconstruir estruturalmente o schema esperado pela linha atual do CIC-IDS2017.

Isso não deve ser confundido com validação de equivalência numérica perfeita com ferramentas externas de extração.

Essa distinção foi mantida explicitamente para evitar tratar compatibilidade estrutural como prova de fidelidade científica.

---

## 12. Separar detecção, decisão e enforcement

Outra decisão foi impedir que o Defender se tornasse responsável diretamente pela reação da plataforma.

O fluxo passou a ser:

```text
Defender
   ↓
score / classificação
   ↓
DecisionPolicy
   ↓
ação
   ↓
RuleEnforcer
```

### DecisionPolicy

Interpreta o resultado do modelo e decide qual ação experimental deve ser tomada.

### RuleEnforcer

É responsável por materializar essa decisão.

Na implementação atual, o enforcer é deliberadamente `dry-run`.

Isso significa que a arquitetura consegue representar:

```text
detecção
   ↓
decisão
   ↓
enforcement
```

sem exigir que o fechamento da Frente 1 dependesse de uma integração real com OVS, OS-Ken ou outro controlador SDN.

Essa decisão também evita misturar duas perguntas diferentes:

```text
o modelo detectou?
```

e:

```text
a infraestrutura conseguiu aplicar uma ação?
```

---

## 13. Reprodutibilidade e rastreabilidade

A evolução da plataforma também buscou reduzir estado implícito.

Cada execução recebe um `run_id` próprio e possui seu diretório de artefatos.

Dependendo do protocolo, um run pode conter:

```text
checkpoints/
metrics/
plots/
logs/
preprocessador/
config_execucao.json
historico_treino.json
matriz_checkpoints.csv
matriz_checkpoints.json
summary.json
```

O `config_execucao.json` funciona como snapshot da execução efetiva.

Essa distinção é importante:

```text
TOML
=
configuração solicitada

config_execucao.json
=
contexto efetivamente registrado durante o run
```

O preprocessador também é persistido porque ele faz parte do estado necessário para reutilizar o modelo corretamente.

Isso evita carregar um checkpoint com um pipeline de preparação diferente daquele utilizado no treinamento.

---

## 14. CLI como interface reproduzível

Outra etapa foi substituir a dependência operacional de scripts específicos por uma interface principal.

A CLI passou a expor operações como:

```text
version
components
train
simulate
observe
config validate
runs
runs show
```

A ideia da CLI é atender principalmente:

- reprodução de experimentos;
- automação;
- execução explícita por configuração;
- inspeção de componentes;
- inspeção de runs.

Ela usa a mesma camada de aplicação utilizada pela TUI.

---

## 15. TUI como interface interativa

Depois da CLI, foi adicionada uma TUI.

O objetivo não foi criar outra implementação da ADArena.

A TUI é apenas outra forma de construir e executar o mesmo `ExperimentConfig`.

A interface foi dividida em:

```text
Dashboard
Executar
Runs
Componentes
```

Na área de execução existem dois caminhos:

```text
Novo experimento
Carregar TOML
```

O Builder permite selecionar componentes, parâmetros e modo sem escrever o TOML diretamente.

A configuração montada pela TUI também pode ser salva para reprodução posterior.

---

## 16. Seleção de recursos pela TUI

A primeira versão do Builder ainda exigia que o usuário conhecesse caminhos internos.

Por exemplo, para reutilizar um treinamento era necessário informar manualmente algo semelhante a:

```text
models/.../experiments/run_.../checkpoints/...
```

Isso funcionava, mas não era uma boa experiência de uso.

A decisão final da v2.7 foi fazer a interface descobrir os recursos disponíveis.

### TRAIN

A TUI lista os arquivos disponíveis em:

```text
data/
```

e permite escolher o dataset.

### SIMULATE e OBSERVE

A interface lista runs de treinamento.

Depois que um run é selecionado, a TUI encontra automaticamente:

```text
checkpoints do Attacker
checkpoints do Defender
preprocessador
```

O usuário passa a escolher semanticamente:

```text
Run de origem
      ↓
Attacker checkpoint
Defender checkpoint
```

em vez de copiar caminhos manualmente.

O preprocessador é resolvido automaticamente.

Essa alteração foi restrita à camada de apresentação.

O `ExperimentConfig` resultante continua contendo os caminhos concretos necessários à execução.

Portanto:

```text
experiência do usuário mudou
        ↓
sem alterar
        ↓
semântica do experimento
```

---

## 17. Observabilidade dos resultados

Durante a validação final da ferramenta surgiu outro problema de usabilidade.

Os protocolos já produziam métricas, históricos, matrizes, gráficos e artefatos, mas a saída da interface escondia parte da informação mais útil.

A v2.7 passou a apresentar resumos específicos por modo.

### TRAIN

Em vez de mostrar apenas as métricas finais do classificador, a TUI pode apresentar a evolução adversarial:

```text
A1 × D0 → A1 × D1
A2 × D1 → A2 × D2
...
```

Isso torna visível o comportamento que motivou o protocolo:

```text
Attacker encontra evasão
        ↓
Defender se adapta
        ↓
evasão é medida novamente
```

Também são destacados artefatos como histórico, matriz de checkpoints e plots.

### SIMULATE

O resultado é apresentado como um funil:

```text
amostras selecionadas
      ↓
evasões matemáticas
      ↓
traduções válidas
      ↓
execuções dry-run
```

Quando possível, a interface também mostra o confronto selecionado, por exemplo:

```text
A1 × D0
```

Essa visualização reforça uma decisão científica importante da linha anterior:

```text
evasão matemática
        ≠
tradução válida
```

### OBSERVE

O resumo destaca:

```text
pacotes capturados
fluxos reconstruídos
benignos / ataques
decisões BLOCK
regras aplicadas
```

A interface possui níveis `Normal` e `Detalhada`, mas o arquivo técnico do run continua preservando os logs completos.

---

## 18. Inspeção de runs

A aba `Runs` também foi ampliada.

Além dos metadados básicos, a TUI aponta para artefatos importantes quando presentes:

```text
config_execucao.json
summary.json
historico_treino.json
matriz_checkpoints.csv
logs/run.log
metrics/network_observation.json
plots/
```

A TUI não altera nem recalcula esses resultados.

Ela apenas facilita a localização dos arquivos persistidos.

---

## 19. Estratégia de validação

A Frente 1 foi validada em diferentes níveis.

### Testes automatizados

A suíte cobre contratos e integração entre componentes, incluindo:

- Registry;
- configuração;
- Runner;
- protocolos;
- Builder;
- TOML;
- seleção de recursos;
- captura e extração;
- inferência;
- controle;
- TUI e CLI.

### Smoke tests

Foram usados experimentos pequenos para verificar integração sem tratar esses valores como evidência científica.

### TRAIN completo

Foi executado um ciclo completo de treinamento com checkpoints, histórico, avaliação cruzada, métricas e gráficos.

O comportamento adversarial observado na linha anterior continuou aparecendo na arquitetura nova, indicando que a refatoração não eliminou a dinâmica experimental que o projeto buscava estudar.

### SIMULATE

O caminho de simulação foi validado usando checkpoints históricos em `dry-run`.

Essa etapa confirmou a separação entre:

```text
evasão
tradução válida
backend
```

### OBSERVE

O caminho de observação foi validado fisicamente sobre tráfego benigno existente no laboratório Docker:

```text
rede Docker
   ↓
Capture
   ↓
Extractor
   ↓
Preprocessor
   ↓
Defender
   ↓
DecisionPolicy
   ↓
DryRunRuleEnforcer
```

O objetivo dessa validação foi provar o caminho arquitetural, não produzir um resultado científico sobre ataques.

---

## 20. Decisões deliberadamente deixadas de fora

Algumas possibilidades foram consideradas, mas não foram tratadas como requisitos para fechar a Frente 1.

### Enforcement real

A arquitetura possui `DecisionPolicy` e `RuleEnforcer`, mas a implementação atual permanece em `dry-run`.

Integração real com SDN pode ser estudada posteriormente.

### OVS / OS-Ken

A arquitetura de referência considerava uma camada de controle de rede, mas a validação atual não depende de OVS ou OS-Ken.

A rede física utilizada no laboratório atual é uma bridge Docker.

### Aprendizado online

O caminho de observação produz evidência que pode ser consumida por experimentos futuros, mas o Core não obriga o Defender a se retreinar continuamente em produção.

### Suporte automático a qualquer ataque

A arquitetura é extensível, não universal.

Novos domínios ainda precisam de componentes e representações apropriados.

### Honeypot

A inclusão de honeypots/deception continua sendo uma possibilidade de evolução, mas não foi adicionada apenas para completar a arquitetura.

A intenção é tratá-la como uma variável experimental quando houver uma pergunta concreta sobre como essa fonte de informação ou mecanismo de engano afeta Attacker, Defender ou o protocolo.

---

## 21. Resumo das principais mudanças

A transição pode ser resumida assim:

| v1.7 | v2.7 |
| --- | --- |
| experimento fortemente centrado em implementações concretas | plataforma centrada em contratos |
| Controller com múltiplas responsabilidades | Runner genérico + Protocols |
| componentes conhecidos diretamente pelo fluxo | `ComponentRegistry` |
| compatibilidade em grande parte implícita | representações de entrada/saída declaradas |
| configuração mais dependente de scripts e convenções | `ExperimentConfig` + TOML |
| Translator | Renderer |
| Sender | NetworkBackend |
| caminho feature-space → rede | caminhos feature-space → rede e rede → feature-space |
| sem abstração formal de captura | Capture |
| sem abstração formal de extração | Extractor |
| decisão defensiva pouco separada | DecisionPolicy |
| resposta pouco separada | RuleEnforcer |
| treino e rede mais acoplados | TRAIN / SIMULATE / OBSERVE |
| scripts como principal interface operacional | CLI |
| configuração manual | TUI Builder |
| caminhos informados manualmente | descoberta de datasets, runs e checkpoints |
| saída principalmente técnica | resumos contextuais por modo |
| artefatos existentes, mas menos centralizados | runs rastreáveis e inspecionáveis |

---

## 22. Estado ao final da v2.7

Ao final dessa etapa, considero a Frente 1 concluída no sentido em que ela foi definida: **construir a plataforma experimental necessária para as próximas fases da pesquisa**.

A ADArena atual fornece:

```text
Core modular
Registry
ExperimentConfig
ExperimentRunner
Protocols

Dataset
Attacker
Defender

Renderer
NetworkBackend

Capture
Extractor

DecisionPolicy
RuleEnforcer

TRAIN
SIMULATE
OBSERVE

CLI
TUI
TUI Builder

TOML
runs
logging
artefatos
preflight
testes
```

A arquitetura ainda pode evoluir, mas novas mudanças devem surgir principalmente de necessidades encontradas durante o estudo dos modelos e dos experimentos, e não da tentativa de antecipar todos os cenários possíveis.

Esse ponto é importante para a continuidade do projeto.

A plataforma passa a funcionar como infraestrutura para as próximas perguntas:

```text
Frente 2
modelos
   ↓
como Attacker e Defender devem evoluir?

Frente 3
experimentos
   ↓
como medir e comparar essa evolução?
```

A partir daqui, mudanças arquiteturais devem ser justificadas por necessidades concretas dessas frentes.

---

## 23. Relação com os outros documentos

Este arquivo registra **decisões e evolução**.

Para evitar duplicação, os detalhes atuais permanecem separados:

```text
docs/
├── architecture.md
│   └── como a ADArena está organizada hoje
│
├── flow.md
│   └── como cada fluxo executa
│
├── usage.md
│   └── como utilizar a ferramenta
│
└── history/
    ├── resultados_v1_7.md
    │   └── resultados e limitações históricas da v1.7
    │
    └── evolucao_tecnica_v1_7_v2_7.md
        └── decisões que levaram da v1.7 à v2.7
```

Documentos em `history/` devem ser interpretados dentro do contexto da versão que registram e não substituem a documentação viva da implementação atual.
