# E5-P — Registro da implementação

**Estado:** implementação e testes unitários concluídos em ambiente de verificação; execução no ambiente de trabalho e validação dos runs locais ainda pendentes.

## Decisões

- Código original `gan.models.Atacante` e Registry preservados. Novo módulo `gan.e5_conditional.AtacanteCondicional` usado somente pelos novos scripts E5.
- Backbone do A1 é um `Atacante` original estritamente carregado do A20 histórico.
- Ramo residual condicionado a `concat(x,z)` com saída inicial zero. Formulação: `A1(x,z) = b(z) + (1 - |b(z)|) * tanh(R(x,z))`.
- `delta` permanece por construção em `[-1,1]`; com `epsilon=0.3`, alteração absoluta normalizada de cada feature <=0,3.
- Enquanto o ramo residual está zerado, A1 reproduz **exatamente** A0, sem dependência efetiva de x. O efeito do condicionamento só deve ser atribuído após treinamento.
- A1 possui 104.538 parâmetros (40.525 a mais que A0 de 64.013): futura análise precisa de ablação de capacidade para separar impacto da entrada condicionada e do aumento de parâmetros.
- Checkpoints D20 congelados; nenhuma avaliação de evasão é feita nesta etapa.

## Verificações realizadas na preparação

- 10 testes unitários aprovados.
- Teste de compatibilidade com checkpoints reais E0 seeds 20 e 42 em ambiente isolado: carga `strict=True`, identidade de saída `erro_max = 0`, ausência de não-finitos, respeito a epsilon e retropropagação sem gradientes no D20.
- Documentação `audit_info.pkl` nas duas seeds registra split 154.884/22.127/44.253 e zero interseções. Isso apenas verifica o registro persistido, não reconstrói o split.

## Condição para avançar ao E5-A

Executar `e5_preflight.py` no próprio repositório e analisar `models/e5/preflight/e5_preflight.json`; não iniciar treino se incompatibilidade ocorrer.

## Limitações

- O protocolo E5-P não mede sucesso de evasão, diversidade ou transferência.
- A preservação do limite numérico no espaço normalizado não garante plausibilidade física ou semântica de fluxos de rede.
- A implementação A1 ainda não está registrada como componente selecionável no Registry, de propósito: o E5-A usará seu próprio executor experimental até validar o novo modelo.
