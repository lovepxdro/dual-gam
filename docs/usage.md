# Como usar

Este documento descreve os principais comandos disponíveis para executar a ADArena.

> **Pré-requisito:** Docker com permissão de acesso ao socket, utilizando `root` ou um usuário pertencente ao grupo `docker`.

> **Segurança:** os modos que produzem pacotes reais devem ser utilizados somente em ambiente de laboratório controlado e contra alvos autorizados. A topologia padrão utiliza a rede Docker local do próprio projeto.

## 1. Preparar o dataset

Crie o diretório de dados:

```bash
mkdir -p data
```

Baixe o CIC-IDS2017 e coloque o arquivo:

```text
DDoS-Friday-no-metadata.parquet
```

dentro de:

```text
data/
```

Dataset utilizado:

```text
https://www.kaggle.com/datasets/dhoogla/cicids2017
```

## 2. Baixar o PyTorch CPU

A imagem de treinamento utiliza a versão CPU-only.

```bash
wget -c https://download.pytorch.org/whl/cpu/torch-2.3.1%2Bcpu-cp311-cp311-linux_x86_64.whl      -O docker/torch-2.3.1+cpu-cp311-cp311-linux_x86_64.whl
```

## 3. Preparar o script

```bash
chmod +x scripts/run.sh
```

## 4. Executar os testes

```bash
./scripts/run.sh test
```

A suíte cobre os principais componentes da implementação, incluindo:

- pré-processamento;
- Translator;
- Sender;
- Controller;
- integração do pipeline em modo seguro.

## 5. Treinar uma nova execução

```bash
./scripts/run.sh train ./data/DDoS-Friday-no-metadata.parquet
```

O fluxo padrão executa:

```text
preprocessing
    ↓
pré-treino do Defensor
    ↓
ciclo adversarial
    ↓
checkpoints
    ↓
matriz histórica
    ↓
métricas e gráficos
    ↓
avaliação final
```

Cada treinamento cria um experimento independente, identificado por um `run_id`, por exemplo:

```text
run_YYYYMMDD_HHMMSS_seed42
```

Os artefatos persistentes ficam no volume de modelos utilizado pelos containers.

## 6. Exportar os resultados

```bash
./scripts/run.sh results
```

Os resultados do experimento ativo são copiados para o diretório:

```text
results/
```

Entre os arquivos exportados podem estar:

```text
summary.json
attacker_metrics.csv
defender_metrics.csv
matriz_checkpoints.csv
gráficos
network_runs/
```

## 7. Executar o pipeline em dry-run

### Checkpoint de demonstração

```bash
./scripts/run.sh dry-run
```

Percorre o pipeline completo sem enviar pacotes reais.

### Modelos finais

```bash
./scripts/run.sh dry-run final
```

Utiliza os modelos finais da execução ativa.

Em ambos os casos, o pipeline registra:

```text
gerados
evasões
traduções válidas
traduções rejeitadas
execuções simuladas
```

## 8. Executar tráfego real

```bash
./scripts/run.sh attack
```

O comando:

1. inicia a topologia Docker necessária;
2. informa o alvo utilizado;
3. exibe um aviso;
4. solicita confirmação antes da execução.

A execução registra, entre outros valores:

- vetores gerados;
- evasões;
- traduções válidas;
- execuções encaminhadas ao Sender;
- PPS solicitado;
- PPS produzido;
- duração solicitada;
- duração observada;
- pacotes;
- throughput produzido.

Cada execução possui seu próprio arquivo de histórico em:

```text
/models/experiments/<run_id>/network_runs/
```

## 9. Exportar novamente após dry-run ou attack

Depois de novas execuções de rede:

```bash
./scripts/run.sh results
```

Isso atualiza a cópia em `results/` com os históricos mais recentes.

## 10. Monitorar o servidor alvo

Em outro terminal:

```bash
watch -n1 'curl -s http://localhost:8080/metrics | python3 -m json.tool'
```

## 11. Visualizar logs

```bash
./scripts/run.sh logs h-attack
```

## 12. Encerrar o ambiente

```bash
./scripts/run.sh down
```

## Observação sobre `/models`

O caminho:

```text
/models/
```

existe dentro dos containers e é respaldado por um volume Docker.

Por isso, ele não aparece necessariamente como um diretório comum na raiz do repositório.

Para localizar o volume no host:

```bash
docker volume ls
```

Depois:

```bash
docker volume inspect NOME_DO_VOLUME
```

O campo `Mountpoint` mostra onde os dados persistentes estão armazenados no host.

Para uso cotidiano, porém, o caminho recomendado para acessar os artefatos é simplesmente:

```bash
./scripts/run.sh results
```

e consultar o diretório local `results/`.
