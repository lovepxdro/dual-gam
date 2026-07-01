#!/usr/bin/env bash
# Dual-GAM — Script principal de execução
#
# Uso:
#   ./scripts/run.sh train   [/caminho/para/dataset.parquet]
#   ./scripts/run.sh attack
#   ./scripts/run.sh dry-run
#   ./scripts/run.sh up      (sobe a rede sem atacar)
#   ./scripts/run.sh down

set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${2:-./data/DDoS-Friday-no-metadata.parquet}"

case "${1:-help}" in

  train)
    echo "=== Fase 1: Subindo rede de fundo ==="
    docker compose up -d h-target h1 h2 h3 h4

    echo ""
    echo "=== Fase 2: Copiando dataset para o volume ==="
    docker compose --profile train build h-attack-train
    docker compose run --rm \
      -v "$(realpath "$DATASET"):/tmp/dataset.parquet:ro" \
      --entrypoint cp \
      h-attack-train /tmp/dataset.parquet /data/DDoS-Friday-no-metadata.parquet

    echo ""
    echo "=== Fase 3: Treinando GAN ==="
    docker compose --profile train run --rm h-attack-train \
      train --data /data/DDoS-Friday-no-metadata.parquet

    echo ""
    echo "✓ Treino concluído. Modelos salvos em volume 'models'."
    echo "  Execute './scripts/run.sh attack' para iniciar os ataques."
    ;;

  attack)
    echo "=== Executando ataques reais ==="
    echo "ATENÇÃO: Apenas use em ambiente de laboratório isolado!"
    read -p "Confirmar? (s/N): " confirm
    [[ "$confirm" == "s" || "$confirm" == "S" ]] || exit 0

    docker compose up -d h-target h1 h2 h3 h4
    docker compose --profile attack up h-attack
    ;;

  dry-run)
    echo "=== Dry-run (sem pacotes reais) ==="
    docker compose up -d h-target h1 h2 h3 h4
    docker compose --profile dry-run up h-attack-dry
    ;;

  up)
    echo "=== Subindo rede (sem atacante) ==="
    docker compose up -d h-target h1 h2 h3 h4
    echo "h-target disponível em http://localhost:8080"
    echo "Métricas: http://localhost:8080/metrics"
    ;;

  down)
    echo "=== Encerrando tudo ==="
    docker compose --profile train --profile attack --profile dry-run down
    ;;

  logs)
    docker compose logs -f "${2:-h-attack}"
    ;;

  *)
    echo "Uso: $0 {train|attack|dry-run|up|down|logs}"
    exit 1
    ;;
esac
