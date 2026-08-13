#!/usr/bin/env bash
# Dual-GAM — Script principal de execução
#
# Uso:
#   ./scripts/run.sh train [dataset.parquet] [--verbose]
#   ./scripts/run.sh attack [demo|final|explicit <A> <D>]
#   ./scripts/run.sh dry-run [demo|final|explicit <A> <D>]
#   ./scripts/run.sh build
#   ./scripts/run.sh up
#   ./scripts/run.sh down
#   ./scripts/run.sh logs [serviço]
#   ./scripts/run.sh experiments
#   ./scripts/run.sh results
#   ./scripts/run.sh test

set -euo pipefail
cd "$(dirname "$0")/.."

DEFAULT_DATASET="./data/DDoS-Friday-no-metadata.parquet"
TARGET_IP="172.20.0.10"
TARGET_PORT="80"

usage() {
  cat <<'USAGE'
Uso:
  ./scripts/run.sh train [dataset.parquet] [--verbose]
  ./scripts/run.sh attack [demo|final|explicit <A> <D>]
  ./scripts/run.sh dry-run [demo|final|explicit <A> <D>]
  ./scripts/run.sh build
  ./scripts/run.sh up
  ./scripts/run.sh down
  ./scripts/run.sh logs [serviço]
  ./scripts/run.sh experiments
  ./scripts/run.sh results
  ./scripts/run.sh test

Exemplos:
  ./scripts/run.sh train ./data/DDoS-Friday-no-metadata.parquet
  ./scripts/run.sh train ./data/DDoS-Friday-no-metadata.parquet --verbose
  ./scripts/run.sh attack
  ./scripts/run.sh attack final
  ./scripts/run.sh attack explicit 12 11
  ./scripts/run.sh dry-run explicit 1 0
  ./scripts/run.sh experiments
  ./scripts/run.sh results
  ./scripts/run.sh test
USAGE
}

checkpoint_args() {
  local mode="${1:-demo}"

  case "$mode" in
    demo|final)
      printf '%s\n' "--checkpoint-mode" "$mode"
      ;;

    explicit)
      local attacker_round="${2:-}"
      local defender_round="${3:-}"

      if [[ -z "$attacker_round" || -z "$defender_round" ]]; then
        echo "Erro: modo explicit exige as rodadas do Atacante e do Defensor." >&2
        echo "Exemplo: ./scripts/run.sh attack explicit 12 11" >&2
        exit 1
      fi

      printf '%s\n' \
        "--checkpoint-mode" "explicit" \
        "--attacker-round" "$attacker_round" \
        "--defender-round" "$defender_round"
      ;;

    *)
      echo "Erro: modo de checkpoint inválido: $mode" >&2
      echo "Use: demo, final ou explicit <A> <D>." >&2
      exit 1
      ;;
  esac
}

command="${1:-help}"
shift || true

case "$command" in
  train)
    dataset="$DEFAULT_DATASET"
    verbose=false

    for arg in "$@"; do
      case "$arg" in
        --verbose|-v)
          verbose=true
          ;;
        *)
          dataset="$arg"
          ;;
      esac
    done

    if [[ ! -f "$dataset" ]]; then
      echo "Erro: dataset não encontrado: $dataset" >&2
      exit 1
    fi

    dataset_abs="$(realpath "$dataset")"

    echo "=== Dual-GAM | Treinamento ==="
    echo "  Dataset: $dataset_abs"
    echo ""
    echo "[1/2] Preparando imagem de treinamento"

    if [[ "$verbose" == true ]]; then
      docker compose --profile train build --progress=plain h-attack-train
    else
      docker compose --profile train build --quiet h-attack-train
    fi

    echo ""
    echo "[2/2] Executando treinamento"
    docker compose --profile train run --rm \
      -v "$dataset_abs:/data/DDoS-Friday-no-metadata.parquet:ro" \
      h-attack-train \
      train --data /data/DDoS-Friday-no-metadata.parquet

    echo ""
    echo "✓ Treino concluído."
    echo "  Execute './scripts/run.sh results' para exportar os resultados."
    echo "  Execute './scripts/run.sh attack' para iniciar os ataques."
    ;;

  attack)
    mode="${1:-demo}"
    attacker_round="${2:-}"
    defender_round="${3:-}"
    mapfile -t ckpt_args < <(checkpoint_args "$mode" "$attacker_round" "$defender_round")

    echo "=== Dual-GAM | Ataque ==="
    echo "Checkpoint: $mode"
    echo "ATENÇÃO: use apenas em ambiente de laboratório isolado."
    read -r -p "Confirmar? (s/N): " confirm
    [[ "$confirm" == "s" || "$confirm" == "S" ]] || exit 0

    docker compose up -d h-target h1 h2 h3 h4
    docker compose --profile attack run --rm h-attack \
      attack \
      --target "$TARGET_IP" \
      --port "$TARGET_PORT" \
      --ciclos 10 \
      "${ckpt_args[@]}"
    ;;

  dry-run)
    mode="${1:-demo}"
    attacker_round="${2:-}"
    defender_round="${3:-}"
    mapfile -t ckpt_args < <(checkpoint_args "$mode" "$attacker_round" "$defender_round")

    echo "=== Dual-GAM | Dry-run ==="
    echo "Checkpoint: $mode"

    docker compose up -d h-target h1 h2 h3 h4
    docker compose --profile dry-run run --rm h-attack-dry \
      dry-run \
      --target "$TARGET_IP" \
      --port "$TARGET_PORT" \
      --ciclos 3 \
      "${ckpt_args[@]}"
    ;;

  build)
    echo "=== Dual-GAM | Build ==="
    docker compose build h-target h-attack-train
    ;;

  up)
    echo "=== Dual-GAM | Rede ==="
    docker compose up -d h-target h1 h2 h3 h4
    echo "h-target: http://localhost:8080"
    echo "Métricas: http://localhost:8080/metrics"
    ;;

  down)
    echo "=== Encerrando Dual-GAM ==="
    docker compose --profile train --profile attack --profile dry-run down
    ;;

  logs)
    docker compose logs -f "${1:-h-attack}"
    ;;

  experiments)
    echo "=== Dual-GAM | Experimentos ==="
    docker compose --profile train run --rm --entrypoint sh h-attack-train -c '
      if [ ! -d /models/experiments ]; then
        echo "Nenhum experimento encontrado."
        exit 0
      fi
      latest="$(readlink -f /models/latest 2>/dev/null || true)"
      for dir in $(ls -1dt /models/experiments/* 2>/dev/null); do
        name=$(basename "$dir")
        if [ "$dir" = "$latest" ]; then
          printf "* %s (latest)\n" "$name"
        else
          printf "  %s\n" "$name"
        fi
      done
    '
    ;;

  test)
    echo "=== Dual-GAM | Testes automatizados ==="
    echo "[1/2] Preparando imagem de teste"
    docker compose --profile dry-run build --quiet h-attack-dry
    echo ""
    echo "[2/2] Executando pytest"
    docker compose --profile dry-run run --rm \
      --entrypoint python \
      h-attack-dry -m pytest -v
    ;;

  results)
    mkdir -p ./results
    results_abs="$(realpath ./results)"
    echo "=== Dual-GAM | Exportando resultado mais recente ==="
    docker compose --profile train run --rm \
      -v "$results_abs:/export" \
      --entrypoint sh h-attack-train -c '
        latest="$(readlink -f /models/latest)"
        if [ -z "$latest" ] || [ ! -d "$latest" ]; then
          echo "Nenhum experimento latest encontrado." >&2
          exit 1
        fi
        run=$(basename "$latest")
        rm -rf "/export/$run"
        mkdir -p "/export/$run"
        cp -a "$latest"/. "/export/$run"/
        echo "$run"
      '
    echo "Resultados exportados para: ./results/"
    ;;

  help|-h|--help)
    usage
    ;;

  *)
    usage
    exit 1
    ;;
esac
