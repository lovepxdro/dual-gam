from __future__ import annotations

import argparse
import json
from pathlib import Path

from adarena.builtin import (
    BASIC_FLOW_EXTRACTOR_ID,
    BINARY_MLP_DEFENDER_ID,
    CICIDS2017_DATASET_ID,
    CICIDS2017_RENDERER_ID,
    NETWORK_SIMULATION_PROTOCOL_ID,
    PERTURBATION_ATTACKER_ID,
    SCAPY_CAPTURE_ID,
    SCAPY_NETWORK_BACKEND_ID,
    create_default_registry,
)

from adarena.core.config import (
    ComponentSelection,
    ExperimentConfig,
    ExperimentMode,
    NetworkSettings,
    TrainingSettings,
)

from adarena.core.experiment import (
    ExperimentRunner,
)


def _require(
    path: Path,
    description: str,
) -> Path:

    if not path.exists():
        raise FileNotFoundError(
            f"{description} não encontrado: "
            f"{path}"
        )

    return path


def _load_training_settings(
    run_dir: Path,
) -> TrainingSettings:

    config_path = (
        run_dir
        / "config_execucao.json"
    )

    settings = TrainingSettings()

    if not config_path.exists():
        print(
            "[WARN] config_execucao.json "
            "não encontrado."
        )

        print(
            "[WARN] Usando parâmetros "
            "padrão de inferência."
        )

        return settings

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        snapshot = json.load(
            file
        )

    dados = snapshot.get(
        "dados",
        {},
    )

    treinamento = snapshot.get(
        "treinamento",
        {},
    )

    if (
        "noise_dim"
        in dados
    ):
        settings.noise_dim = int(
            dados[
                "noise_dim"
            ]
        )

    if (
        "epsilon"
        in treinamento
    ):
        settings.epsilon = float(
            treinamento[
                "epsilon"
            ]
        )

    if (
        "classification_threshold"
        in treinamento
    ):
        settings.classification_threshold = float(
            treinamento[
                "classification_threshold"
            ]
        )

    return settings


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Smoke test seguro do "
            "NetworkSimulationProtocol "
            "utilizando artefatos reais."
        )
    )

    parser.add_argument(
        "--run-dir",
        required=True,

        help=(
            "Diretório de uma execução "
            "de treinamento da ADArena."
        ),
    )

    parser.add_argument(
        "--dataset",

        default=(
            "data/"
            "DDoS-Friday-no-metadata.parquet"
        ),

        help=(
            "Dataset utilizado para "
            "a simulação."
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=20,

        help=(
            "Quantidade máxima de "
            "amostras DDoS utilizadas."
        ),
    )

    parser.add_argument(
        "--attacker-checkpoint",

        default=(
            "atacante_final.pth"
        ),

        help=(
            "Nome do checkpoint do atacante "
            "dentro de run-dir/checkpoints."
        ),
    )

    parser.add_argument(
        "--defender-checkpoint",

        default=(
            "defensor_adaptativo_final.pth"
        ),

        help=(
            "Nome do checkpoint do defensor "
            "dentro de run-dir/checkpoints."
        ),
    )

    args = parser.parse_args()

    run_dir = Path(
        args.run_dir
    ).resolve()

    dataset_path = Path(
        args.dataset
    ).resolve()

    _require(
        run_dir,
        "Run de treinamento",
    )

    _require(
        dataset_path,
        "Dataset",
    )

    attacker_checkpoint = _require(
        run_dir
        / "checkpoints"
        / args.attacker_checkpoint,

        "Checkpoint do atacante",
    )

    defender_checkpoint = _require(
        run_dir
        / "checkpoints"
        / args.defender_checkpoint,

        "Checkpoint do defensor",
    )

    preprocessor_path = _require(
        run_dir
        / "preprocessador",

        "Preprocessador",
    )

    training = (
        _load_training_settings(
            run_dir
        )
    )

    print(
        "=== ADArena | Smoke Test ==="
    )

    print(
        f"Run de origem: {run_dir}"
    )

    print(
        f"Dataset: {dataset_path}"
    )

    print(
        f"Amostras: {args.samples}"
    )

    print(
        "Checkpoint atacante: "
        f"{attacker_checkpoint.name}"
    )

    print(
        "Checkpoint defensor: "
        f"{defender_checkpoint.name}"
    )

    print(
        f"Epsilon: "
        f"{training.epsilon}"
    )

    print(
        "Threshold: "
        f"{training.classification_threshold}"
    )

    print(
        "Modo de rede: DRY-RUN"
    )

    print(
        "Observação de rede: DESATIVADA"
    )

    print(
        "Transmissão de pacotes: DESATIVADA"
    )

    registry = (
        create_default_registry()
    )

    config = ExperimentConfig(
        attacker=ComponentSelection(
            component_id=(
                PERTURBATION_ATTACKER_ID
            ),

            source=str(
                attacker_checkpoint
            ),
        ),

        defender=ComponentSelection(
            component_id=(
                BINARY_MLP_DEFENDER_ID
            ),

            source=str(
                defender_checkpoint
            ),
        ),

        attack_dataset=(
            ComponentSelection(
                component_id=(
                    CICIDS2017_DATASET_ID
                ),

                source=str(
                    dataset_path
                ),
            )
        ),

        protocol=ComponentSelection(
            component_id=(
                NETWORK_SIMULATION_PROTOCOL_ID
            )
        ),

        mode=(
            ExperimentMode.SIMULATE
        ),

        seed=42,

        device="cpu",

        # Não mexe no models/latest
        # utilizado pelos treinamentos.
        output_dir=(
            "models/"
            "network-smoke"
        ),

        training=training,

        network=NetworkSettings(
            renderer=ComponentSelection(
                component_id=(
                    CICIDS2017_RENDERER_ID
                ),

                params={
                    # O renderer exige um alvo
                    # sintático para construir os
                    # parâmetros traduzidos.
                    #
                    # O backend continuará em
                    # dry-run e não transmitirá
                    # esses parâmetros.
                    "target_ip": (
                        "10.0.0.10"
                    ),

                    "target_port": 80,
                },
            ),

            network_backend=(
                ComponentSelection(
                    component_id=(
                        SCAPY_NETWORK_BACKEND_ID
                    ),

                    params={
                        "require_private_target": (
                            True
                        ),
                    },
                )
            ),

            # Ainda são declarados porque
            # fazem parte do contrato da
            # configuração de rede.
            #
            # observe=False significa que
            # eles não serão executados.
            capture=ComponentSelection(
                component_id=(
                    SCAPY_CAPTURE_ID
                )
            ),

            extractor=ComponentSelection(
                component_id=(
                    BASIC_FLOW_EXTRACTOR_ID
                )
            ),

            preprocessor_source=str(
                preprocessor_path
            ),

            sample_count=(
                args.samples
            ),

            classification_threshold=(
                training
                .classification_threshold
            ),

            dry_run=True,

            observe=False,
        ),
    )

    runner = ExperimentRunner(
        config=config,
        registry=registry,
    )

    result = runner.run()

    print("")
    print(
        "=== Resultado ==="
    )

    metrics = (
        result.final_metrics
    )

    print(
        "Amostras selecionadas: "
        f"{metrics['samples_selected']}"
    )

    print(
        "Evasões matemáticas: "
        f"{metrics['mathematical_evasions']}"
    )

    print(
        "Taxa de evasão matemática: "
        f"{metrics['mathematical_evasion_rate']:.2%}"
    )

    print(
        "Traduções válidas: "
        f"{metrics['valid_renderings']}"
    )

    print(
        "Execuções do backend dry-run: "
        f"{metrics['dry_run_executions']}"
    )

    print(
        "Dry-run: "
        f"{metrics['dry_run']}"
    )

    print(
        "Observação: "
        f"{metrics['observation_enabled']}"
    )

    print("")
    print(
        f"Artefatos: {result.run_dir}"
    )


if __name__ == "__main__":
    main()
