"""
ADArena — CLI

A CLI apenas traduz argumentos do usuário para
ExperimentConfig e entrega a execução ao Core.
"""

import argparse
import logging
import sys

from pathlib import Path

from adarena.builtin import (
    ADVERSARIAL_PROTOCOL_ID,
    BINARY_MLP_DEFENDER_ID,
    CICIDS2017_DATASET_ID,
    CICIDS2017_RENDERER_ID,
    PERTURBATION_ATTACKER_ID,
    SCAPY_NETWORK_BACKEND_ID,
    create_default_registry,
)

from adarena.core import (
    ComponentKind,
    ComponentSelection,
    ExperimentConfig,
    ExperimentRunner,
    TrainingSettings,
)


def configurar_logging() -> None:
    log_dir = Path(
        "/logs"
    )

    log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    root = logging.getLogger()

    root.setLevel(
        logging.DEBUG
    )

    root.handlers.clear()

    console = (
        logging.StreamHandler(
            sys.stdout
        )
    )

    console.setLevel(
        logging.INFO
    )

    console.setFormatter(
        logging.Formatter(
            "%(message)s"
        )
    )

    arquivo = logging.FileHandler(
        log_dir
        / "h-attack.log",

        encoding="utf-8",
    )

    arquivo.setLevel(
        logging.DEBUG
    )

    arquivo.setFormatter(
        logging.Formatter(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(name)s — %(message)s",

            datefmt=(
                "%Y-%m-%d %H:%M:%S"
            ),
        )
    )

    root.addHandler(
        console
    )

    root.addHandler(
        arquivo
    )

    logging.getLogger(
        "matplotlib"
    ).setLevel(
        logging.WARNING
    )


configurar_logging()

logger = logging.getLogger(
    "main"
)


def cmd_train(
    args: argparse.Namespace,
) -> None:

    registry = (
        create_default_registry()
    )

    config = ExperimentConfig(
        attacker=ComponentSelection(
            component_id=(
                args.attacker_model
            ),
        ),

        defender=ComponentSelection(
            component_id=(
                args.defender_model
            ),
        ),

        attack_dataset=(
            ComponentSelection(
                component_id=(
                    args.dataset
                ),

                source=args.data,

                params={
                    "attack_labels": (
                        args.attack_labels
                    ),
                },
            )
        ),

        protocol=ComponentSelection(
            component_id=(
                args.protocol
            ),
        ),

        seed=args.seed,

        device=args.device,

        output_dir=(
            args.models_dir
        ),

        training=TrainingSettings(
            noise_dim=(
                args.noise_dim
            ),

            n_rodadas=(
                args.rodadas
            ),

            epochs_por_rodada=(
                args.epochs
            ),
        ),
    )

    runner = ExperimentRunner(
        config=config,
        registry=registry,
    )

    result = runner.run()

    logger.info(
        "  Run finalizado: %s",
        result.run_id,
    )


def cmd_attack(
    args: argparse.Namespace,
    dry_run: bool = False,
) -> None:

    from controller.controller import (
        AttackController,
    )

    logger.info(
        "=== ADArena | %s ===",
        (
            "Dry-run"
            if dry_run
            else "Execução experimental"
        ),
    )

    logger.info(
        "Target experimental: %s:%d",
        args.target,
        args.port,
    )

    controller = AttackController(
        target_ip=args.target,

        target_port=args.port,

        models_dir=Path(
            args.models_dir
        ),

        preprocessador_dir=None,

        dry_run=dry_run,

        device=args.device,

        checkpoint_mode=(
            args.checkpoint_mode
        ),

        attacker_round=(
            args.attacker_round
        ),

        defender_round=(
            args.defender_round
        ),

        renderer_component_id=(
            args.renderer
        ),

        network_backend_component_id=(
            args.network_backend
        ),
    )

    controller.executar_loop(
        n_ciclos=(
            args.ciclos
        ),

        intervalo_entre_ciclos=(
            args.intervalo
        ),

        n_vetores=(
            args.n_vetores
        ),
    )


def adicionar_argumentos_checkpoint(
    parser: argparse.ArgumentParser,
) -> None:

    parser.add_argument(
        "--checkpoint-mode",

        choices=[
            "demo",
            "final",
            "explicit",
        ],

        default="demo",
    )

    parser.add_argument(
        "--attacker-round",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--defender-round",
        type=int,
        default=None,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "ADArena — "
            "Adaptive Defense Arena"
        )
    )

    parser.add_argument(
        "--models-dir",
        default="/models",
    )

    parser.add_argument(
        "--device",

        default="cpu",

        choices=[
            "cpu",
            "cuda",
        ],
    )

    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
    )

    registry = (
        create_default_registry()
    )

    attackers = [
        spec.component_id
        for spec
        in registry.list(
            kind=(
                ComponentKind.ATTACKER
            )
        )
    ]

    defenders = [
        spec.component_id
        for spec
        in registry.list(
            kind=(
                ComponentKind.DEFENDER
            )
        )
    ]

    datasets = [
        spec.component_id
        for spec
        in registry.list(
            kind=(
                ComponentKind.DATASET
            )
        )
    ]

    protocols = [
        spec.component_id
        for spec
        in registry.list(
            kind=(
                ComponentKind
                .EXPERIMENT_PROTOCOL
            )
        )
    ]

    renderers = [
        spec.component_id
        for spec
        in registry.list(
            kind=(
                ComponentKind.RENDERER
            )
        )
    ]


    network_backends = [
        spec.component_id
        for spec
        in registry.list(
            kind=(
                ComponentKind
                .NETWORK_BACKEND
            )
        )
    ]

    # ── train ──────────────────────────

    train = sub.add_parser(
        "train"
    )

    train.add_argument(
        "--data",
        required=True,
    )

    train.add_argument(
        "--dataset",

        default=(
            CICIDS2017_DATASET_ID
        ),

        choices=datasets,
    )

    train.add_argument(
        "--attack-label",

        dest="attack_labels",

        action="append",

        default=None,
    )

    train.add_argument(
        "--attacker-model",

        default=(
            PERTURBATION_ATTACKER_ID
        ),

        choices=attackers,
    )

    train.add_argument(
        "--defender-model",

        default=(
            BINARY_MLP_DEFENDER_ID
        ),

        choices=defenders,
    )

    train.add_argument(
        "--protocol",

        default=(
            ADVERSARIAL_PROTOCOL_ID
        ),

        choices=protocols,
    )

    train.add_argument(
        "--rodadas",
        type=int,
        default=20,
    )

    train.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    train.add_argument(
        "--noise-dim",
        type=int,
        default=32,
    )

    train.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # ── attack ─────────────────────────

    attack = sub.add_parser(
        "attack"
    )

    attack.add_argument(
        "--target",
        required=True,
    )

    attack.add_argument(
        "--port",
        type=int,
        default=80,
    )

    attack.add_argument(
        "--ciclos",
        type=int,
        default=10,
    )

    attack.add_argument(
        "--intervalo",
        type=float,
        default=5.0,
    )

    attack.add_argument(
        "--n-vetores",
        type=int,
        default=100,
    )

    attack.add_argument(
        "--renderer",

        default=(
            CICIDS2017_RENDERER_ID
        ),

        choices=renderers,
    )


    attack.add_argument(
        "--network-backend",

        default=(
            SCAPY_NETWORK_BACKEND_ID
        ),

        choices=network_backends,
    )

    adicionar_argumentos_checkpoint(
        attack
    )

    # ── dry-run ────────────────────────

    dry = sub.add_parser(
        "dry-run"
    )

    dry.add_argument(
        "--target",
        required=True,
    )

    dry.add_argument(
        "--port",
        type=int,
        default=80,
    )

    dry.add_argument(
        "--ciclos",
        type=int,
        default=3,
    )

    dry.add_argument(
        "--intervalo",
        type=float,
        default=2.0,
    )

    dry.add_argument(
        "--n-vetores",
        type=int,
        default=20,
    )

    dry.add_argument(
        "--renderer",

        default=(
            CICIDS2017_RENDERER_ID
        ),

        choices=renderers,
    )


    dry.add_argument(
        "--network-backend",

        default=(
            SCAPY_NETWORK_BACKEND_ID
        ),

        choices=network_backends,
    )

    adicionar_argumentos_checkpoint(
        dry
    )

    args = parser.parse_args()

    if args.cmd == "train":
        cmd_train(
            args
        )

    elif args.cmd == "attack":
        cmd_attack(
            args,
            dry_run=False,
        )

    elif args.cmd == "dry-run":
        cmd_attack(
            args,
            dry_run=True,
        )


if __name__ == "__main__":
    main()
