"""
Dual-GAM — Entrypoint do container h-attack

Modos de execução:
  train   — Treina GAN (pré-treino + ciclo adversarial) e salva modelos
  attack  — Carrega modelos treinados e executa ataques reais na rede
  dry-run — Igual a attack mas sem enviar pacotes (para testes)

Uso:
  python main.py train --data /data/DDoS-Friday-no-metadata.parquet
  python main.py attack --target 172.20.0.10 --port 80
  python main.py dry-run --target 172.20.0.10
"""

import argparse
import json
import logging
import sys
from pathlib import Path


# Logging configurado antes de qualquer import interno.
logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s [%(levelname)s] "
        "%(name)s — %(message)s"
    ),
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            "/logs/h-attack.log"
        ),
    ],
)

logger = logging.getLogger("main")


def cmd_train(
    args: argparse.Namespace,
) -> None:
    """Treina os modelos GAN com o dataset CIC-IDS2017."""

    from gan.preprocessing import Preprocessador
    from gan.trainer import (
        AdversarialTrainer,
        TrainingConfig,
        definir_seed,
    )

    logger.info("=== MODO TREINO ===")
    logger.info(
        "Dataset: %s",
        args.data,
    )

    # A seed deve ser definida antes:
    # - do split;
    # - da criação dos modelos;
    # - da criação dos DataLoaders.
    definir_seed(args.seed)

    # 1. Pré-processamento.
    prep = Preprocessador()

    X, y = prep.carregar_parquet(
        args.data
    )

    X_train, X_test, y_train, y_test = (
        prep.split_e_normalizar(
            X,
            y,
            random_state=args.seed,
        )
    )

    # Salvar preprocessador.
    prep.salvar(
        Path(args.models_dir)
        / "preprocessador"
    )

    # Salvar amostras reais de DDoS
    # provenientes exclusivamente do treino.
    import torch as _torch

    mask_ddos_train = y_train == 1

    X_ddos_export = _torch.FloatTensor(
        X_train[mask_ddos_train]
    )

    _torch.save(
        X_ddos_export,
        Path(args.models_dir)
        / "ddos_samples.pt",
    )

    logger.info(
        "Amostras DDoS exportadas: %d",
        len(X_ddos_export),
    )

    # 2. Configuração do treinamento.
    cfg = TrainingConfig(
        seed=args.seed,
        input_dim=X_train.shape[1],
        noise_dim=args.noise_dim,
        n_rodadas=args.rodadas,
        epochs_por_rodada=args.epochs,
        device=args.device,
        checkpoint_dir=Path(
            args.models_dir
        ),
    )

    # 3. Criar Trainer.
    trainer = AdversarialTrainer(cfg)

    # Salvar configuração completa da execução
    # antes de iniciar o treinamento.
    config_execucao = {
        "seed": cfg.seed,
        "dataset": str(args.data),

        "dados": {
            "input_dim": cfg.input_dim,
            "noise_dim": cfg.noise_dim,
            "test_size": 0.2,
        },

        "treinamento": {
            "lr_defensor": cfg.lr_defensor,
            "lr_atacante": cfg.lr_atacante,
            "adam_betas_atacante": list(
                cfg.adam_betas
            ),
            "epsilon": cfg.epsilon,
            "epochs_pretrain": (
                cfg.epochs_pretrain
            ),
            "epochs_por_rodada": (
                cfg.epochs_por_rodada
            ),
            "n_rodadas": cfg.n_rodadas,
            "amostras_por_rodada": (
                cfg.amostras_por_rodada
            ),
            "batch_size": cfg.batch_size,
            "device": cfg.device,
        },

        "arquitetura": {
            "defensor": str(
                trainer.defensor
            ),
            "atacante": str(
                trainer.atacante
            ),
        },
    }

    config_path = (
        Path(args.models_dir)
        / "config_execucao.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config_execucao,
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Configuração da execução salva em %s",
        config_path,
    )

    # 4. Treino adversarial.
    trainer.carregar_dados(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    trainer.pretreinar_defensor()

    historico = trainer.rodar_ciclo()

    # 5. Salvar modelos finais.
    import torch

    models_path = Path(
        args.models_dir
    )

    torch.save(
        trainer.atacante.state_dict(),
        models_path
        / "atacante_final.pth",
    )

    torch.save(
        trainer.defensor.state_dict(),
        models_path
        / "defensor_adaptativo_final.pth",
    )

    # 6. Salvar histórico.
    historico_path = (
        models_path
        / "historico_treino.json"
    )

    with open(
        historico_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            historico,
            f,
            indent=2,
        )

    logger.info(
        "Histórico salvo em %s",
        historico_path,
    )

    logger.info(
        "=== TREINO CONCLUÍDO ==="
    )

    taxa_final = (
        historico["taxa_evasao"][-1]
        * 100
    )

    acc_final = (
        historico[
            "acuracia_defensor"
        ][-1]
        * 100
    )

    logger.info(
        "Taxa de evasão final: %.1f%%",
        taxa_final,
    )

    logger.info(
        "Acurácia defensor final: %.2f%%",
        acc_final,
    )


def cmd_attack(
    args: argparse.Namespace,
    dry_run: bool = False,
) -> None:
    """Executa ataques usando modelos treinados."""

    from controller.controller import (
        AttackController,
    )

    logger.info(
        "=== MODO %s ===",
        (
            "DRY-RUN"
            if dry_run
            else "ATAQUE"
        ),
    )

    logger.info(
        "Target: %s:%d",
        args.target,
        args.port,
    )

    controller = AttackController(
        target_ip=args.target,
        target_port=args.port,
        models_dir=Path(
            args.models_dir
        ),
        preprocessador_dir=(
            Path(args.models_dir)
            / "preprocessador"
        ),
        dry_run=dry_run,
        device=args.device,
    )

    controller.executar_loop(
        n_ciclos=args.ciclos,
        intervalo_entre_ciclos=(
            args.intervalo
        ),
        n_vetores=args.n_vetores,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-GAM h-attack"
    )

    parser.add_argument(
        "--models-dir",
        default="/models",
        help="Diretório dos modelos",
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

    # ── train ─────────────────────────────

    p_train = sub.add_parser(
        "train",
        help="Treinar modelos GAN",
    )

    p_train.add_argument(
        "--data",
        required=True,
        help=(
            "Path do .parquet "
            "CIC-IDS2017"
        ),
    )

    p_train.add_argument(
        "--rodadas",
        type=int,
        default=20,
    )

    p_train.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    p_train.add_argument(
        "--noise-dim",
        type=int,
        default=32,
    )

    p_train.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Seed para "
            "reprodutibilidade"
        ),
    )

    # ── attack ────────────────────────────

    p_attack = sub.add_parser(
        "attack",
        help="Executar ataques reais",
    )

    p_attack.add_argument(
        "--target",
        required=True,
        help="IP do h-target",
    )

    p_attack.add_argument(
        "--port",
        type=int,
        default=80,
    )

    p_attack.add_argument(
        "--ciclos",
        type=int,
        default=10,
    )

    p_attack.add_argument(
        "--intervalo",
        type=float,
        default=5.0,
    )

    p_attack.add_argument(
        "--n-vetores",
        type=int,
        default=100,
    )

    # ── dry-run ───────────────────────────

    p_dry = sub.add_parser(
        "dry-run",
        help=(
            "Simular ataques "
            "sem enviar pacotes"
        ),
    )

    p_dry.add_argument(
        "--target",
        required=True,
    )

    p_dry.add_argument(
        "--port",
        type=int,
        default=80,
    )

    p_dry.add_argument(
        "--ciclos",
        type=int,
        default=3,
    )

    p_dry.add_argument(
        "--intervalo",
        type=float,
        default=2.0,
    )

    p_dry.add_argument(
        "--n-vetores",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    Path("/logs").mkdir(
        exist_ok=True
    )

    if args.cmd == "train":
        cmd_train(args)

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
