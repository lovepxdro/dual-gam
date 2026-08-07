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
import logging
import os
import sys
from pathlib import Path


# Logging configurado antes de qualquer import interno.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/logs/h-attack.log"),
    ],
)

logger = logging.getLogger("main")


def cmd_train(args: argparse.Namespace) -> None:
    """Treina os modelos GAN com o dataset CIC-IDS2017."""
    from gan.preprocessing import Preprocessador
    from gan.trainer import AdversarialTrainer, TrainingConfig

    logger.info("=== MODO TREINO ===")
    logger.info("Dataset: %s", args.data)

    # 1. Pré-processamento.
    #
    # O dataset é carregado sem normalização.
    # O split treino/teste ocorre antes do ajuste do StandardScaler,
    # evitando vazamento de informações do conjunto de teste.
    prep = Preprocessador()

    X, y = prep.carregar_parquet(args.data)

    X_train, X_test, y_train, y_test = prep.split_e_normalizar(
        X,
        y,
    )

    # Salvar preprocessador.
    # Necessário posteriormente para o Translator desnormalizar
    # os vetores gerados pelo atacante.
    prep.salvar(
        Path(args.models_dir) / "preprocessador"
    )

    # Salvar amostras reais de DDoS para uso do Controller.
    #
    # As amostras vêm exclusivamente do conjunto de treino.
    import torch as _torch

    mask_ddos_train = y_train == 1

    X_ddos_export = _torch.FloatTensor(
        X_train[mask_ddos_train]
    )

    _torch.save(
        X_ddos_export,
        Path(args.models_dir) / "ddos_samples.pt",
    )

    logger.info(
        "Amostras DDoS exportadas: %d",
        len(X_ddos_export),
    )

    # 2. Configuração do treino.
    cfg = TrainingConfig(
        input_dim=X_train.shape[1],
        noise_dim=args.noise_dim,
        n_rodadas=args.rodadas,
        epochs_por_rodada=args.epochs,
        device=args.device,
        checkpoint_dir=Path(args.models_dir),
    )

    # 3. Treino adversarial.
    trainer = AdversarialTrainer(cfg)

    trainer.carregar_dados(
        X_train,
        y_train,
        X_test,
        y_test,
    )

    trainer.pretreinar_defensor()

    historico = trainer.rodar_ciclo()

    # 4. Salvar modelos finais.
    import torch

    models_path = Path(args.models_dir)

    torch.save(
        trainer.atacante.state_dict(),
        models_path / "atacante_final.pth",
    )

    torch.save(
        trainer.defensor.state_dict(),
        models_path / "defensor_adaptativo_final.pth",
    )

    # 5. Salvar histórico com taxa de evasão por rodada.
    # Utilizado posteriormente pelo Controller para escolher checkpoints.
    import json

    historico_path = (
        Path(args.models_dir) / "historico_treino.json"
    )

    with open(historico_path, "w") as f:
        json.dump(
            historico,
            f,
            indent=2,
        )

    logger.info(
        "Histórico salvo em %s",
        historico_path,
    )

    logger.info("=== TREINO CONCLUÍDO ===")

    taxa_final = (
        historico["taxa_evasao"][-1] * 100
    )

    acc_final = (
        historico["acuracia_defensor"][-1] * 100
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
    from controller.controller import AttackController

    logger.info(
        "=== MODO %s ===",
        "DRY-RUN" if dry_run else "ATAQUE",
    )

    logger.info(
        "Target: %s:%d",
        args.target,
        args.port,
    )

    controller = AttackController(
        target_ip=args.target,
        target_port=args.port,
        models_dir=Path(args.models_dir),
        preprocessador_dir=(
            Path(args.models_dir) / "preprocessador"
        ),
        dry_run=dry_run,
        device=args.device,
    )

    controller.executar_loop(
        n_ciclos=args.ciclos,
        intervalo_entre_ciclos=args.intervalo,
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
        choices=["cpu", "cuda"],
    )

    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
    )

    # Subcomando: train.
    p_train = sub.add_parser(
        "train",
        help="Treinar modelos GAN",
    )

    p_train.add_argument(
        "--data",
        required=True,
        help="Path do .parquet CIC-IDS2017",
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

    # Subcomando: attack.
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

    # Subcomando: dry-run.
    p_dry = sub.add_parser(
        "dry-run",
        help="Simular ataques sem enviar pacotes",
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

    # Criar diretório de logs se não existir.
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
