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
import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def configurar_logging() -> None:
    """
    Mantém o terminal enxuto em INFO e grava detalhes em DEBUG no arquivo.

    Console: acompanhamento operacional do experimento.
    Arquivo: diagnóstico detalhado para depuração.
    """

    log_dir = Path("/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(
        logging.Formatter("%(message)s")
    )

    arquivo = logging.FileHandler(
        log_dir / "h-attack.log",
        encoding="utf-8",
    )
    arquivo.setLevel(logging.DEBUG)
    arquivo.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] "
            "%(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root.addHandler(console)
    root.addHandler(arquivo)

    # Bibliotecas de visualização podem emitir mensagens operacionais
    # (por exemplo, criação do fontManager) que não pertencem ao experimento.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


configurar_logging()
logger = logging.getLogger("main")


def criar_diretorio_experimento(models_root: Path, seed: int) -> tuple[str, Path]:
    experiments_dir = models_root / "experiments"
    experiments_dir.mkdir(parents=True, exist_ok=True)

    base = datetime.now().strftime("run_%Y%m%d_%H%M%S") + f"_seed{seed}"
    run_id = base
    run_dir = experiments_dir / run_id
    sufixo = 1
    while run_dir.exists():
        run_id = f"{base}_{sufixo:02d}"
        run_dir = experiments_dir / run_id
        sufixo += 1

    for subdir in ("checkpoints", "metrics", "plots", "logs"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    return run_id, run_dir


def adicionar_log_experimento(run_dir: Path) -> None:
    handler = logging.FileHandler(
        run_dir / "logs" / "train.log",
        encoding="utf-8",
    )
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)


def atualizar_latest(models_root: Path, run_dir: Path) -> None:
    latest = models_root / "latest"
    if latest.is_symlink() or latest.exists():
        if latest.is_dir() and not latest.is_symlink():
            raise RuntimeError(f"{latest} existe e não é um symlink")
        latest.unlink()
    latest.symlink_to(run_dir.relative_to(models_root), target_is_directory=True)



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
    from gan.reporting import (
        gerar_graficos,
        salvar_metricas_csv,
        salvar_summary,
    )

    models_root = Path(args.models_dir)
    run_id, run_dir = criar_diretorio_experimento(models_root, args.seed)
    adicionar_log_experimento(run_dir)

    logger.info("=== Dual-GAM | Treinamento ===")
    logger.info(
        "  Dataset: %s | Seed: %d | Device: %s | "
        "Rodadas: %d | Epochs/rodada: %d",
        args.data,
        args.seed,
        args.device,
        args.rodadas,
        args.epochs,
    )
    logger.info("  Experimento: %s", run_id)
    logger.info("")
    logger.info("[1/4] Preparando dataset")

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

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = prep.split_e_normalizar(
        X,
        y,
        test_size=0.2,
        validation_size=0.1,
        random_state=args.seed,
    )

    # Salvar preprocessador.
    prep.salvar(run_dir / "preprocessador")

    # Salvar amostras reais de DDoS
    # provenientes exclusivamente do treino.
    import torch as _torch

    mask_ddos_train = y_train == 1

    X_ddos_export = _torch.FloatTensor(
        X_train[mask_ddos_train]
    )

    _torch.save(
        X_ddos_export,
        run_dir / "ddos_samples.pt",
    )

    logger.info(
        "  Amostras DDoS exportadas: %d",
        len(X_ddos_export),
    )

    logger.info("")
    logger.info("[2/4] Configurando experimento")

    # 2. Configuração do treinamento.
    cfg = TrainingConfig(
        seed=args.seed,
        input_dim=X_train.shape[1],
        noise_dim=args.noise_dim,
        n_rodadas=args.rodadas,
        epochs_por_rodada=args.epochs,
        device=args.device,
        checkpoint_dir=run_dir / "checkpoints",
    )

    # 3. Criar Trainer.
    trainer = AdversarialTrainer(cfg)

    # Salvar configuração completa da execução
    # antes de iniciar o treinamento.
    config_execucao = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "dataset": str(args.data),

        "dados": {
            "input_dim": cfg.input_dim,
            "noise_dim": cfg.noise_dim,
            "validation_size": 0.1,
            "test_size": 0.2,
        },

        "treinamento": {
            "lr_defensor": cfg.lr_defensor,
            "lr_atacante": cfg.lr_atacante,
            "adam_betas_atacante": list(
                cfg.adam_betas
            ),
            "epsilon": cfg.epsilon,
            "classification_threshold": cfg.classification_threshold,
            "amostras_avaliacao_adversarial": (
                cfg.amostras_avaliacao_adversarial
            ),
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

    config_path = run_dir / "config_execucao.json"

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

    logger.info("  Configuração: OK")
    logger.debug("Configuração salva em %s", config_path)

    logger.info("")
    logger.info("[3/4] Executando treinamento")

    # 4. Treino adversarial.
    trainer.carregar_dados(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
    )

    trainer.pretreinar_defensor()

    # D0: Defensor após o pré-treino e antes de qualquer adaptação adversarial.
    trainer.salvar_defensor_inicial()

    historico = trainer.rodar_ciclo()

    logger.info("")
    logger.info("  Avaliando robustez acumulada")
    avaliacao_cruzada = trainer.avaliar_matriz_checkpoints()
    historico["avaliacao_cruzada_checkpoints"] = avaliacao_cruzada

    # O conjunto de teste é reservado para a avaliação final.
    metricas_teste_final = trainer.medir_metricas_teste_final()
    historico["metricas_defensor_teste_final"] = metricas_teste_final
    # Compatibilidade com históricos anteriores.
    historico["acuracia_defensor_teste_final"] = metricas_teste_final["accuracy"]

    logger.info("")
    logger.info("[4/4] Salvando resultados")

    # 5. Salvar modelos finais.
    import torch

    models_path = run_dir

    torch.save(
        trainer.atacante.state_dict(),
        models_path / "checkpoints" / "atacante_final.pth",
    )

    torch.save(
        trainer.defensor.state_dict(),
        models_path / "checkpoints" / "defensor_adaptativo_final.pth",
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

    matriz_path = models_path / "matriz_checkpoints.json"
    with open(
        matriz_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            avaliacao_cruzada,
            f,
            indent=2,
        )

    # A matriz completa também é persistida como CSV para inspeção humana,
    # análise estatística e geração posterior de heatmaps.
    matriz_csv_path = models_path / "matriz_checkpoints.csv"
    colunas_defensores = avaliacao_cruzada["colunas_defensores"]
    linhas_atacantes = avaliacao_cruzada["linhas_atacantes"]
    matriz_evasao = avaliacao_cruzada["matriz_evasao"]

    with open(
        matriz_csv_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Atacante"]
            + [f"D{d}" for d in colunas_defensores]
        )

        for atacante, linha in zip(
            linhas_atacantes,
            matriz_evasao,
        ):
            writer.writerow(
                [f"A{atacante}"]
                + [f"{valor * 100:.4f}" for valor in linha]
            )

    metric_paths = salvar_metricas_csv(historico, run_dir / "metrics")
    summary_path = run_dir / "summary.json"
    salvar_summary(config_execucao, historico, summary_path)
    plot_paths = gerar_graficos(historico, run_dir / "plots")
    atualizar_latest(models_root, run_dir)

    logger.info("  Histórico: OK")
    logger.info("  Matriz de checkpoints: JSON + CSV")
    logger.info("  Métricas: Defensor + Atacante")
    logger.info("  Resumo: OK")
    logger.info("  Gráficos: %d gerados", len(plot_paths))
    logger.info("  Experimento ativo: %s", run_id)

    logger.debug("Histórico salvo em %s", historico_path)
    logger.debug("Matriz de checkpoints (JSON) salva em %s", matriz_path)
    logger.debug("Matriz de checkpoints (CSV) salva em %s", matriz_csv_path)
    logger.debug("Métricas do Defensor salvas em %s", metric_paths["defender"])
    logger.debug("Métricas do Atacante salvas em %s", metric_paths["attacker"])
    logger.debug("Resumo salvo em %s", summary_path)
    logger.debug("Gráficos salvos em %s", run_dir / "plots")
    logger.debug("Latest atualizado: %s -> %s", models_root / "latest", run_dir)

    logger.info("")
    logger.info("=== Treinamento concluído ===")

    evasao_pre_final = (
        historico["taxa_evasao_pre_adaptacao"][-1]
        * 100
    )

    evasao_pos_final = (
        historico["taxa_evasao_pos_adaptacao"][-1]
        * 100
    )

    logger.info(
        "  Evasão final pré-adaptação: %.1f%%",
        evasao_pre_final,
    )
    logger.info(
        "  Evasão final pós-adaptação: %.1f%%",
        evasao_pos_final,
    )
    logger.info(
        "  Teste reservado: Acc %.2f%% | Precision %.2f%% | Recall %.2f%% | "
        "F1 %.2f%% | FPR %.2f%% | FNR %.2f%% | ROC-AUC %.4f",
        metricas_teste_final["accuracy"] * 100,
        metricas_teste_final["precision"] * 100,
        metricas_teste_final["recall"] * 100,
        metricas_teste_final["f1"] * 100,
        metricas_teste_final["fpr"] * 100,
        metricas_teste_final["fnr"] * 100,
        metricas_teste_final["roc_auc"],
    )
    tn, fp = metricas_teste_final["confusion_matrix"][0]
    fn, tp = metricas_teste_final["confusion_matrix"][1]

    logger.info("  Matriz de confusão final:")
    logger.info("                 Pred. Benigno   Pred. DDoS")
    logger.info("    Real Benigno   %12d   %10d", tn, fp)
    logger.info("    Real DDoS      %12d   %10d", fn, tp)


def cmd_attack(
    args: argparse.Namespace,
    dry_run: bool = False,
) -> None:
    """Executa ataques usando modelos treinados."""

    from controller.controller import (
        AttackController,
    )

    logger.info(
        "=== Dual-GAM | %s ===",
        (
            "Dry-run"
            if dry_run
            else "Ataque"
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
        models_dir=Path(args.models_dir),
        preprocessador_dir=None,
        dry_run=dry_run,
        device=args.device,
        checkpoint_mode=args.checkpoint_mode,
        attacker_round=args.attacker_round,
        defender_round=args.defender_round,
    )

    controller.executar_loop(
        n_ciclos=args.ciclos,
        intervalo_entre_ciclos=(
            args.intervalo
        ),
        n_vetores=args.n_vetores,
    )


def adicionar_argumentos_checkpoint(
    parser: argparse.ArgumentParser,
) -> None:
    """Adiciona opções de seleção de checkpoints a attack e dry-run."""

    parser.add_argument(
        "--checkpoint-mode",
        choices=[
            "demo",
            "final",
            "explicit",
        ],
        default="demo",
        help=(
            "demo: maior evasão A_r x D_(r-1); "
            "final: modelos finais; "
            "explicit: rodadas informadas manualmente"
        ),
    )

    parser.add_argument(
        "--attacker-round",
        type=int,
        default=None,
        help=(
            "Rodada do Atacante no modo explicit "
            "(mínimo 1)"
        ),
    )

    parser.add_argument(
        "--defender-round",
        type=int,
        default=None,
        help=(
            "Rodada do Defensor no modo explicit "
            "(0 representa D0, o Defensor pré-treinado)"
        ),
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

    adicionar_argumentos_checkpoint(
        p_attack
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

    adicionar_argumentos_checkpoint(
        p_dry
    )

    args = parser.parse_args()

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
