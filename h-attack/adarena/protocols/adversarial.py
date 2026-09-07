from __future__ import annotations

import csv
import json
import logging

from pathlib import Path

import torch

from gan.reporting import (
    gerar_graficos,
    salvar_metricas_csv,
    salvar_summary,
)

from gan.trainer import (
    AdversarialTrainer,
    TrainingConfig,
    definir_seed,
)

from .base import (
    ExperimentProtocol,
    ProtocolContext,
    ProtocolResult,
)


logger = logging.getLogger(
    __name__
)


class AdversarialTrainingProtocol(
    ExperimentProtocol
):
    """
    Protocolo utilizado pela linha 1.x da ADArena.

    O Core não conhece AdversarialTrainer.
    Todo o conhecimento sobre o ciclo atacante ↔
    defensor fica encapsulado aqui.
    """

    def run(
        self,
        context: ProtocolContext,
    ) -> ProtocolResult:

        config = context.config
        settings = config.training

        definir_seed(
            config.seed
        )

        logger.info("")
        logger.info(
            "[2/4] Configurando protocolo"
        )

        training_config = (
            TrainingConfig(
                seed=config.seed,

                input_dim=(
                    context
                    .X_train
                    .shape[1]
                ),

                noise_dim=(
                    settings.noise_dim
                ),

                lr_defensor=(
                    settings.lr_defensor
                ),

                lr_atacante=(
                    settings.lr_atacante
                ),

                adam_betas=(
                    settings.adam_betas
                ),

                epsilon=(
                    settings.epsilon
                ),

                classification_threshold=(
                    settings
                    .classification_threshold
                ),

                epochs_pretrain=(
                    settings
                    .epochs_pretrain
                ),

                epochs_por_rodada=(
                    settings
                    .epochs_por_rodada
                ),

                n_rodadas=(
                    settings.n_rodadas
                ),

                amostras_por_rodada=(
                    settings
                    .amostras_por_rodada
                ),

                amostras_avaliacao_adversarial=(
                    settings
                    .amostras_avaliacao_adversarial
                ),

                batch_size=(
                    settings.batch_size
                ),

                device=config.device,

                checkpoint_dir=(
                    context.run_dir
                    / "checkpoints"
                ),
            )
        )

        trainer = (
            AdversarialTrainer(
                training_config,

                registry=(
                    context.registry
                ),

                attacker_component_id=(
                    config
                    .attacker
                    .component_id
                ),

                defender_component_id=(
                    config
                    .defender
                    .component_id
                ),
            )
        )

        context.config_snapshot[
            "arquitetura"
        ] = {
            "defensor": str(
                trainer.defensor
            ),

            "atacante": str(
                trainer.atacante
            ),
        }

        self._save_json(
            context.run_dir
            / "config_execucao.json",

            context.config_snapshot,
        )

        logger.info(
            "  Protocolo: treinamento "
            "adversarial"
        )

        logger.info("")
        logger.info(
            "[3/4] Executando treinamento"
        )

        trainer.carregar_dados(
            context.X_train,
            context.y_train,

            context.X_val,
            context.y_val,

            context.X_test,
            context.y_test,
        )

        trainer.pretreinar_defensor()

        trainer.salvar_defensor_inicial()

        history = (
            trainer.rodar_ciclo()
        )

        logger.info("")
        logger.info(
            "  Avaliando robustez acumulada"
        )

        cross_evaluation = (
            trainer
            .avaliar_matriz_checkpoints()
        )

        history[
            "avaliacao_cruzada_checkpoints"
        ] = cross_evaluation

        final_metrics = (
            trainer
            .medir_metricas_teste_final()
        )

        history[
            "metricas_defensor_teste_final"
        ] = final_metrics

        history[
            "acuracia_defensor_teste_final"
        ] = final_metrics[
            "accuracy"
        ]

        logger.info("")
        logger.info(
            "[4/4] Salvando resultados"
        )

        artifacts = (
            self._save_results(
                context=context,
                trainer=trainer,
                history=history,
                cross_evaluation=(
                    cross_evaluation
                ),
            )
        )

        self._log_final_metrics(
            history,
            final_metrics,
        )

        return ProtocolResult(
            history=history,

            final_metrics=(
                final_metrics
            ),

            evaluations={
                "checkpoint_matrix": (
                    cross_evaluation
                ),
            },

            artifacts=artifacts,
        )

    def _save_results(
        self,
        *,
        context: ProtocolContext,
        trainer: AdversarialTrainer,
        history: dict,
        cross_evaluation: dict,
    ) -> dict[str, str]:

        run_dir = context.run_dir

        attacker_final = (
            run_dir
            / "checkpoints"
            / "atacante_final.pth"
        )

        defender_final = (
            run_dir
            / "checkpoints"
            / "defensor_adaptativo_final.pth"
        )

        torch.save(
            trainer
            .atacante
            .state_dict(),

            attacker_final,
        )

        torch.save(
            trainer
            .defensor
            .state_dict(),

            defender_final,
        )

        # Nome canônico v2.
        attack_samples = (
            run_dir
            / "attack_samples.pt"
        )

        attack_mask = (
            context.y_train
            == 1
        )

        attack_tensor = (
            torch.FloatTensor(
                context.X_train[
                    attack_mask
                ]
            )
        )

        torch.save(
            attack_tensor,
            attack_samples,
        )

        # Compatibilidade temporária
        # com o Controller da linha 1.x.
        legacy_samples = (
            run_dir
            / "ddos_samples.pt"
        )

        torch.save(
            attack_tensor,
            legacy_samples,
        )

        history_path = (
            run_dir
            / "historico_treino.json"
        )

        self._save_json(
            history_path,
            history,
        )

        matrix_json = (
            run_dir
            / "matriz_checkpoints.json"
        )

        self._save_json(
            matrix_json,
            cross_evaluation,
        )

        matrix_csv = (
            run_dir
            / "matriz_checkpoints.csv"
        )

        self._save_matrix_csv(
            cross_evaluation,
            matrix_csv,
        )

        metric_paths = (
            salvar_metricas_csv(
                history,
                run_dir
                / "metrics",
            )
        )

        summary_path = (
            run_dir
            / "summary.json"
        )

        salvar_summary(
            context.config_snapshot,
            history,
            summary_path,
        )

        plot_paths = (
            gerar_graficos(
                history,
                run_dir
                / "plots",
            )
        )

        logger.info(
            "  Histórico: OK"
        )

        logger.info(
            "  Matriz de checkpoints: "
            "JSON + CSV"
        )

        logger.info(
            "  Métricas: Defensor + Atacante"
        )

        logger.info(
            "  Resumo: OK"
        )

        logger.info(
            "  Gráficos: %d gerados",
            len(plot_paths),
        )

        return {
            "attacker_final": str(
                attacker_final
            ),

            "defender_final": str(
                defender_final
            ),

            "attack_samples": str(
                attack_samples
            ),

            "legacy_ddos_samples": str(
                legacy_samples
            ),

            "history": str(
                history_path
            ),

            "checkpoint_matrix_json": str(
                matrix_json
            ),

            "checkpoint_matrix_csv": str(
                matrix_csv
            ),

            "defender_metrics": str(
                metric_paths[
                    "defender"
                ]
            ),

            "attacker_metrics": str(
                metric_paths[
                    "attacker"
                ]
            ),

            "summary": str(
                summary_path
            ),
        }

    @staticmethod
    def _save_json(
        path: Path,
        payload: dict,
    ) -> None:

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                indent=2,
                ensure_ascii=False,
            )

    @staticmethod
    def _save_matrix_csv(
        evaluation: dict,
        path: Path,
    ) -> None:

        columns = evaluation[
            "colunas_defensores"
        ]

        rows = evaluation[
            "linhas_atacantes"
        ]

        matrix = evaluation[
            "matriz_evasao"
        ]

        with open(
            path,
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow(
                ["Atacante"]
                + [
                    f"D{value}"
                    for value in columns
                ]
            )

            for (
                attacker,
                line,
            ) in zip(
                rows,
                matrix,
            ):
                writer.writerow(
                    [
                        f"A{attacker}"
                    ]
                    + [
                        f"{value * 100:.4f}"
                        for value in line
                    ]
                )

    @staticmethod
    def _log_final_metrics(
        history: dict,
        metrics: dict,
    ) -> None:

        logger.info("")
        logger.info(
            "=== Treinamento concluído ==="
        )

        if history.get(
            "taxa_evasao_pre_adaptacao"
        ):
            logger.info(
                "  Evasão final "
                "pré-adaptação: %.1f%%",

                history[
                    "taxa_evasao_pre_adaptacao"
                ][-1]
                * 100,
            )

        if history.get(
            "taxa_evasao_pos_adaptacao"
        ):
            logger.info(
                "  Evasão final "
                "pós-adaptação: %.1f%%",

                history[
                    "taxa_evasao_pos_adaptacao"
                ][-1]
                * 100,
            )

        logger.info(
            "  Teste reservado: "
            "Acc %.2f%% | "
            "Precision %.2f%% | "
            "Recall %.2f%% | "
            "F1 %.2f%% | "
            "FPR %.2f%% | "
            "FNR %.2f%% | "
            "ROC-AUC %.4f",

            metrics[
                "accuracy"
            ] * 100,

            metrics[
                "precision"
            ] * 100,

            metrics[
                "recall"
            ] * 100,

            metrics[
                "f1"
            ] * 100,

            metrics[
                "fpr"
            ] * 100,

            metrics[
                "fnr"
            ] * 100,

            metrics[
                "roc_auc"
            ],
        )

        tn, fp = metrics[
            "confusion_matrix"
        ][0]

        fn, tp = metrics[
            "confusion_matrix"
        ][1]

        logger.info(
            "  Matriz de confusão final:"
        )

        logger.info(
            "                 "
            "Pred. Benigno   Pred. Ataque"
        )

        logger.info(
            "    Real Benigno "
            "%12d   %10d",
            tn,
            fp,
        )

        logger.info(
            "    Real Ataque  "
            "%12d   %10d",
            fn,
            tp,
        )
