from __future__ import annotations

import json
import logging

from pathlib import Path

import torch

from adarena.control.enforcer import (
    DryRunRuleEnforcer,
)

from adarena.control.observation import (
    ControlledObservationPipeline,
)

from adarena.control.pipeline import (
    ControlPipeline,
)

from adarena.control.policy import (
    ThresholdDecisionPolicy,
)

from adarena.core.config import (
    ExperimentMode,
)

from adarena.network.inference import (
    NetworkInferencePipeline,
    TorchBinaryPredictor,
)

from .base import (
    ExperimentProtocol,
    ProtocolContext,
    ProtocolResult,
)


logger = logging.getLogger(
    __name__
)


class NetworkObservationProtocol(
    ExperimentProtocol
):
    """
    Protocolo de observação passiva da rede.

    Fluxo:

        tráfego já existente
                ↓
            Capture
                ↓
            Extractor
                ↓
         Preprocessor
                ↓
            Defender
                ↓
        DecisionPolicy
                ↓
        DryRunRuleEnforcer

    O protocolo não gera tráfego e não aplica
    alterações reais na rede.
    """

    def run(
        self,
        context: ProtocolContext,
    ) -> ProtocolResult:

        config = context.config

        if (
            config.mode
            != ExperimentMode.OBSERVE
        ):
            raise ValueError(
                "NetworkObservationProtocol "
                "exige mode=OBSERVE"
            )

        network = config.network

        if network is None:
            raise ValueError(
                "NetworkObservationProtocol "
                "exige configuração de rede"
            )

        if not network.dry_run:
            raise ValueError(
                "NetworkObservationProtocol "
                "opera apenas com dry_run=True"
            )

        if network.capture is None:
            raise ValueError(
                "OBSERVE exige "
                "network.capture"
            )

        if network.extractor is None:
            raise ValueError(
                "OBSERVE exige "
                "network.extractor"
            )

        if not config.defender.source:
            raise ValueError(
                "OBSERVE exige checkpoint "
                "em defender.source"
            )

        logger.info("")
        logger.info(
            "[2/4] Configurando "
            "observação de rede"
        )

        defender = self._create_defender(
            context
        )

        predictor = (
            TorchBinaryPredictor(
                defender,
                device=config.device,
            )
        )

        capture = self._create_capture(
            context
        )

        extractor = self._create_extractor(
            context
        )

        threshold = network.threshold(
            config
            .training
            .classification_threshold
        )

        inference_pipeline = (
            NetworkInferencePipeline(
                extractor=extractor,
                preprocessor=(
                    context.preprocessor
                ),
                predictor=predictor,
                threshold=threshold,
            )
        )

        enforcer = DryRunRuleEnforcer()

        control_pipeline = (
            ControlPipeline(
                policy=(
                    ThresholdDecisionPolicy(
                        threshold=threshold,
                    )
                ),
                enforcer=enforcer,
            )
        )

        observation_pipeline = (
            ControlledObservationPipeline(
                inference_pipeline=(
                    inference_pipeline
                ),
                control_pipeline=(
                    control_pipeline
                ),
            )
        )

        logger.info(
            "  Interface de captura: %s",
            getattr(
                capture,
                "iface",
                None,
            ),
        )

        logger.info(
            "  Duração: %.3fs",
            network.capture_duration,
        )

        logger.info(
            "  Threshold: %.4f",
            threshold,
        )

        logger.info("")
        logger.info(
            "[3/4] Observando tráfego"
        )

        capture_batch = capture.capture(
            duration=(
                network.capture_duration
            ),
            packet_limit=(
                network.packet_limit
            ),
        )

        result = (
            observation_pipeline
            .process(
                capture_batch
            )
        )

        inference = result.inference
        control = result.control

        n_flows = inference.n_flows
        attack_count = (
            inference.attack_count
        )
        benign_count = (
            inference.benign_count
        )

        block_count = (
            control.block_count
        )

        applied_count = (
            control.applied_count
        )

        captured_packets = len(
            capture_batch.packets
        )

        attack_rate = (
            float(
                attack_count
                / n_flows
            )
            if n_flows
            else 0.0
        )

        metrics = {
            "captured_packets": (
                captured_packets
            ),

            "reconstructed_flows": (
                n_flows
            ),

            "network_attack_count": (
                attack_count
            ),

            "network_benign_count": (
                benign_count
            ),

            "network_attack_rate": (
                attack_rate
            ),

            "block_decisions": (
                block_count
            ),

            "rules_applied": (
                applied_count
            ),

            "classification_threshold": (
                float(
                    threshold
                )
            ),

            "capture_duration": (
                float(
                    network
                    .capture_duration
                )
            ),

            "packet_limit": (
                network.packet_limit
            ),

            "dry_run": True,
        }

        decisions = []

        for (
            decision,
            enforcement,
        ) in zip(
            control.decisions,
            control.enforcement_results,
            strict=True,
        ):
            decisions.append(
                {
                    "flow_id": (
                        decision.flow_id
                    ),

                    "prediction": int(
                        decision.prediction
                    ),

                    "score": float(
                        decision.score
                    ),

                    "action": (
                        decision
                        .action
                        .value
                    ),

                    "reason": (
                        decision.reason
                    ),

                    "applied": bool(
                        enforcement.applied
                    ),

                    "success": bool(
                        enforcement.success
                    ),

                    "dry_run": bool(
                        enforcement.dry_run
                    ),

                    "message": (
                        enforcement.message
                    ),
                }
            )

        evaluations = {
            "flow_ids": list(
                inference
                .extracted
                .flow_ids
            ),

            "probabilities": (
                inference
                .probabilities
                .astype(
                    float
                )
                .tolist()
            ),

            "predictions": (
                inference
                .predictions
                .astype(
                    int
                )
                .tolist()
            ),

            "decisions": decisions,

            "capture_metadata": dict(
                capture_batch.metadata
            ),
        }

        artifacts = (
            self._save_artifacts(
                context=context,
                metrics=metrics,
                evaluations=(
                    evaluations
                ),
            )
        )

        logger.info(
            "  Pacotes observados: %d",
            captured_packets,
        )

        logger.info(
            "  Fluxos reconstruídos: %d",
            n_flows,
        )

        logger.info(
            "  Classificação: "
            "%d ataque | %d benigno",
            attack_count,
            benign_count,
        )

        logger.info(
            "  Decisões BLOCK: %d",
            block_count,
        )

        logger.info(
            "  Regras aplicadas: %d",
            applied_count,
        )

        logger.info("")
        logger.info(
            "[4/4] Observação concluída"
        )

        return ProtocolResult(
            history={},
            final_metrics=metrics,
            evaluations=evaluations,
            artifacts=artifacts,
        )

    def _create_defender(
        self,
        context: ProtocolContext,
    ):

        config = context.config
        selection = config.defender

        if not selection.source:
            raise ValueError(
                "Defensor não possui "
                "checkpoint em source"
            )

        params = dict(
            selection.params
        )

        params.setdefault(
            "input_dim",
            len(
                context
                .preprocessor
                .feature_names
            ),
        )

        defender = (
            context.registry.create(
                selection.component_id,
                **params,
            )
        )

        checkpoint = torch.load(
            selection.source,
            map_location=config.device,
            weights_only=True,
        )

        if (
            isinstance(
                checkpoint,
                dict,
            )
            and "state_dict"
            in checkpoint
            and isinstance(
                checkpoint["state_dict"],
                dict,
            )
        ):
            checkpoint = (
                checkpoint[
                    "state_dict"
                ]
            )

        defender.load_state_dict(
            checkpoint
        )

        defender = defender.to(
            config.device
        )

        defender.eval()

        return defender

    @staticmethod
    def _create_capture(
        context: ProtocolContext,
    ):

        network = context.config.network

        if (
            network is None
            or network.capture is None
        ):
            raise RuntimeError(
                "Capture não configurado"
            )

        selection = network.capture

        return (
            context.registry.create(
                selection.component_id,
                **selection.params,
            )
        )

    @staticmethod
    def _create_extractor(
        context: ProtocolContext,
    ):

        network = context.config.network

        if (
            network is None
            or network.extractor is None
        ):
            raise RuntimeError(
                "Extractor não configurado"
            )

        selection = network.extractor

        return (
            context.registry.create(
                selection.component_id,
                **selection.params,
            )
        )

    def _save_artifacts(
        self,
        *,
        context: ProtocolContext,
        metrics: dict,
        evaluations: dict,
    ) -> dict[str, str]:

        metrics_path = (
            context.run_dir
            / "metrics"
            / "network_observation.json"
        )

        evaluations_path = (
            context.run_dir
            / "metrics"
            / (
                "network_observation_"
                "evaluations.json"
            )
        )

        self._save_json(
            metrics_path,
            metrics,
        )

        self._save_json(
            evaluations_path,
            evaluations,
        )

        return {
            "network_observation": str(
                metrics_path
            ),

            "network_observation_evaluations": str(
                evaluations_path
            ),
        }

    @staticmethod
    def _save_json(
        path: Path,
        payload: dict,
    ) -> None:

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

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
