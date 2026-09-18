from __future__ import annotations

import json
import logging

from pathlib import Path

import numpy as np
import torch

from adarena.core.config import (
    ExperimentMode,
)

from adarena.network.inference import (
    NetworkInferencePipeline,
    NetworkObservationPipeline,
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


class NetworkSimulationProtocol(
    ExperimentProtocol
):
    """
    Protocolo de simulação da camada de rede.

    O protocolo opera exclusivamente em dry-run.

    Fluxo básico:

        amostras DDoS do teste
                ↓
            Atacante
                ↓
        features adversariais
                ↓
            Defensor
                ↓
        evasões matemáticas
                ↓
            Renderer
                ↓
        payloads traduzíveis
                ↓
        NetworkBackend dry-run

    Quando ``network.observe=True``:

        payloads
            ↓
        NetworkObservationPipeline
            ├── Backend
            ├── Capture
            ├── Extractor
            └── Defensor
            ↓
        classificação do fluxo reconstruído

    A opção de observação permite validar o
    caminho completo com componentes sintéticos
    ou fakes sem habilitar transmissão real.
    """

    def run(
        self,
        context: ProtocolContext,
    ) -> ProtocolResult:

        config = context.config

        if (
            config.mode
            != ExperimentMode.SIMULATE
        ):
            raise ValueError(
                "NetworkSimulationProtocol "
                "exige mode=SIMULATE"
            )

        network = config.network

        if network is None:
            raise ValueError(
                "NetworkSimulationProtocol "
                "exige configuração de rede"
            )

        if not network.dry_run:
            raise ValueError(
                "NetworkSimulationProtocol "
                "opera apenas em dry-run"
            )

        logger.info("")
        logger.info(
            "[2/4] Configurando "
            "simulação de rede"
        )

        attacker = (
            self._create_attacker(
                context
            )
        )

        defender = (
            self._create_defender(
                context
            )
        )

        predictor = (
            TorchBinaryPredictor(
                defender,

                device=(
                    config.device
                ),
            )
        )

        renderer = (
            self._create_renderer(
                context
            )
        )

        backend = (
            self._create_backend(
                context
            )
        )

        (
            selected_indices,
            attack_samples,
        ) = self._select_attack_samples(
            context
        )

        logger.info(
            "  Amostras selecionadas: %d",
            len(
                attack_samples
            ),
        )

        adversarial_samples = (
            self._generate_adversarial_samples(
                context=context,
                attacker=attacker,
                samples=attack_samples,
            )
        )

        threshold = (
            network.threshold(
                config
                .training
                .classification_threshold
            )
        )

        probabilities = (
            predictor.predict_proba(
                adversarial_samples
            )
        )

        predictions = (
            probabilities
            >= threshold
        ).astype(
            np.int64
        )

        evasion_mask = (
            predictions == 0
        )

        evasion_count = int(
            evasion_mask.sum()
        )

        sample_count = int(
            len(
                adversarial_samples
            )
        )

        evasion_rate = (
            float(
                evasion_count
                / sample_count
            )
            if sample_count
            else 0.0
        )

        logger.info(
            "  Evasões matemáticas: "
            "%d/%d (%.2f%%)",
            evasion_count,
            sample_count,
            evasion_rate * 100.0,
        )

        evasive_samples = (
            adversarial_samples[
                evasion_mask
            ]
        )

        evasive_probabilities = (
            probabilities[
                evasion_mask
            ]
        )

        evasion_scores = (
            1.0
            - evasive_probabilities
        )

        payloads = (
            renderer.render_batch(
                evasive_samples,

                scores=(
                    evasion_scores
                    .astype(
                        float
                    )
                    .tolist()
                ),

                only_valid=True,
            )
        )

        rendered_count = int(
            len(
                payloads
            )
        )

        logger.info(
            "  Traduções válidas: "
            "%d/%d",
            rendered_count,
            evasion_count,
        )

        observation = None

        if network.observe:
            logger.info(
                "  Observação pós-backend: "
                "habilitada"
            )

            capture = (
                self._create_capture(
                    context
                )
            )

            extractor = (
                self._create_extractor(
                    context
                )
            )

            inference = (
                NetworkInferencePipeline(
                    extractor=extractor,

                    preprocessor=(
                        context.preprocessor
                    ),

                    predictor=predictor,

                    threshold=(
                        threshold
                    ),
                )
            )

            observation_pipeline = (
                NetworkObservationPipeline(
                    capture=capture,
                    backend=backend,
                    inference=inference,
                )
            )

            observation = (
                observation_pipeline
                .execute_many(
                    payloads,

                    packet_limit=(
                        network
                        .packet_limit
                    ),
                )
            )

            backend_results = (
                observation
                .backend_results
            )

        else:
            backend_results = (
                backend.execute_many(
                    payloads
                )
            )

        backend_result_count = int(
            len(
                backend_results
            )
        )

        logger.info(
            "  Execuções dry-run: %d",
            backend_result_count,
        )

        metrics = {
            "samples_selected": (
                sample_count
            ),

            "mathematical_evasions": (
                evasion_count
            ),

            "mathematical_evasion_rate": (
                evasion_rate
            ),

            "valid_renderings": (
                rendered_count
            ),

            "valid_rendering_rate": (
                float(
                    rendered_count
                    / sample_count
                )
                if sample_count
                else 0.0
            ),

            "valid_rendering_rate_given_evasion": (
                float(
                    rendered_count
                    / evasion_count
                )
                if evasion_count
                else 0.0
            ),

            "dry_run_executions": (
                backend_result_count
            ),

            "classification_threshold": (
                float(
                    threshold
                )
            ),

            "epsilon": (
                float(
                    config
                    .training
                    .epsilon
                )
            ),

            "dry_run": True,

            "observation_enabled": (
                bool(
                    network.observe
                )
            ),
        }

        evaluations = {
            "selected_indices": (
                selected_indices
                .astype(
                    int
                )
                .tolist()
            ),

            "ddos_probabilities": (
                probabilities
                .astype(
                    float
                )
                .tolist()
            ),

            "predictions": (
                predictions
                .astype(
                    int
                )
                .tolist()
            ),

            "evasion_mask": (
                evasion_mask
                .astype(
                    bool
                )
                .tolist()
            ),
        }

        if observation is not None:
            network_inference = (
                observation.inference
            )

            n_flows = (
                network_inference
                .n_flows
            )

            attack_count = (
                network_inference
                .attack_count
            )

            benign_count = (
                network_inference
                .benign_count
            )

            metrics.update(
                {
                    "captured_packets": (
                        observation
                        .n_captured_packets
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
                        float(
                            attack_count
                            / n_flows
                        )
                        if n_flows
                        else 0.0
                    ),
                }
            )

            evaluations.update(
                {
                    "network_probabilities": (
                        network_inference
                        .probabilities
                        .astype(
                            float
                        )
                        .tolist()
                    ),

                    "network_predictions": (
                        network_inference
                        .predictions
                        .astype(
                            int
                        )
                        .tolist()
                    ),

                    "network_flow_ids": (
                        list(
                            network_inference
                            .extracted
                            .flow_ids
                        )
                    ),
                }
            )

            logger.info(
                "  Pacotes observados: %d",
                observation
                .n_captured_packets,
            )

            logger.info(
                "  Fluxos reconstruídos: %d",
                n_flows,
            )

            logger.info(
                "  Classificação na rede: "
                "%d ataque | %d benigno",
                attack_count,
                benign_count,
            )

        artifacts = (
            self._save_artifacts(
                context=context,

                attack_samples=(
                    attack_samples
                ),

                adversarial_samples=(
                    adversarial_samples
                ),

                metrics=metrics,

                evaluations=(
                    evaluations
                ),
            )
        )

        logger.info("")
        logger.info(
            "[4/4] Simulação dry-run "
            "concluída"
        )

        return ProtocolResult(
            history={},

            final_metrics=metrics,

            evaluations=(
                evaluations
            ),

            artifacts=artifacts,
        )

    def _create_attacker(
        self,
        context: ProtocolContext,
    ):

        config = context.config

        selection = (
            config.attacker
        )

        if not selection.source:
            raise ValueError(
                "Atacante não possui "
                "checkpoint em source"
            )

        params = dict(
            selection.params
        )

        params.setdefault(
            "noise_dim",
            config
            .training
            .noise_dim,
        )

        params.setdefault(
            "output_dim",
            context
            .X_test
            .shape[1],
        )

        attacker = (
            context.registry.create(
                selection.component_id,
                **params,
            )
        )

        self._load_checkpoint(
            model=attacker,

            source=(
                selection.source
            ),

            device=(
                config.device
            ),
        )

        attacker = attacker.to(
            config.device
        )

        attacker.eval()

        return attacker

    def _create_defender(
        self,
        context: ProtocolContext,
    ):

        config = context.config

        selection = (
            config.defender
        )

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
            context
            .X_test
            .shape[1],
        )

        defender = (
            context.registry.create(
                selection.component_id,
                **params,
            )
        )

        self._load_checkpoint(
            model=defender,

            source=(
                selection.source
            ),

            device=(
                config.device
            ),
        )

        defender = defender.to(
            config.device
        )

        defender.eval()

        return defender

    def _create_renderer(
        self,
        context: ProtocolContext,
    ):

        network = (
            context
            .config
            .network
        )

        if (
            network is None
            or network.renderer is None
        ):
            raise RuntimeError(
                "Renderer não configurado"
            )

        selection = (
            network.renderer
        )

        params = dict(
            selection.params
        )

        params[
            "preprocessor"
        ] = context.preprocessor

        return (
            context.registry.create(
                selection.component_id,
                **params,
            )
        )

    def _create_backend(
        self,
        context: ProtocolContext,
    ):

        network = (
            context
            .config
            .network
        )

        if (
            network is None
            or network
            .network_backend
            is None
        ):
            raise RuntimeError(
                "Network backend "
                "não configurado"
            )

        selection = (
            network
            .network_backend
        )

        params = dict(
            selection.params
        )

        # O protocolo força dry-run.
        params[
            "dry_run"
        ] = True

        return (
            context.registry.create(
                selection.component_id,
                **params,
            )
        )

    def _create_capture(
        self,
        context: ProtocolContext,
    ):

        network = (
            context
            .config
            .network
        )

        if (
            network is None
            or network.capture is None
        ):
            raise RuntimeError(
                "Capture não configurado"
            )

        selection = (
            network.capture
        )

        return (
            context.registry.create(
                selection.component_id,
                **selection.params,
            )
        )

    def _create_extractor(
        self,
        context: ProtocolContext,
    ):

        network = (
            context
            .config
            .network
        )

        if (
            network is None
            or network.extractor is None
        ):
            raise RuntimeError(
                "Extractor não configurado"
            )

        selection = (
            network.extractor
        )

        return (
            context.registry.create(
                selection.component_id,
                **selection.params,
            )
        )

    @staticmethod
    def _load_checkpoint(
        *,
        model,
        source: str,
        device: str,
    ) -> None:

        checkpoint_path = Path(
            source
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Checkpoint não encontrado: "
                f"{checkpoint_path}"
            )

        payload = torch.load(
            checkpoint_path,
            map_location=device,
        )

        state_dict = payload

        if isinstance(
            payload,
            dict,
        ):
            if (
                "state_dict"
                in payload
            ):
                state_dict = (
                    payload[
                        "state_dict"
                    ]
                )

            elif (
                "model_state_dict"
                in payload
            ):
                state_dict = (
                    payload[
                        "model_state_dict"
                    ]
                )

        model.load_state_dict(
            state_dict
        )

    @staticmethod
    def _set_seed(
        seed: int,
        device: str,
    ) -> None:

        np.random.seed(
            seed
        )

        torch.manual_seed(
            seed
        )

        if (
            device == "cuda"
            and torch.cuda
            .is_available()
        ):
            torch.cuda.manual_seed_all(
                seed
            )

    def _select_attack_samples(
        self,
        context: ProtocolContext,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:

        network = (
            context
            .config
            .network
        )

        if network is None:
            raise RuntimeError(
                "Configuração de rede "
                "não disponível"
            )

        attack_indices = (
            np.flatnonzero(
                context.y_test
                == 1
            )
        )

        if (
            len(
                attack_indices
            )
            == 0
        ):
            raise ValueError(
                "Conjunto de teste "
                "não possui amostras "
                "de ataque"
            )

        n_samples = min(
            network.sample_count,
            len(
                attack_indices
            ),
        )

        rng = (
            np.random
            .default_rng(
                context
                .config
                .seed
            )
        )

        selected_indices = (
            rng.choice(
                attack_indices,

                size=n_samples,

                replace=False,
            )
        )

        attack_samples = (
            context.X_test[
                selected_indices
            ]
            .astype(
                np.float32,
                copy=True,
            )
        )

        return (
            selected_indices,
            attack_samples,
        )

    def _generate_adversarial_samples(
        self,
        *,
        context: ProtocolContext,
        attacker,
        samples: np.ndarray,
    ) -> np.ndarray:

        config = context.config

        self._set_seed(
            config.seed,
            config.device,
        )

        tensor = torch.as_tensor(
            samples,

            dtype=torch.float32,

            device=config.device,
        )

        with torch.no_grad():
            adversarial = (
                attacker.perturba(
                    tensor,

                    epsilon=(
                        config
                        .training
                        .epsilon
                    ),

                    device=(
                        config.device
                    ),
                )
            )

        result = (
            adversarial
            .detach()
            .cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        if (
            result.shape
            != samples.shape
        ):
            raise RuntimeError(
                "Atacante retornou shape "
                "incompatível: "
                f"{result.shape} != "
                f"{samples.shape}"
            )

        if not np.isfinite(
            result
        ).all():
            raise RuntimeError(
                "Atacante produziu "
                "valores não finitos"
            )

        return result

    def _save_artifacts(
        self,
        *,
        context: ProtocolContext,
        attack_samples: np.ndarray,
        adversarial_samples: np.ndarray,
        metrics: dict,
        evaluations: dict,
    ) -> dict[str, str]:

        artifacts_dir = (
            context.run_dir
            / "network"
        )

        artifacts_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        original_path = (
            artifacts_dir
            / "selected_attack_samples.npy"
        )

        adversarial_path = (
            artifacts_dir
            / "adversarial_samples.npy"
        )

        summary_path = (
            artifacts_dir
            / "simulation_summary.json"
        )

        np.save(
            original_path,
            attack_samples,
        )

        np.save(
            adversarial_path,
            adversarial_samples,
        )

        summary = {
            "run_id": (
                context.run_id
            ),

            "mode": (
                context
                .config
                .mode
                .value
            ),

            "attacker_checkpoint": (
                context
                .config
                .attacker
                .source
            ),

            "defender_checkpoint": (
                context
                .config
                .defender
                .source
            ),

            "metrics": (
                metrics
            ),

            "evaluations": (
                evaluations
            ),
        }

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return {
            "selected_attack_samples": str(
                original_path
            ),

            "adversarial_samples": str(
                adversarial_path
            ),

            "simulation_summary": str(
                summary_path
            ),
        }
