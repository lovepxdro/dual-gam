from __future__ import annotations

import json
import logging

from dataclasses import (
    dataclass,
)

from datetime import datetime
from pathlib import Path

from typing import Any

from adarena.protocols.base import (
    ExperimentProtocol,
    ProtocolContext,
    ProtocolResult,
)

from .config import (
    ExperimentConfig,
)

from .registry import (
    ComponentRegistry,
)


logger = logging.getLogger(
    __name__
)


@dataclass(slots=True)
class ExperimentResult:
    run_id: str
    run_dir: Path

    protocol_result: ProtocolResult

    @property
    def history(self) -> dict:
        return (
            self
            .protocol_result
            .history
        )

    @property
    def final_metrics(self) -> dict:
        return (
            self
            .protocol_result
            .final_metrics
        )

    @property
    def artifacts(self) -> dict:
        return (
            self
            .protocol_result
            .artifacts
        )


class ExperimentRunner:
    """
    Orquestrador central da ADArena.

    Responsabilidades:
    - validar a configuração;
    - resolver componentes;
    - carregar o dataset;
    - executar preprocessing;
    - criar a unidade experimental;
    - delegar execução ao protocolo;
    - ativar o experimento concluído.

    O Runner NÃO conhece:
    - arquitetura interna do Atacante;
    - arquitetura interna do Defensor;
    - AdversarialTrainer;
    - CIC-IDS2017;
    - detalhes do protocolo científico.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        registry: ComponentRegistry,
        *,
        preprocessor_factory=None,
    ) -> None:

        self.config = config
        self.registry = registry

        self._preprocessor_factory = (
            preprocessor_factory
            or self
            ._default_preprocessor_factory
        )

    def run(
        self,
    ) -> ExperimentResult:

        self.config.validate(
            self.registry
        )

        if self.config.protocol is None:
            raise ValueError(
                "ExperimentConfig precisa "
                "definir um protocolo"
            )

        models_root = Path(
            self.config.output_dir
        )

        (
            run_id,
            run_dir,
        ) = self._create_run_directory(
            models_root
        )

        log_handler = (
            self._add_run_log(
                run_dir
            )
        )

        try:
            logger.info(
                "=== ADArena | Experimento ==="
            )

            logger.info(
                "  Run: %s",
                run_id,
            )

            logger.info(
                "  Dataset: %s",
                self
                .config
                .attack_dataset
                .component_id,
            )

            logger.info(
                "  Atacante: %s",
                self
                .config
                .attacker
                .component_id,
            )

            logger.info(
                "  Defensor: %s",
                self
                .config
                .defender
                .component_id,
            )

            logger.info(
                "  Protocolo: %s",
                self
                .config
                .protocol
                .component_id,
            )

            logger.info("")
            logger.info(
                "[1/4] Preparando dados"
            )

            dataset_data = (
                self._load_dataset()
            )

            self._validate_dataset(
                dataset_data
            )

            preprocessor = (
                self
                ._preprocessor_factory()
            )

            preprocessor.configurar_dataset(
                dataset_data
            )

            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
            ) = (
                preprocessor
                .split_e_normalizar(
                    dataset_data.X,
                    dataset_data.y,

                    test_size=(
                        self
                        .config
                        .split
                        .test_size
                    ),

                    validation_size=(
                        self
                        .config
                        .split
                        .validation_size
                    ),

                    random_state=(
                        self.config.seed
                    ),
                )
            )

            preprocessor.salvar(
                run_dir
                / "preprocessador"
            )

            snapshot = (
                self._build_snapshot(
                    run_id=run_id,
                    dataset_data=(
                        dataset_data
                    ),
                    input_dim=(
                        X_train.shape[1]
                    ),
                    feature_names=(
                        preprocessor
                        .feature_names
                    ),
                )
            )

            self._save_json(
                run_dir
                / "config_execucao.json",

                snapshot,
            )

            protocol = (
                self._create_protocol()
            )

            context = ProtocolContext(
                config=self.config,

                registry=self.registry,

                run_id=run_id,
                run_dir=run_dir,

                X_train=X_train,
                X_val=X_val,
                X_test=X_test,

                y_train=y_train,
                y_val=y_val,
                y_test=y_test,

                preprocessor=preprocessor,

                dataset_data=(
                    dataset_data
                ),

                config_snapshot=(
                    snapshot
                ),
            )

            protocol_result = (
                protocol.run(
                    context
                )
            )

            self._update_latest(
                models_root,
                run_dir,
            )

            logger.info(
                "  Experimento ativo: %s",
                run_id,
            )

            return ExperimentResult(
                run_id=run_id,
                run_dir=run_dir,
                protocol_result=(
                    protocol_result
                ),
            )

        finally:
            root = logging.getLogger()

            root.removeHandler(
                log_handler
            )

            log_handler.close()

    def _load_dataset(self):
        selection = (
            self
            .config
            .attack_dataset
        )

        if not selection.source:
            raise ValueError(
                "Dataset selecionado "
                "não possui source"
            )

        dataset = (
            self.registry.create(
                selection.component_id,
                **selection.params,
            )
        )

        return dataset.load(
            selection.source
        )

    @staticmethod
    def _validate_dataset(
        dataset_data,
    ) -> None:

        if not (
            dataset_data
            .attack_mask
            .any()
        ):
            raise ValueError(
                "Dataset não possui "
                "amostras de ataque"
            )

        if not (
            dataset_data
            .benign_mask
            .any()
        ):
            raise ValueError(
                "Dataset não possui "
                "amostras benignas"
            )

    def _create_protocol(
        self,
    ) -> ExperimentProtocol:

        selection = (
            self.config.protocol
        )

        if selection is None:
            raise RuntimeError(
                "Protocolo não configurado"
            )

        protocol = (
            self.registry.create(
                selection.component_id,
                **selection.params,
            )
        )

        if not isinstance(
            protocol,
            ExperimentProtocol,
        ):
            raise TypeError(
                f"{selection.component_id} "
                "não implementa "
                "ExperimentProtocol"
            )

        return protocol

    def _build_snapshot(
        self,
        *,
        run_id: str,
        dataset_data,
        input_dim: int,
        feature_names: list[str],
    ) -> dict[str, Any]:

        training = (
            self.config.training
        )

        dataset_selection = (
            self
            .config
            .attack_dataset
        )

        return {
            "run_id": run_id,

            "created_at": (
                datetime.now()
                .isoformat(
                    timespec="seconds"
                )
            ),

            "seed": (
                self.config.seed
            ),

            # Compatibilidade com
            # ferramentas da linha 1.x.
            "dataset": (
                dataset_selection.source
            ),

            "dataset_config": {
                "component_id": (
                    dataset_selection
                    .component_id
                ),

                "source": (
                    dataset_selection
                    .source
                ),

                "params": dict(
                    dataset_selection
                    .params
                ),

                "metadata": (
                    dataset_data
                    .metadata
                ),
            },

            "components": {
                "attacker": (
                    self
                    .config
                    .attacker
                    .component_id
                ),

                "defender": (
                    self
                    .config
                    .defender
                    .component_id
                ),

                "dataset": (
                    dataset_selection
                    .component_id
                ),

                "protocol": (
                    self
                    .config
                    .protocol
                    .component_id
                    if (
                        self
                        .config
                        .protocol
                        is not None
                    )
                    else None
                ),
            },

            "dados": {
                "input_dim": (
                    input_dim
                ),

                "noise_dim": (
                    training.noise_dim
                ),

                "validation_size": (
                    self
                    .config
                    .split
                    .validation_size
                ),

                "test_size": (
                    self
                    .config
                    .split
                    .test_size
                ),

                "feature_names": (
                    feature_names
                ),
            },

            "treinamento": {
                "lr_defensor": (
                    training.lr_defensor
                ),

                "lr_atacante": (
                    training.lr_atacante
                ),

                "adam_betas_atacante": (
                    list(
                        training
                        .adam_betas
                    )
                ),

                "epsilon": (
                    training.epsilon
                ),

                "classification_threshold": (
                    training
                    .classification_threshold
                ),

                "amostras_avaliacao_adversarial": (
                    training
                    .amostras_avaliacao_adversarial
                ),

                "epochs_pretrain": (
                    training
                    .epochs_pretrain
                ),

                "epochs_por_rodada": (
                    training
                    .epochs_por_rodada
                ),

                "n_rodadas": (
                    training.n_rodadas
                ),

                "amostras_por_rodada": (
                    training
                    .amostras_por_rodada
                ),

                "batch_size": (
                    training.batch_size
                ),

                "device": (
                    self.config.device
                ),
            },

            "metadata": dict(
                self.config.metadata
            ),
        }

    def _create_run_directory(
        self,
        models_root: Path,
    ) -> tuple[
        str,
        Path,
    ]:

        experiments = (
            models_root
            / "experiments"
        )

        experiments.mkdir(
            parents=True,
            exist_ok=True,
        )

        base = (
            datetime.now()
            .strftime(
                "run_%Y%m%d_%H%M%S"
            )
            + f"_seed{self.config.seed}"
        )

        run_id = base

        run_dir = (
            experiments
            / run_id
        )

        suffix = 1

        while run_dir.exists():
            run_id = (
                f"{base}_{suffix:02d}"
            )

            run_dir = (
                experiments
                / run_id
            )

            suffix += 1

        for subdir in (
            "checkpoints",
            "metrics",
            "plots",
            "logs",
        ):
            (
                run_dir
                / subdir
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

        return (
            run_id,
            run_dir,
        )

    @staticmethod
    def _update_latest(
        models_root: Path,
        run_dir: Path,
    ) -> None:

        latest = (
            models_root
            / "latest"
        )

        if (
            latest.is_symlink()
            or latest.exists()
        ):
            if (
                latest.is_dir()
                and not latest.is_symlink()
            ):
                raise RuntimeError(
                    f"{latest} existe "
                    "e não é symlink"
                )

            latest.unlink()

        latest.symlink_to(
            run_dir.relative_to(
                models_root
            ),
            target_is_directory=True,
        )

    @staticmethod
    def _add_run_log(
        run_dir: Path,
    ) -> logging.FileHandler:

        handler = logging.FileHandler(
            run_dir
            / "logs"
            / "train.log",

            encoding="utf-8",
        )

        handler.setLevel(
            logging.DEBUG
        )

        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s "
                "[%(levelname)s] "
                "%(name)s — %(message)s",

                datefmt=(
                    "%Y-%m-%d %H:%M:%S"
                ),
            )
        )

        logging.getLogger().addHandler(
            handler
        )

        return handler

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
    def _default_preprocessor_factory():
        # Import tardio enquanto o
        # preprocessador ainda reside
        # no pacote legado gan/.
        from gan.preprocessing import (
            Preprocessador,
        )

        return Preprocessador()
