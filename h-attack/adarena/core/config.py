from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
)

from enum import Enum

from typing import Any

from .components import (
    ComponentKind,
)

from .registry import (
    ComponentRegistry,
)


class ExperimentMode(
    str,
    Enum,
):
    TRAIN = "train"

    SIMULATE = "simulate"

    TRAIN_AND_SIMULATE = (
        "train_and_simulate"
    )


@dataclass(slots=True)
class ComponentSelection:
    component_id: str

    params: dict[str, Any] = field(
        default_factory=dict
    )

    source: str | None = None

    def to_dict(
        self,
    ) -> dict[str, Any]:

        return {
            "component_id": (
                self.component_id
            ),

            "params": dict(
                self.params
            ),

            "source": (
                self.source
            ),
        }


@dataclass(slots=True)
class DataSplitConfig:
    test_size: float = 0.2

    validation_size: float = 0.1

    def validate(
        self,
    ) -> None:

        if self.test_size <= 0:
            raise ValueError(
                "test_size deve ser > 0"
            )

        if (
            self.validation_size
            <= 0
        ):
            raise ValueError(
                "validation_size deve ser > 0"
            )

        if (
            self.test_size
            + self.validation_size
            >= 1
        ):
            raise ValueError(
                "test_size + validation_size "
                "deve ser menor que 1"
            )


@dataclass(slots=True)
class TrainingSettings:
    """
    Configuração do protocolo adversarial atual.

    Estes valores pertencem ao protocolo de
    treinamento, não ao modelo em si.
    """

    noise_dim: int = 32

    lr_defensor: float = 1e-3

    lr_atacante: float = 2e-4

    adam_betas: tuple[
        float,
        float,
    ] = (
        0.5,
        0.999,
    )

    epsilon: float = 0.3

    classification_threshold: (
        float
    ) = 0.5

    epochs_pretrain: int = 5

    epochs_por_rodada: int = 3

    n_rodadas: int = 20

    amostras_por_rodada: int = 5000

    amostras_avaliacao_adversarial: (
        int
    ) = 5000

    batch_size: int = 512

    def validate(
        self,
    ) -> None:

        if self.noise_dim <= 0:
            raise ValueError(
                "noise_dim deve ser > 0"
            )

        if self.n_rodadas <= 0:
            raise ValueError(
                "n_rodadas deve ser > 0"
            )

        if (
            self.epochs_por_rodada
            <= 0
        ):
            raise ValueError(
                "epochs_por_rodada "
                "deve ser > 0"
            )

        if self.epochs_pretrain <= 0:
            raise ValueError(
                "epochs_pretrain deve ser > 0"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size deve ser > 0"
            )

        if not (
            0.0
            <= self
            .classification_threshold
            <= 1.0
        ):
            raise ValueError(
                "classification_threshold "
                "deve estar entre 0 e 1"
            )


@dataclass(slots=True)
class NetworkSettings:
    """
    Configuração declarativa da etapa de rede.

    Os componentes concretos continuam sendo
    selecionados pelo Registry.

    Parâmetros específicos de implementação,
    como interface, endereço do alvo ou porta,
    pertencem a ComponentSelection.params.
    """

    renderer: (
        ComponentSelection
        | None
    ) = None

    network_backend: (
        ComponentSelection
        | None
    ) = None

    capture: (
        ComponentSelection
        | None
    ) = None

    extractor: (
        ComponentSelection
        | None
    ) = None

    sample_count: int = 20

    packet_limit: (
        int
        | None
    ) = None

    classification_threshold: (
        float
        | None
    ) = None

    dry_run: bool = True

    def validate(
        self,
        *,
        require_components: bool,
    ) -> None:

        if self.sample_count <= 0:
            raise ValueError(
                "network.sample_count "
                "deve ser > 0"
            )

        if (
            self.packet_limit
            is not None
            and self.packet_limit <= 0
        ):
            raise ValueError(
                "network.packet_limit "
                "deve ser > 0"
            )

        if (
            self.classification_threshold
            is not None
            and not (
                0.0
                <= self
                .classification_threshold
                <= 1.0
            )
        ):
            raise ValueError(
                "network."
                "classification_threshold "
                "deve estar entre 0 e 1"
            )

        if not require_components:
            return

        required = {
            "renderer": (
                self.renderer
            ),

            "network_backend": (
                self.network_backend
            ),

            "capture": (
                self.capture
            ),

            "extractor": (
                self.extractor
            ),
        }

        missing = [
            name
            for (
                name,
                selection,
            )
            in required.items()
            if selection is None
        ]

        if missing:
            raise ValueError(
                "Configuração de rede "
                "incompleta. Componentes "
                "ausentes: "
                + ", ".join(
                    missing
                )
            )

    def threshold(
        self,
        fallback: float,
    ) -> float:

        if (
            self.classification_threshold
            is None
        ):
            return float(
                fallback
            )

        return float(
            self.classification_threshold
        )


@dataclass(slots=True)
class ExperimentConfig:
    """
    Descrição declarativa de um experimento
    ADArena.
    """

    attacker: ComponentSelection

    defender: ComponentSelection

    attack_dataset: ComponentSelection

    benign_dataset: (
        ComponentSelection
        | None
    ) = None

    protocol: (
        ComponentSelection
        | None
    ) = None

    mode: ExperimentMode = (
        ExperimentMode.TRAIN
    )

    seed: int = 42

    device: str = "cpu"

    output_dir: str = "/models"

    split: DataSplitConfig = field(
        default_factory=(
            DataSplitConfig
        )
    )

    training: TrainingSettings = field(
        default_factory=(
            TrainingSettings
        )
    )

    network: (
        NetworkSettings
        | None
    ) = None

    metadata: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    def validate(
        self,
        registry: ComponentRegistry,
    ) -> None:

        # ---------------------------------
        # Componentes centrais
        # ---------------------------------

        self._require_kind(
            registry,
            self.attacker,
            ComponentKind.ATTACKER,
        )

        self._require_kind(
            registry,
            self.defender,
            ComponentKind.DEFENDER,
        )

        self._require_kind(
            registry,
            self.attack_dataset,
            ComponentKind.DATASET,
        )

        if (
            self.benign_dataset
            is not None
        ):
            self._require_kind(
                registry,
                self.benign_dataset,
                ComponentKind.DATASET,
            )

        if self.protocol is not None:
            self._require_kind(
                registry,
                self.protocol,
                ComponentKind
                .EXPERIMENT_PROTOCOL,
            )

        attacker_spec = (
            registry.spec(
                self
                .attacker
                .component_id
            )
        )

        defender_spec = (
            registry.spec(
                self
                .defender
                .component_id
            )
        )

        dataset_spec = (
            registry.spec(
                self
                .attack_dataset
                .component_id
            )
        )

        # ---------------------------------
        # Dataset -> atacante
        # ---------------------------------

        self._require_representation_match(
            producer_name=(
                "dataset de ataque"
            ),

            producer_output=(
                dataset_spec
                .output_representation
            ),

            consumer_name=(
                "atacante"
            ),

            consumer_input=(
                attacker_spec
                .input_representation
            ),
        )

        # ---------------------------------
        # Caminho de treinamento
        # ---------------------------------

        trains = (
            self.mode
            in {
                ExperimentMode.TRAIN,
                ExperimentMode
                .TRAIN_AND_SIMULATE,
            }
        )

        simulates = (
            self.mode
            in {
                ExperimentMode.SIMULATE,
                ExperimentMode
                .TRAIN_AND_SIMULATE,
            }
        )

        if trains:
            self._require_representation_match(
                producer_name=(
                    "atacante"
                ),

                producer_output=(
                    attacker_spec
                    .output_representation
                ),

                consumer_name=(
                    "defensor"
                ),

                consumer_input=(
                    defender_spec
                    .input_representation
                ),
            )

            if (
                self.benign_dataset
                is not None
            ):
                benign_spec = (
                    registry.spec(
                        self
                        .benign_dataset
                        .component_id
                    )
                )

                self._require_representation_match(
                    producer_name=(
                        "dataset benigno"
                    ),

                    producer_output=(
                        benign_spec
                        .output_representation
                    ),

                    consumer_name=(
                        "defensor"
                    ),

                    consumer_input=(
                        defender_spec
                        .input_representation
                    ),
                )

        # ---------------------------------
        # Caminho de rede
        # ---------------------------------

        if self.network is not None:
            self.network.validate(
                require_components=(
                    simulates
                )
            )

        elif simulates:
            raise ValueError(
                "ExperimentMode de simulação "
                "exige network configurado"
            )

        if simulates:
            self._validate_network_components(
                registry=registry,

                attacker_spec=(
                    attacker_spec
                ),

                defender_spec=(
                    defender_spec
                ),
            )

        # ---------------------------------
        # Configuração geral
        # ---------------------------------

        if self.seed < 0:
            raise ValueError(
                "seed deve ser >= 0"
            )

        if self.device not in {
            "cpu",
            "cuda",
        }:
            raise ValueError(
                "device deve ser "
                "'cpu' ou 'cuda'"
            )

        if not self.output_dir.strip():
            raise ValueError(
                "output_dir não pode "
                "ser vazio"
            )

        self.split.validate()

        self.training.validate()

    def _validate_network_components(
        self,
        *,
        registry: ComponentRegistry,
        attacker_spec,
        defender_spec,
    ) -> None:

        network = self.network

        if network is None:
            raise RuntimeError(
                "Configuração de rede "
                "não disponível"
            )

        # NetworkSettings.validate()
        # garantiu estes valores.
        assert (
            network.renderer
            is not None
        )

        assert (
            network.network_backend
            is not None
        )

        assert (
            network.capture
            is not None
        )

        assert (
            network.extractor
            is not None
        )

        self._require_kind(
            registry,
            network.renderer,
            ComponentKind.RENDERER,
        )

        self._require_kind(
            registry,
            network.network_backend,
            ComponentKind
            .NETWORK_BACKEND,
        )

        self._require_kind(
            registry,
            network.capture,
            ComponentKind.CAPTURE,
        )

        self._require_kind(
            registry,
            network.extractor,
            ComponentKind.EXTRACTOR,
        )

        renderer_spec = (
            registry.spec(
                network
                .renderer
                .component_id
            )
        )

        backend_spec = (
            registry.spec(
                network
                .network_backend
                .component_id
            )
        )

        capture_spec = (
            registry.spec(
                network
                .capture
                .component_id
            )
        )

        extractor_spec = (
            registry.spec(
                network
                .extractor
                .component_id
            )
        )

        # atacante -> renderer
        self._require_representation_match(
            producer_name=(
                "atacante"
            ),

            producer_output=(
                attacker_spec
                .output_representation
            ),

            consumer_name=(
                "renderer"
            ),

            consumer_input=(
                renderer_spec
                .input_representation
            ),
        )

        # renderer -> backend
        self._require_representation_match(
            producer_name=(
                "renderer"
            ),

            producer_output=(
                renderer_spec
                .output_representation
            ),

            consumer_name=(
                "network backend"
            ),

            consumer_input=(
                backend_spec
                .input_representation
            ),
        )

        # Capture é um observador da rede.
        #
        # Não existe:
        #
        #   backend.output -> capture.input
        #
        # porque o backend não entrega seus
        # resultados diretamente ao Capture.
        # O Capture observa os pacotes que
        # ocorreram no ambiente.

        # capture -> extractor
        self._require_representation_match(
            producer_name=(
                "capture"
            ),

            producer_output=(
                capture_spec
                .output_representation
            ),

            consumer_name=(
                "extractor"
            ),

            consumer_input=(
                extractor_spec
                .input_representation
            ),
        )

        # extractor -> defensor
        self._require_representation_match(
            producer_name=(
                "extractor"
            ),

            producer_output=(
                extractor_spec
                .output_representation
            ),

            consumer_name=(
                "defensor"
            ),

            consumer_input=(
                defender_spec
                .input_representation
            ),
        )

    @staticmethod
    def _require_kind(
        registry: ComponentRegistry,
        selection: ComponentSelection,
        expected_kind: ComponentKind,
    ) -> None:

        spec = registry.spec(
            selection.component_id
        )

        if spec.kind != expected_kind:
            raise ValueError(
                f"{selection.component_id} é "
                f"{spec.kind.value}, mas o "
                f"experimento exige "
                f"{expected_kind.value}"
            )

    @staticmethod
    def _require_representation_match(
        *,
        producer_name: str,
        producer_output,
        consumer_name: str,
        consumer_input,
    ) -> None:

        if (
            producer_output is None
            or consumer_input is None
        ):
            return

        if (
            producer_output
            != consumer_input
        ):
            raise ValueError(
                "Representações incompatíveis: "
                f"{producer_name} produz "
                f"'{producer_output.value}', "
                f"mas {consumer_name} exige "
                f"'{consumer_input.value}'"
            )

    def to_dict(
        self,
    ) -> dict[str, Any]:

        payload = asdict(
            self
        )

        payload[
            "mode"
        ] = self.mode.value

        return payload
