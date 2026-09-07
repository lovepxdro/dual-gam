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


class ExperimentMode(str, Enum):
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": (
                self.component_id
            ),
            "params": dict(
                self.params
            ),
            "source": self.source,
        }


@dataclass(slots=True)
class DataSplitConfig:
    test_size: float = 0.2
    validation_size: float = 0.1

    def validate(self) -> None:
        if self.test_size <= 0:
            raise ValueError(
                "test_size deve ser > 0"
            )

        if self.validation_size <= 0:
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

    classification_threshold: float = 0.5

    epochs_pretrain: int = 5

    epochs_por_rodada: int = 3
    n_rodadas: int = 20

    amostras_por_rodada: int = 5000

    amostras_avaliacao_adversarial: int = (
        5000
    )

    batch_size: int = 512

    def validate(self) -> None:
        if self.noise_dim <= 0:
            raise ValueError(
                "noise_dim deve ser > 0"
            )

        if self.n_rodadas <= 0:
            raise ValueError(
                "n_rodadas deve ser > 0"
            )

        if self.epochs_por_rodada <= 0:
            raise ValueError(
                "epochs_por_rodada deve ser > 0"
            )

        if self.epochs_pretrain <= 0:
            raise ValueError(
                "epochs_pretrain deve ser > 0"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size deve ser > 0"
            )


@dataclass(slots=True)
class ExperimentConfig:
    """
    Descrição declarativa de um experimento ADArena.
    """

    attacker: ComponentSelection
    defender: ComponentSelection
    attack_dataset: ComponentSelection

    benign_dataset: (
        ComponentSelection | None
    ) = None

    protocol: (
        ComponentSelection | None
    ) = None

    mode: ExperimentMode = (
        ExperimentMode.TRAIN
    )

    seed: int = 42

    device: str = "cpu"

    output_dir: str = "/models"

    split: DataSplitConfig = field(
        default_factory=DataSplitConfig
    )

    training: TrainingSettings = field(
        default_factory=TrainingSettings
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def validate(
        self,
        registry: ComponentRegistry,
    ) -> None:

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

        if self.benign_dataset is not None:
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

        attacker_spec = registry.spec(
            self.attacker.component_id
        )

        defender_spec = registry.spec(
            self.defender.component_id
        )

        dataset_spec = registry.spec(
            self.attack_dataset.component_id
        )

        self._require_representation_match(
            producer_name=(
                "dataset de ataque"
            ),
            producer_output=(
                dataset_spec
                .output_representation
            ),
            consumer_name="atacante",
            consumer_input=(
                attacker_spec
                .input_representation
            ),
        )

        self._require_representation_match(
            producer_name="atacante",
            producer_output=(
                attacker_spec
                .output_representation
            ),
            consumer_name="defensor",
            consumer_input=(
                defender_spec
                .input_representation
            ),
        )

        if (
            self.benign_dataset
            is not None
        ):
            benign_spec = registry.spec(
                self
                .benign_dataset
                .component_id
            )

            self._require_representation_match(
                producer_name=(
                    "dataset benigno"
                ),
                producer_output=(
                    benign_spec
                    .output_representation
                ),
                consumer_name="defensor",
                consumer_input=(
                    defender_spec
                    .input_representation
                ),
            )

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
                "output_dir não pode ser vazio"
            )

        self.split.validate()
        self.training.validate()

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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(
            self
        )

        payload[
            "mode"
        ] = self.mode.value

        return payload
