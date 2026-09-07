from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from .components import ComponentKind
from .registry import ComponentRegistry


class ExperimentMode(str, Enum):
    TRAIN = "train"
    SIMULATE = "simulate"
    TRAIN_AND_SIMULATE = "train_and_simulate"


@dataclass(slots=True)
class ComponentSelection:
    component_id: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "params": dict(self.params),
        }


@dataclass(slots=True)
class ExperimentConfig:
    """Descrição declarativa de um experimento ADArena."""

    attacker: ComponentSelection
    defender: ComponentSelection
    attack_dataset: ComponentSelection

    benign_dataset: ComponentSelection | None = None

    mode: ExperimentMode = ExperimentMode.TRAIN_AND_SIMULATE

    seed: int = 42
    device: str = "cpu"

    metadata: dict[str, Any] = field(default_factory=dict)

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

        attacker_spec = registry.spec(
            self.attacker.component_id
        )

        defender_spec = registry.spec(
            self.defender.component_id
        )

        attack_dataset_spec = registry.spec(
            self.attack_dataset.component_id
        )

        self._require_representation_match(
            producer_name="dataset de ataque",
            producer_output=(
                attack_dataset_spec.output_representation
            ),
            consumer_name="atacante",
            consumer_input=(
                attacker_spec.input_representation
            ),
        )

        self._require_representation_match(
            producer_name="atacante",
            producer_output=(
                attacker_spec.output_representation
            ),
            consumer_name="defensor",
            consumer_input=(
                defender_spec.input_representation
            ),
        )

        if self.benign_dataset is not None:
            benign_spec = registry.spec(
                self.benign_dataset.component_id
            )

            self._require_representation_match(
                producer_name="dataset benigno",
                producer_output=(
                    benign_spec.output_representation
                ),
                consumer_name="defensor",
                consumer_input=(
                    defender_spec.input_representation
                ),
            )

        if self.seed < 0:
            raise ValueError(
                "seed deve ser >= 0"
            )

        if self.device not in {"cpu", "cuda"}:
            raise ValueError(
                "device deve ser 'cpu' ou 'cuda'"
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
                f"{spec.kind.value}, mas o experimento "
                f"exige {expected_kind.value}"
            )

    @staticmethod
    def _require_representation_match(
        *,
        producer_name: str,
        producer_output,
        consumer_name: str,
        consumer_input,
    ) -> None:
        # None = componente ainda não declarou restrição.
        if (
            producer_output is None
            or consumer_input is None
        ):
            return

        if producer_output != consumer_input:
            raise ValueError(
                f"Representações incompatíveis: "
                f"{producer_name} produz "
                f"'{producer_output.value}', mas "
                f"{consumer_name} exige "
                f"'{consumer_input.value}'"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload
