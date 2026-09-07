from __future__ import annotations

from .core.components import (
    ComponentKind,
    ComponentSpec,
    DataRepresentation,
)

from .core.registry import (
    ComponentRegistry,
)

from .datasets import (
    CICIDS2017Dataset,
)


PERTURBATION_ATTACKER_ID = (
    "adarena.perturbation_attacker"
)

BINARY_MLP_DEFENDER_ID = (
    "adarena.binary_mlp_defender"
)

CICIDS2017_DATASET_ID = (
    "adarena.cicids2017"
)


def _create_perturbation_attacker(
    noise_dim: int = 32,
    output_dim: int = 77,
    **_,
):
    from gan.models import Atacante

    return Atacante(
        noise_dim=noise_dim,
        output_dim=output_dim,
    )


def _create_binary_mlp_defender(
    input_dim: int = 77,
    **_,
):
    from gan.models import Defensor

    return Defensor(
        input_dim=input_dim,
    )


def _create_cicids2017_dataset(
    label_column: str = "Label",
    benign_labels=(
        "BENIGN",
    ),
    attack_labels=None,
    **_,
):
    return CICIDS2017Dataset(
        label_column=label_column,
        benign_labels=tuple(
            benign_labels
        ),
        attack_labels=attack_labels,
    )


def register_builtin_components(
    registry: ComponentRegistry,
) -> ComponentRegistry:

    registry.register(
        ComponentSpec(
            component_id=(
                PERTURBATION_ATTACKER_ID
            ),
            kind=ComponentKind.ATTACKER,
            name="Perturbation Attacker",
            version="1.7",
            description=(
                "Atacante da linha 1.x: aprende "
                "perturbações adversariais sobre "
                "amostras reais."
            ),
            input_representation=(
                DataRepresentation.FLOW_FEATURES
            ),
            output_representation=(
                DataRepresentation.FLOW_FEATURES
            ),
            tags=(
                "builtin",
                "pytorch",
                "adversarial",
                "legacy-v1",
            ),
        ),
        _create_perturbation_attacker,
    )

    registry.register(
        ComponentSpec(
            component_id=(
                BINARY_MLP_DEFENDER_ID
            ),
            kind=ComponentKind.DEFENDER,
            name="Binary MLP Defender",
            version="1.7",
            description=(
                "Classificador binário de referência "
                "da ADArena."
            ),
            input_representation=(
                DataRepresentation.FLOW_FEATURES
            ),
            output_representation=(
                DataRepresentation
                .BINARY_CLASSIFICATION
            ),
            tags=(
                "builtin",
                "pytorch",
                "binary-classifier",
                "legacy-v1",
            ),
        ),
        _create_binary_mlp_defender,
    )

    registry.register(
        ComponentSpec(
            component_id=(
                CICIDS2017_DATASET_ID
            ),
            kind=ComponentKind.DATASET,
            name="CIC-IDS2017",
            version="2.0",
            description=(
                "Adapter do CIC-IDS2017 para "
                "flow features e classificação "
                "binária benigno/ataque."
            ),
            output_representation=(
                DataRepresentation.FLOW_FEATURES
            ),
            tags=(
                "builtin",
                "dataset",
                "flow-features",
                "cicids2017",
            ),
            metadata={
                "label_column": "Label",
            },
        ),
        _create_cicids2017_dataset,
    )

    return registry


def create_default_registry() -> ComponentRegistry:
    registry = ComponentRegistry()

    return register_builtin_components(
        registry
    )
