import pytest

from adarena.builtin import (
    ADVERSARIAL_PROTOCOL_ID,
    BINARY_MLP_DEFENDER_ID,
    CICIDS2017_DATASET_ID,
    CICIDS2017_RENDERER_ID,
    PERTURBATION_ATTACKER_ID,
    SCAPY_NETWORK_BACKEND_ID,
    create_default_registry,
)

from adarena.core.components import (
    ComponentKind,
    ComponentSpec,
    DataRepresentation,
)

from adarena.core.config import (
    ComponentSelection,
    ExperimentConfig,
)

from adarena.core.registry import (
    ComponentRegistry,
)

from adarena.datasets import (
    CICIDS2017Dataset,
)

from gan.models import (
    Atacante,
    Defensor,
)


def _default_experiment() -> ExperimentConfig:

    return ExperimentConfig(
        attacker=ComponentSelection(
            PERTURBATION_ATTACKER_ID,
            {
                "noise_dim": 16,
                "output_dim": 10,
            },
        ),

        defender=ComponentSelection(
            BINARY_MLP_DEFENDER_ID,
            {
                "input_dim": 10,
            },
        ),

        attack_dataset=ComponentSelection(
            CICIDS2017_DATASET_ID,
            {
                "path": (
                    "/data/ddos.parquet"
                ),
            },
        ),

        seed=42,
        device="cpu",
    )


def test_registry_builtin_expoe_componentes_atuais():

    registry = (
        create_default_registry()
    )

    attackers = registry.list(
        kind=(
            ComponentKind.ATTACKER
        )
    )

    defenders = registry.list(
        kind=(
            ComponentKind.DEFENDER
        )
    )

    datasets = registry.list(
        kind=(
            ComponentKind.DATASET
        )
    )

    protocols = registry.list(
        kind=(
            ComponentKind
            .EXPERIMENT_PROTOCOL
        )
    )

    renderers = registry.list(
        kind=(
            ComponentKind.RENDERER
        )
    )

    network_backends = registry.list(
        kind=(
            ComponentKind
            .NETWORK_BACKEND
        )
    )

    assert [
        spec.component_id
        for spec
        in attackers
    ] == [
        PERTURBATION_ATTACKER_ID
    ]

    assert [
        spec.component_id
        for spec
        in defenders
    ] == [
        BINARY_MLP_DEFENDER_ID
    ]

    assert [
        spec.component_id
        for spec
        in datasets
    ] == [
        CICIDS2017_DATASET_ID
    ]

    assert [
        spec.component_id
        for spec
        in protocols
    ] == [
        ADVERSARIAL_PROTOCOL_ID
    ]

    assert [
        spec.component_id
        for spec
        in renderers
    ] == [
        CICIDS2017_RENDERER_ID
    ]


    assert [
        spec.component_id
        for spec
        in network_backends
    ] == [
        SCAPY_NETWORK_BACKEND_ID
    ]

def test_registry_instancia_implementacoes_atuais():

    registry = (
        create_default_registry()
    )

    attacker = registry.create(
        PERTURBATION_ATTACKER_ID,
        noise_dim=16,
        output_dim=10,
    )

    defender = registry.create(
        BINARY_MLP_DEFENDER_ID,
        input_dim=10,
    )

    dataset = registry.create(
        CICIDS2017_DATASET_ID
    )

    assert isinstance(
        attacker,
        Atacante,
    )

    assert isinstance(
        defender,
        Defensor,
    )

    assert isinstance(
        dataset,
        CICIDS2017Dataset,
    )

    assert (
        attacker.noise_dim
        == 16
    )


def test_registry_rejeita_id_duplicado():

    registry = (
        ComponentRegistry()
    )

    spec = ComponentSpec(
        component_id=(
            "test.attacker"
        ),
        kind=(
            ComponentKind.ATTACKER
        ),
        name="Test",
    )

    registry.register(
        spec,
        object,
    )

    with pytest.raises(
        ValueError,
        match="já registrado",
    ):
        registry.register(
            spec,
            object,
        )


def test_experiment_config_valida_componentes_compativeis():

    registry = (
        create_default_registry()
    )

    config = (
        _default_experiment()
    )

    config.validate(
        registry
    )


def test_experiment_config_rejeita_dataset_incompativel():

    registry = (
        create_default_registry()
    )

    registry.register(
        ComponentSpec(
            component_id=(
                "test.packet_dataset"
            ),
            kind=(
                ComponentKind.DATASET
            ),
            name=(
                "Packet Dataset"
            ),
            output_representation=(
                DataRepresentation
                .PACKET_SEQUENCE
            ),
        ),
        object,
    )

    config = (
        _default_experiment()
    )

    config.attack_dataset = (
        ComponentSelection(
            "test.packet_dataset"
        )
    )

    with pytest.raises(
        ValueError,
        match=(
            "Representações incompatíveis"
        ),
    ):
        config.validate(
            registry
        )


def test_config_serializa_ids_e_parametros():

    config = (
        _default_experiment()
    )

    payload = (
        config.to_dict()
    )

    assert (
        payload[
            "attacker"
        ][
            "component_id"
        ]
        == PERTURBATION_ATTACKER_ID
    )

    assert (
        payload[
            "attacker"
        ][
            "params"
        ][
            "noise_dim"
        ]
        == 16
    )

    assert (
        payload["mode"]
        == "train"
    )
