from types import SimpleNamespace

import pytest

from adarena.core.components import (
    ComponentKind,
    DataRepresentation,
)

from adarena.core.config import (
    ComponentSelection,
    ExperimentConfig,
    ExperimentMode,
    NetworkSettings,
)


class FakeRegistry:

    def __init__(
        self,
        specs,
    ):
        self.specs = specs

    def spec(
        self,
        component_id,
    ):
        return self.specs[
            component_id
        ]


def _spec(
    kind,
    *,
    input_representation=None,
    output_representation=None,
):

    return SimpleNamespace(
        kind=kind,

        input_representation=(
            input_representation
        ),

        output_representation=(
            output_representation
        ),
    )


def _registry():

    return FakeRegistry(
        {
            "attacker": _spec(
                ComponentKind.ATTACKER,

                input_representation=(
                    DataRepresentation
                    .FLOW_FEATURES
                ),

                output_representation=(
                    DataRepresentation
                    .FLOW_FEATURES
                ),
            ),

            "defender": _spec(
                ComponentKind.DEFENDER,

                input_representation=(
                    DataRepresentation
                    .FLOW_FEATURES
                ),

                output_representation=(
                    DataRepresentation
                    .BINARY_CLASSIFICATION
                ),
            ),

            "dataset": _spec(
                ComponentKind.DATASET,

                output_representation=(
                    DataRepresentation
                    .FLOW_FEATURES
                ),
            ),

            "renderer": _spec(
                ComponentKind.RENDERER,

                input_representation=(
                    DataRepresentation
                    .FLOW_FEATURES
                ),

                output_representation=(
                    DataRepresentation
                    .ATTACK_PARAMS
                ),
            ),

            "backend": _spec(
                ComponentKind
                .NETWORK_BACKEND,

                input_representation=(
                    DataRepresentation
                    .ATTACK_PARAMS
                ),

                output_representation=(
                    DataRepresentation
                    .NETWORK_RESULT
                ),
            ),

            "capture": _spec(
                ComponentKind.CAPTURE,

                output_representation=(
                    DataRepresentation
                    .PACKET_RECORDS
                ),
            ),

            "extractor": _spec(
                ComponentKind.EXTRACTOR,

                input_representation=(
                    DataRepresentation
                    .PACKET_RECORDS
                ),

                output_representation=(
                    DataRepresentation
                    .FLOW_FEATURES
                ),
            ),
        }
    )


def _network():

    return NetworkSettings(
        renderer=ComponentSelection(
            "renderer"
        ),

        network_backend=(
            ComponentSelection(
                "backend"
            )
        ),

        capture=ComponentSelection(
            "capture"
        ),

        extractor=ComponentSelection(
            "extractor"
        ),

        sample_count=20,

        packet_limit=100,

        dry_run=True,
    )


def _config(
    *,
    mode=ExperimentMode.SIMULATE,
    network=None,
):

    return ExperimentConfig(
        attacker=ComponentSelection(
            "attacker"
        ),

        defender=ComponentSelection(
            "defender"
        ),

        attack_dataset=(
            ComponentSelection(
                "dataset",

                source="dataset.parquet",
            )
        ),

        mode=mode,

        network=network,
    )


def test_simulate_aceita_cadeia_de_rede_compativel():

    config = _config(
        network=_network()
    )

    config.validate(
        _registry()
    )


def test_simulate_exige_configuracao_de_rede():

    config = _config(
        network=None
    )

    with pytest.raises(
        ValueError,
        match="exige network",
    ):
        config.validate(
            _registry()
        )


def test_simulate_rejeita_representacao_incompativel():

    registry = _registry()

    # Renderer passa a esperar algo
    # diferente da saída FLOW_FEATURES
    # do atacante.
    registry.specs[
        "renderer"
    ] = _spec(
        ComponentKind.RENDERER,

        input_representation=(
            DataRepresentation
            .ATTACK_PARAMS
        ),

        output_representation=(
            DataRepresentation
            .ATTACK_PARAMS
        ),
    )

    config = _config(
        network=_network()
    )

    with pytest.raises(
        ValueError,
        match="Representações incompatíveis",
    ):
        config.validate(
            registry
        )
