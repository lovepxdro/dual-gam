import pytest

from adarena.builtin import (
    BASIC_FLOW_EXTRACTOR_ID,
    BINARY_MLP_DEFENDER_ID,
    NETWORK_OBSERVATION_PROTOCOL_ID,
    SCAPY_CAPTURE_ID,
    create_default_registry,
)

from adarena.core.config import (
    ComponentSelection,
    ExperimentConfig,
    ExperimentMode,
    NetworkSettings,
)


def _observe_config() -> ExperimentConfig:

    return ExperimentConfig(
        attacker=None,

        defender=ComponentSelection(
            BINARY_MLP_DEFENDER_ID,
            source=(
                "/models/defender.pth"
            ),
        ),

        attack_dataset=None,

        protocol=ComponentSelection(
            NETWORK_OBSERVATION_PROTOCOL_ID
        ),

        mode=ExperimentMode.OBSERVE,

        device="cpu",

        network=NetworkSettings(
            capture=ComponentSelection(
                SCAPY_CAPTURE_ID,
                {
                    "iface": "test0",
                },
            ),

            extractor=ComponentSelection(
                BASIC_FLOW_EXTRACTOR_ID
            ),

            preprocessor_source=(
                "/models/preprocessador"
            ),

            capture_duration=5.0,

            dry_run=True,
        ),
    )


def test_observe_nao_exige_attacker_ou_dataset():

    registry = (
        create_default_registry()
    )

    config = _observe_config()

    config.validate(
        registry
    )


def test_observe_exige_preprocessador():

    registry = (
        create_default_registry()
    )

    config = _observe_config()

    config.network.preprocessor_source = None

    with pytest.raises(
        ValueError,
        match=(
            "OBSERVE exige "
            "network.preprocessor_source"
        ),
    ):
        config.validate(
            registry
        )


def test_observe_exige_checkpoint_defensor():

    registry = (
        create_default_registry()
    )

    config = _observe_config()

    config.defender.source = None

    with pytest.raises(
        ValueError,
        match=(
            "OBSERVE exige checkpoint "
            "em defender.source"
        ),
    ):
        config.validate(
            registry
        )


def test_observe_nao_exige_renderer_ou_backend():

    registry = (
        create_default_registry()
    )

    config = _observe_config()

    assert (
        config.network.renderer
        is None
    )

    assert (
        config
        .network
        .network_backend
        is None
    )

    config.validate(
        registry
    )
