from __future__ import annotations

import pytest

from adarena.builtin import (
    create_default_registry,
)
from adarena.core.components import (
    ComponentKind,
)
from adarena.core.config import (
    ExperimentMode,
)
from adarena.tui.builder import (
    BuilderValues,
    build_experiment_config,
    first_component_id,
    preferred_protocol_id,
)


def _ids():
    registry = create_default_registry()

    return {
        "dataset": first_component_id(
            registry,
            ComponentKind.DATASET,
        ),
        "attacker": first_component_id(
            registry,
            ComponentKind.ATTACKER,
        ),
        "defender": first_component_id(
            registry,
            ComponentKind.DEFENDER,
        ),
        "renderer": first_component_id(
            registry,
            ComponentKind.RENDERER,
        ),
        "backend": first_component_id(
            registry,
            ComponentKind.NETWORK_BACKEND,
        ),
        "capture": first_component_id(
            registry,
            ComponentKind.CAPTURE,
        ),
        "extractor": first_component_id(
            registry,
            ComponentKind.EXTRACTOR,
        ),
        "train_protocol": (
            preferred_protocol_id(
                registry,
                ExperimentMode.TRAIN,
            )
        ),
        "simulate_protocol": (
            preferred_protocol_id(
                registry,
                ExperimentMode.SIMULATE,
            )
        ),
        "observe_protocol": (
            preferred_protocol_id(
                registry,
                ExperimentMode.OBSERVE,
            )
        ),
    }


def test_builder_train_cria_config_declarativa():
    ids = _ids()

    config = build_experiment_config(
        BuilderValues(
            mode=ExperimentMode.TRAIN,
            dataset_id=ids["dataset"],
            dataset_source="data/train.parquet",
            attacker_id=ids["attacker"],
            attacker_source=None,
            defender_id=ids["defender"],
            defender_source=None,
            protocol_id=ids["train_protocol"],
            seed=7,
            classification_threshold=0.6,
            n_rodadas=3,
        )
    )

    assert config.mode == ExperimentMode.TRAIN
    assert config.seed == 7
    assert (
        config.attack_dataset.source
        == "data/train.parquet"
    )
    assert config.attacker.source is None
    assert config.defender.source is None
    assert config.network is None
    assert (
        config.training
        .classification_threshold
        == pytest.approx(0.6)
    )
    assert config.training.n_rodadas == 3


def test_builder_simulate_forca_dry_run():
    ids = _ids()

    config = build_experiment_config(
        BuilderValues(
            mode=ExperimentMode.SIMULATE,
            dataset_id=ids["dataset"],
            dataset_source="data/test.parquet",
            attacker_id=ids["attacker"],
            attacker_source="attacker.pth",
            defender_id=ids["defender"],
            defender_source="defender.pth",
            protocol_id=ids["simulate_protocol"],
            renderer_id=ids["renderer"],
            network_backend_id=ids["backend"],
            capture_id=ids["capture"],
            extractor_id=ids["extractor"],
            preprocessor_source="preprocessor",
            target_ip="172.20.0.10",
        )
    )

    assert config.mode == ExperimentMode.SIMULATE
    assert config.network is not None
    assert config.network.dry_run is True
    assert (
        config.network
        .renderer
        .params["target_ip"]
        == "172.20.0.10"
    )


def test_builder_observe_nao_exige_attacker_dataset():
    ids = _ids()

    config = build_experiment_config(
        BuilderValues(
            mode=ExperimentMode.OBSERVE,
            dataset_id=None,
            dataset_source=None,
            attacker_id=None,
            attacker_source=None,
            defender_id=ids["defender"],
            defender_source="defender.pth",
            protocol_id=ids["observe_protocol"],
            capture_id=ids["capture"],
            extractor_id=ids["extractor"],
            preprocessor_source="preprocessor",
            capture_iface="veth-test",
        )
    )

    assert config.mode == ExperimentMode.OBSERVE
    assert config.attacker is None
    assert config.attack_dataset is None
    assert config.network is not None
    assert config.network.observe is True
    assert (
        config.network.capture
        .params["iface"]
        == "veth-test"
    )


def test_builder_simulate_exige_checkpoint():
    ids = _ids()

    with pytest.raises(
        ValueError,
        match="checkpoint.*Attacker",
    ):
        build_experiment_config(
            BuilderValues(
                mode=ExperimentMode.SIMULATE,
                dataset_id=ids["dataset"],
                dataset_source="data/test.parquet",
                attacker_id=ids["attacker"],
                attacker_source=None,
                defender_id=ids["defender"],
                defender_source="defender.pth",
                protocol_id=ids["simulate_protocol"],
                renderer_id=ids["renderer"],
                network_backend_id=ids["backend"],
                capture_id=ids["capture"],
                extractor_id=ids["extractor"],
                preprocessor_source="preprocessor",
                target_ip="172.20.0.10",
            )
        )
