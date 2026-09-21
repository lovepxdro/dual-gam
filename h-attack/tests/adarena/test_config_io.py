from __future__ import annotations

import pytest

from adarena.config_io import (
    ConfigLoadError,
    experiment_config_from_dict,
)
from adarena.core.config import (
    ExperimentMode,
)


def test_config_io_converte_observe():
    config = experiment_config_from_dict(
        {
            "mode": "observe",
            "seed": 42,
            "device": "cpu",
            "output_dir": "models",
            "defender": {
                "component_id": (
                    "adarena.binary_mlp_defender"
                ),
                "source": "defender.pth",
            },
            "protocol": {
                "component_id": (
                    "adarena.network_observation"
                ),
            },
            "network": {
                "preprocessor_source": (
                    "preprocessador"
                ),
                "capture_duration": 5.0,
                "packet_limit": 500,
                "classification_threshold": 0.5,
                "dry_run": True,
                "capture": {
                    "component_id": (
                        "adarena.scapy_capture"
                    ),
                    "params": {
                        "iface": "test0",
                        "bpf_filter": "ip",
                    },
                },
                "extractor": {
                    "component_id": (
                        "adarena.basic_flow_extractor"
                    ),
                },
            },
        }
    )

    assert (
        config.mode
        == ExperimentMode.OBSERVE
    )
    assert config.attacker is None
    assert config.attack_dataset is None

    assert (
        config.defender.source
        == "defender.pth"
    )

    assert config.network is not None

    assert (
        config.network.capture_duration
        == 5.0
    )
    assert (
        config.network.packet_limit
        == 500
    )
    assert (
        config.network.capture
        is not None
    )
    assert (
        config.network.capture.params[
            "iface"
        ]
        == "test0"
    )


def test_config_io_converte_training_settings():
    config = experiment_config_from_dict(
        {
            "defender": {
                "component_id": (
                    "adarena.binary_mlp_defender"
                ),
            },
            "attacker": {
                "component_id": (
                    "adarena.perturbation_attacker"
                ),
            },
            "attack_dataset": {
                "component_id": (
                    "adarena.cicids2017"
                ),
                "source": "dataset.parquet",
            },
            "protocol": {
                "component_id": (
                    "adarena.adversarial_training"
                ),
            },
            "training": {
                "noise_dim": 16,
                "adam_betas": [
                    0.4,
                    0.99,
                ],
                "batch_size": 128,
            },
            "split": {
                "test_size": 0.25,
                "validation_size": 0.15,
            },
        }
    )

    assert config.training.noise_dim == 16
    assert (
        config.training.adam_betas
        == (0.4, 0.99)
    )
    assert config.training.batch_size == 128
    assert config.split.test_size == 0.25
    assert (
        config.split.validation_size
        == 0.15
    )


def test_config_io_rejeita_campo_desconhecido():
    with pytest.raises(
        ConfigLoadError,
        match="campo.*desconhecido",
    ):
        experiment_config_from_dict(
            {
                "defender": {
                    "component_id": (
                        "adarena.binary_mlp_defender"
                    ),
                },
                "banana": True,
            }
        )


def test_config_io_exige_defender():
    with pytest.raises(
        ConfigLoadError,
        match="defender",
    ):
        experiment_config_from_dict(
            {
                "mode": "observe",
            }
        )
