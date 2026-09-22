from __future__ import annotations

from adarena.config_export import (
    dumps_experiment_config,
    save_experiment_config,
)
from adarena.config_io import (
    load_experiment_config,
)
from adarena.core.config import (
    ComponentSelection,
    DataSplitConfig,
    ExperimentConfig,
    ExperimentMode,
    NetworkSettings,
    TrainingSettings,
)


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        attacker=ComponentSelection(
            component_id="attacker.test",
            source="models/attacker.pth",
            params={
                "alpha": 1,
            },
        ),
        defender=ComponentSelection(
            component_id="defender.test",
            source="models/defender.pth",
        ),
        attack_dataset=(
            ComponentSelection(
                component_id="dataset.test",
                source="data/test.parquet",
            )
        ),
        benign_dataset=None,
        protocol=ComponentSelection(
            component_id="protocol.test",
        ),
        mode=ExperimentMode.SIMULATE,
        seed=7,
        device="cpu",
        output_dir="models/test",
        split=DataSplitConfig(
            test_size=0.25,
            validation_size=0.15,
        ),
        training=TrainingSettings(
            noise_dim=16,
            lr_defensor=0.001,
            lr_atacante=0.0002,
            epsilon=0.2,
            classification_threshold=0.6,
            epochs_pretrain=2,
            epochs_por_rodada=1,
            n_rodadas=3,
            amostras_por_rodada=100,
            amostras_avaliacao_adversarial=50,
            batch_size=32,
        ),
        network=NetworkSettings(
            renderer=ComponentSelection(
                component_id="renderer.test",
                params={
                    "target_ip": "172.20.0.10",
                    "target_port": 80,
                },
            ),
            network_backend=(
                ComponentSelection(
                    component_id="backend.test",
                    params={
                        "dry_run": True,
                    },
                )
            ),
            capture=ComponentSelection(
                component_id="capture.test",
                params={
                    "iface": "veth-test",
                },
            ),
            extractor=ComponentSelection(
                component_id="extractor.test",
            ),
            preprocessor_source=(
                "models/preprocessor"
            ),
            sample_count=10,
            packet_limit=100,
            capture_duration=3.0,
            classification_threshold=0.6,
            dry_run=True,
            observe=False,
        ),
        metadata={
            "configured_via": "test",
            "label": "round trip",
        },
    )


def test_dumps_experiment_config_gera_toml():
    text = dumps_experiment_config(
        _config()
    )

    assert 'mode = "simulate"' in text
    assert "[attacker]" in text
    assert "[attacker.params]" in text
    assert "[network.renderer]" in text
    assert (
        "[network.renderer.params]"
        in text
    )
    assert "dry_run = true" in text


def test_save_experiment_config_round_trip(
    tmp_path,
):
    path = (
        tmp_path
        / "experiment.toml"
    )

    save_experiment_config(
        _config(),
        path,
    )

    loaded = load_experiment_config(
        path
    )

    assert (
        loaded.mode
        == ExperimentMode.SIMULATE
    )
    assert loaded.seed == 7
    assert (
        loaded.attack_dataset.source
        == "data/test.parquet"
    )
    assert (
        loaded.attacker.params[
            "alpha"
        ]
        == 1
    )
    assert (
        loaded.training
        .classification_threshold
        == 0.6
    )
    assert loaded.network is not None
    assert (
        loaded.network
        .renderer
        .params["target_ip"]
        == "172.20.0.10"
    )
    assert (
        loaded.metadata[
            "configured_via"
        ]
        == "test"
    )


def test_save_nao_sobrescreve_por_padrao(
    tmp_path,
):
    path = (
        tmp_path
        / "experiment.toml"
    )

    path.write_text(
        "existing = true\n",
        encoding="utf-8",
    )

    try:
        save_experiment_config(
            _config(),
            path,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError(
            "deveria recusar sobrescrita"
        )
