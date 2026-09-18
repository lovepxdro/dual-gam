from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import json

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

from adarena.core.experiment import (
    ExperimentRunner,
)

from adarena.network.base import (
    CaptureBatch,
    CapturedPacket,
    FlowFeatureBatch,
)

from adarena.protocols.network_simulation import (
    NetworkSimulationProtocol,
)

from gan.preprocessing import (
    Preprocessador,
)


class FakeAttacker(
    torch.nn.Module
):

    def perturba(
        self,
        x_ddos,
        epsilon=0.3,
        device="cpu",
    ):
        # Identidade proposital.
        return x_ddos.clone()


class FakeDefender(
    torch.nn.Module
):

    def forward(
        self,
        x,
    ):
        # Probabilidade constante e válida.
        #
        # Como 0.1 < threshold 0.5,
        # todas as amostras serão vistas
        # como benignas.
        return torch.full(
            (
                x.shape[0],
            ),
            0.1,
            dtype=x.dtype,
            device=x.device,
        )


class FakeRenderer:

    def __init__(
        self,
        preprocessor,
        **_,
    ):
        self.preprocessor = (
            preprocessor
        )

    def render_batch(
        self,
        samples,
        *,
        scores=None,
        only_valid=False,
    ):
        samples = np.asarray(
            samples
        )

        return [
            {
                "index": index,
                "dry_run": True,
            }
            for index
            in range(
                len(
                    samples
                )
            )
        ]


class FakeBackend:

    def __init__(
        self,
        *,
        dry_run=True,
        **_,
    ):
        self.dry_run = (
            dry_run
        )

        self.payloads = []

    def execute_many(
        self,
        payloads,
    ):
        self.payloads = list(
            payloads
        )

        return [
            {
                "ok": True,
                "dry_run": True,
            }
            for _
            in self.payloads
        ]


class FakeCapture:

    def __init__(
        self,
        **_,
    ):
        self.started = False
        self.stopped = False

        self.packet_limit = None

    def start(
        self,
        *,
        packet_limit=None,
    ):
        self.started = True

        self.packet_limit = (
            packet_limit
        )

    def stop(
        self,
    ):
        self.stopped = True

        packet = CapturedPacket(
            timestamp=1.0,

            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",

            src_port=12345,
            dst_port=80,

            protocol=6,

            length=60,

            tcp_flags="A",

            payload_length=0,

            metadata={
                "synthetic": True,
            },
        )

        return CaptureBatch(
            packets=[
                packet,
            ],

            started_at=1.0,
            ended_at=2.0,

            interface="fake0",

            metadata={
                "synthetic": True,
            },
        )


class FakeExtractor:

    def extract(
        self,
        capture,
        *,
        feature_names,
        strict=True,
    ):
        assert len(
            capture.packets
        ) == 1

        assert strict is False

        return FlowFeatureBatch(
            X=np.asarray(
                [
                    [
                        0.15,
                        0.25,
                    ],
                    [
                        0.35,
                        0.45,
                    ],
                ],

                dtype=np.float32,
            ),

            feature_names=tuple(
                feature_names
            ),

            flow_ids=(
                "synthetic-flow-1",
                "synthetic-flow-2",
            ),

            unsupported_features=(),

            metadata={
                "synthetic": True,
            },
        )


class FakeDataset:

    def __init__(
        self,
        data,
    ):
        self.data = data

    def load(
        self,
        source,
    ):
        return self.data


class FakeRegistry:

    def __init__(
        self,
        *,
        specs,
        factories,
    ):
        self.specs = specs
        self.factories = factories

    def spec(
        self,
        component_id,
    ):
        return self.specs[
            component_id
        ]

    def create(
        self,
        component_id,
        **params,
    ):
        return self.factories[
            component_id
        ](
            **params
        )


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


def _dataset_data():

    X = np.asarray(
        [
            [
                float(index),
                float(index + 1),
            ]
            for index
            in range(
                40
            )
        ],

        dtype=np.float32,
    )

    y = np.asarray(
        [
            index % 2
            for index
            in range(
                40
            )
        ],

        dtype=np.float32,
    )

    schema = SimpleNamespace(
        feature_names=(
            "feature_1",
            "feature_2",
        ),

        label_mapping={
            "BENIGN": 0,
            "ATTACK": 1,
        },
    )

    return SimpleNamespace(
        X=X,
        y=y,

        schema=schema,

        metadata={
            "synthetic": True,
        },

        attack_mask=(
            y == 1
        ),

        benign_mask=(
            y == 0
        ),
    )


def _save_preprocessor(
    tmp_path,
    dataset_data,
):

    preprocessor = (
        Preprocessador()
    )

    preprocessor.configurar_dataset(
        dataset_data
    )

    preprocessor.split_e_normalizar(
        dataset_data.X,
        dataset_data.y,

        test_size=0.2,
        validation_size=0.1,

        random_state=42,
    )

    path = (
        tmp_path
        / "source_preprocessor"
    )

    preprocessor.salvar(
        path
    )

    return path


def _save_checkpoints(
    tmp_path,
):

    attacker_path = (
        tmp_path
        / "attacker.pth"
    )

    defender_path = (
        tmp_path
        / "defender.pth"
    )

    torch.save(
        FakeAttacker()
        .state_dict(),
        attacker_path,
    )

    torch.save(
        FakeDefender()
        .state_dict(),
        defender_path,
    )

    return (
        attacker_path,
        defender_path,
    )


def _registry(
    dataset_data,
):

    specs = {
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

        "protocol": _spec(
            ComponentKind
            .EXPERIMENT_PROTOCOL,
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

    factories = {
        "attacker": (
            lambda **_: (
                FakeAttacker()
            )
        ),

        "defender": (
            lambda **_: (
                FakeDefender()
            )
        ),

        "dataset": (
            lambda **_: (
                FakeDataset(
                    dataset_data
                )
            )
        ),

        "protocol": (
            lambda **_: (
                NetworkSimulationProtocol()
            )
        ),

        "renderer": (
            lambda **params: (
                FakeRenderer(
                    **params
                )
            )
        ),

        "backend": (
            lambda **params: (
                FakeBackend(
                    **params
                )
            )
        ),

        "capture": (
            lambda **params: (
                FakeCapture(
                    **params
                )
            )
        ),

        "extractor": (
            lambda **_: (
                FakeExtractor()
            )
        ),
    }

    return FakeRegistry(
        specs=specs,
        factories=factories,
    )


def test_runner_executa_network_simulation_end_to_end(
    tmp_path,
):

    dataset_data = (
        _dataset_data()
    )

    preprocessor_path = (
        _save_preprocessor(
            tmp_path,
            dataset_data,
        )
    )

    (
        attacker_path,
        defender_path,
    ) = _save_checkpoints(
        tmp_path
    )

    registry = _registry(
        dataset_data
    )

    models_dir = (
        tmp_path
        / "models"
    )

    config = ExperimentConfig(
        attacker=ComponentSelection(
            component_id="attacker",

            source=str(
                attacker_path
            ),
        ),

        defender=ComponentSelection(
            component_id="defender",

            source=str(
                defender_path
            ),
        ),

        attack_dataset=(
            ComponentSelection(
                component_id="dataset",

                source=(
                    "synthetic.parquet"
                ),
            )
        ),

        protocol=ComponentSelection(
            component_id="protocol"
        ),

        mode=(
            ExperimentMode.SIMULATE
        ),

        seed=42,

        device="cpu",

        output_dir=str(
            models_dir
        ),

        network=NetworkSettings(
            renderer=ComponentSelection(
                component_id="renderer"
            ),

            network_backend=(
                ComponentSelection(
                    component_id="backend"
                )
            ),

            capture=ComponentSelection(
                component_id="capture"
            ),

            extractor=ComponentSelection(
                component_id="extractor"
            ),

            preprocessor_source=str(
                preprocessor_path
            ),

            sample_count=3,

            packet_limit=10,

            classification_threshold=0.5,

            dry_run=True,

            observe=True,
        ),
    )

    runner = ExperimentRunner(
        config=config,
        registry=registry,
    )

    result = runner.run()

    # ---------------------------------
    # Execução
    # ---------------------------------

    assert (
        result.final_metrics[
            "samples_selected"
        ]
        == 3
    )

    assert (
        result.final_metrics[
            "mathematical_evasions"
        ]
        == 3
    )

    assert (
        result.final_metrics[
            "valid_renderings"
        ]
        == 3
    )

    assert (
        result.final_metrics[
            "dry_run_executions"
        ]
        == 3
    )

    assert (
        result.final_metrics[
            "observation_enabled"
        ]
        is True
    )

    assert (
        result.final_metrics[
            "captured_packets"
        ]
        == 1
    )

    assert (
        result.final_metrics[
            "reconstructed_flows"
        ]
        == 2
    )

    assert (
        result.final_metrics[
            "network_benign_count"
        ]
        == 2
    )

    assert (
        result.final_metrics[
            "network_attack_count"
        ]
        == 0
    )

    # ---------------------------------
    # Artefatos do Runner
    # ---------------------------------

    assert (
        result.run_dir
        .joinpath(
            "config_execucao.json"
        )
        .exists()
    )

    assert (
        result.run_dir
        .joinpath(
            "preprocessador",
            "scaler.pkl",
        )
        .exists()
    )

    assert (
        result.run_dir
        .joinpath(
            "preprocessador",
            "feature_names.pkl",
        )
        .exists()
    )

    # ---------------------------------
    # Artefatos do protocolo
    # ---------------------------------

    assert (
        result.run_dir
        .joinpath(
            "network",
            "selected_attack_samples.npy",
        )
        .exists()
    )

    assert (
        result.run_dir
        .joinpath(
            "network",
            "adversarial_samples.npy",
        )
        .exists()
    )

    assert (
        result.run_dir
        .joinpath(
            "network",
            "simulation_summary.json",
        )
        .exists()
    )

    # ---------------------------------
    # Execução ativa
    # ---------------------------------

    latest = (
        models_dir
        / "latest"
    )

    assert (
        latest.is_symlink()
    )

    assert (
        latest.resolve()
        == result.run_dir.resolve()
    )


def test_runner_persiste_configuracao_completa_de_rede(
    tmp_path,
):

    dataset_data = (
        _dataset_data()
    )

    preprocessor_path = (
        _save_preprocessor(
            tmp_path,
            dataset_data,
        )
    )

    (
        attacker_path,
        defender_path,
    ) = _save_checkpoints(
        tmp_path
    )

    registry = _registry(
        dataset_data
    )

    models_dir = (
        tmp_path
        / "models"
    )

    config = ExperimentConfig(
        attacker=ComponentSelection(
            component_id="attacker",

            source=str(
                attacker_path
            ),
        ),

        defender=ComponentSelection(
            component_id="defender",

            source=str(
                defender_path
            ),
        ),

        attack_dataset=(
            ComponentSelection(
                component_id="dataset",

                source=(
                    "synthetic.parquet"
                ),
            )
        ),

        protocol=ComponentSelection(
            component_id="protocol"
        ),

        mode=(
            ExperimentMode.SIMULATE
        ),

        seed=42,

        device="cpu",

        output_dir=str(
            models_dir
        ),

        network=NetworkSettings(
            renderer=ComponentSelection(
                component_id="renderer",

                params={
                    "target_ip": (
                        "10.0.0.10"
                    ),

                    "target_port": 80,
                },
            ),

            network_backend=(
                ComponentSelection(
                    component_id="backend",

                    params={
                        "iface": "fake0",
                    },
                )
            ),

            capture=ComponentSelection(
                component_id="capture",

                params={
                    "iface": "fake0",
                },
            ),

            extractor=ComponentSelection(
                component_id="extractor"
            ),

            preprocessor_source=str(
                preprocessor_path
            ),

            sample_count=3,

            packet_limit=10,

            classification_threshold=0.4,

            dry_run=True,

            observe=True,
        ),
    )

    runner = ExperimentRunner(
        config=config,
        registry=registry,
    )

    result = runner.run()

    config_path = (
        result.run_dir
        / "config_execucao.json"
    )

    assert config_path.exists()

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:
        snapshot = json.load(
            file
        )

    # ---------------------------------
    # Identidade da execução
    # ---------------------------------

    assert (
        snapshot[
            "mode"
        ]
        == "simulate"
    )

    assert (
        snapshot[
            "seed"
        ]
        == 42
    )

    # ---------------------------------
    # Componentes
    # ---------------------------------

    assert (
        snapshot[
            "components"
        ][
            "attacker"
        ]
        == "attacker"
    )

    assert (
        snapshot[
            "components"
        ][
            "defender"
        ]
        == "defender"
    )

    assert (
        snapshot[
            "components"
        ][
            "protocol"
        ]
        == "protocol"
    )

    assert (
        snapshot[
            "components"
        ][
            "renderer"
        ]
        == "renderer"
    )

    assert (
        snapshot[
            "components"
        ][
            "network_backend"
        ]
        == "backend"
    )

    assert (
        snapshot[
            "components"
        ][
            "capture"
        ]
        == "capture"
    )

    assert (
        snapshot[
            "components"
        ][
            "extractor"
        ]
        == "extractor"
    )

    # ---------------------------------
    # Checkpoints/modelos
    # ---------------------------------

    assert (
        snapshot[
            "component_config"
        ][
            "attacker"
        ][
            "source"
        ]
        == str(
            attacker_path
        )
    )

    assert (
        snapshot[
            "component_config"
        ][
            "defender"
        ][
            "source"
        ]
        == str(
            defender_path
        )
    )

    # ---------------------------------
    # Rede
    # ---------------------------------

    network = snapshot[
        "network"
    ]

    assert (
        network[
            "preprocessor_source"
        ]
        == str(
            preprocessor_path
        )
    )

    assert (
        network[
            "sample_count"
        ]
        == 3
    )

    assert (
        network[
            "packet_limit"
        ]
        == 10
    )

    assert (
        network[
            "classification_threshold"
        ]
        == 0.4
    )

    assert (
        network[
            "effective_classification_threshold"
        ]
        == 0.4
    )

    assert (
        network[
            "dry_run"
        ]
        is True
    )

    assert (
        network[
            "observe"
        ]
        is True
    )

    # ---------------------------------
    # Configuração dos componentes
    # ---------------------------------

    assert (
        network[
            "renderer"
        ][
            "component_id"
        ]
        == "renderer"
    )

    assert (
        network[
            "renderer"
        ][
            "params"
        ][
            "target_ip"
        ]
        == "10.0.0.10"
    )

    assert (
        network[
            "renderer"
        ][
            "params"
        ][
            "target_port"
        ]
        == 80
    )

    assert (
        network[
            "network_backend"
        ][
            "params"
        ][
            "iface"
        ]
        == "fake0"
    )

    assert (
        network[
            "capture"
        ][
            "params"
        ][
            "iface"
        ]
        == "fake0"
    )
