from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adarena.core.config import (
    ComponentSelection,
    ExperimentConfig,
    ExperimentMode,
    NetworkSettings,
)

from adarena.network.base import (
    CaptureBatch,
    CapturedPacket,
    FlowFeatureBatch,
)

from adarena.protocols.base import (
    ProtocolContext,
)

from adarena.protocols.network_simulation import (
    NetworkSimulationProtocol,
)


class FakeAttacker(
    torch.nn.Module
):

    def __init__(
        self,
    ):
        super().__init__()

    def perturba(
        self,
        x_ddos,
        epsilon=0.3,
        device="cpu",
    ):
        # Identidade proposital:
        # o teste verifica a orquestração,
        # não a qualidade do modelo.
        return x_ddos.clone()


class FakeDefender(
    torch.nn.Module
):

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        x,
    ):
        # A primeira feature funciona
        # diretamente como probabilidade.
        return x[:, 0]


class FakePreprocessor:

    feature_names = [
        "feature_1",
        "feature_2",
    ]

    def normalizar(
        self,
        X,
    ):
        # Identidade proposital para
        # deixar o teste determinístico.
        return np.asarray(
            X,
            dtype=np.float32,
        )


class FakeRenderer:

    def __init__(
        self,
    ):
        self.samples = None
        self.scores = None
        self.only_valid = None

    def render_batch(
        self,
        samples,
        *,
        scores=None,
        only_valid=False,
    ):
        self.samples = np.asarray(
            samples
        )

        self.scores = scores

        self.only_valid = (
            only_valid
        )

        return [
            {
                "sample_index": index,
            }
            for index
            in range(
                len(
                    self.samples
                )
            )
        ]


class FakeBackend:

    def __init__(
        self,
    ):
        self.payloads = None

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
            }
            for _
            in self.payloads
        ]


class FakeCapture:

    def __init__(
        self,
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

            metadata={},
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

    def __init__(
        self,
    ):
        self.received_capture = None
        self.received_features = None
        self.received_strict = None

    def extract(
        self,
        capture,
        *,
        feature_names,
        strict=True,
    ):
        self.received_capture = (
            capture
        )

        self.received_features = tuple(
            feature_names
        )

        self.received_strict = (
            strict
        )

        # Duas observações reconstruídas:
        #
        # 0.20 < 0.5 -> benigno
        # 0.80 >= 0.5 -> ataque
        return FlowFeatureBatch(
            X=np.asarray(
                [
                    [0.20, 10.0],
                    [0.80, 20.0],
                ],

                dtype=np.float32,
            ),

            feature_names=(
                "feature_1",
                "feature_2",
            ),

            flow_ids=(
                "flow-benign",
                "flow-attack",
            ),

            unsupported_features=(),

            metadata={
                "synthetic": True,
            },
        )


def _config(
    tmp_path,
    *,
    mode=ExperimentMode.SIMULATE,
    dry_run=True,
    observe=False,
):

    return ExperimentConfig(
        attacker=ComponentSelection(
            component_id="attacker",

            source=str(
                tmp_path
                / "attacker.pth"
            ),
        ),

        defender=ComponentSelection(
            component_id="defender",

            source=str(
                tmp_path
                / "defender.pth"
            ),
        ),

        attack_dataset=(
            ComponentSelection(
                component_id="dataset",

                source=(
                    "dataset.parquet"
                ),
            )
        ),

        mode=mode,

        output_dir=str(
            tmp_path
        ),

        network=NetworkSettings(
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

            preprocessor_source=str(
                tmp_path
                / "preprocessador"
            ),

            sample_count=3,

            packet_limit=25,

            classification_threshold=0.5,

            dry_run=dry_run,

            observe=observe,
        ),
    )


def _context(
    tmp_path,
    *,
    mode=ExperimentMode.SIMULATE,
    dry_run=True,
    observe=False,
):

    config = _config(
        tmp_path,

        mode=mode,

        dry_run=dry_run,

        observe=observe,
    )

    # Três ataques:
    #
    # [0.10, ...]
    # [0.20, ...]
    # [0.30, ...]
    #
    # e uma amostra benigna.
    X_test = np.asarray(
        [
            [0.10, 1.0],
            [0.20, 2.0],
            [0.90, 3.0],
            [0.30, 4.0],
        ],

        dtype=np.float32,
    )

    y_test = np.asarray(
        [
            1,
            1,
            0,
            1,
        ],

        dtype=np.float32,
    )

    empty_X = np.empty(
        (
            0,
            2,
        ),

        dtype=np.float32,
    )

    empty_y = np.empty(
        (
            0,
        ),

        dtype=np.float32,
    )

    return ProtocolContext(
        config=config,

        registry=SimpleNamespace(),

        run_id="test_run",

        run_dir=tmp_path,

        X_train=empty_X,
        X_val=empty_X,
        X_test=X_test,

        y_train=empty_y,
        y_val=empty_y,
        y_test=y_test,

        preprocessor=(
            FakePreprocessor()
        ),

        dataset_data=None,

        config_snapshot={},
    )


def _patch_core_components(
    monkeypatch,
    protocol,
    *,
    attacker,
    defender,
    renderer,
    backend,
):

    monkeypatch.setattr(
        protocol,
        "_create_attacker",
        lambda context: attacker,
    )

    monkeypatch.setattr(
        protocol,
        "_create_defender",
        lambda context: defender,
    )

    monkeypatch.setattr(
        protocol,
        "_create_renderer",
        lambda context: renderer,
    )

    monkeypatch.setattr(
        protocol,
        "_create_backend",
        lambda context: backend,
    )


def test_network_simulation_executa_fluxo_dry_run(
    tmp_path,
    monkeypatch,
):

    context = _context(
        tmp_path
    )

    protocol = (
        NetworkSimulationProtocol()
    )

    attacker = FakeAttacker()

    defender = FakeDefender()

    renderer = FakeRenderer()

    backend = FakeBackend()

    _patch_core_components(
        monkeypatch,
        protocol,

        attacker=attacker,
        defender=defender,
        renderer=renderer,
        backend=backend,
    )

    result = protocol.run(
        context
    )

    metrics = (
        result.final_metrics
    )

    assert (
        metrics[
            "samples_selected"
        ]
        == 3
    )

    assert (
        metrics[
            "mathematical_evasions"
        ]
        == 3
    )

    assert (
        metrics[
            "mathematical_evasion_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        metrics[
            "valid_renderings"
        ]
        == 3
    )

    assert (
        metrics[
            "valid_rendering_rate"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        metrics[
            "valid_rendering_rate_given_evasion"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert (
        metrics[
            "dry_run_executions"
        ]
        == 3
    )

    assert (
        metrics[
            "dry_run"
        ]
        is True
    )

    assert (
        metrics[
            "observation_enabled"
        ]
        is False
    )

    assert (
        renderer.only_valid
        is True
    )

    assert (
        len(
            backend.payloads
        )
        == 3
    )

    assert (
        len(
            result.evaluations[
                "selected_indices"
            ]
        )
        == 3
    )

    assert (
        result.evaluations[
            "predictions"
        ]
        == [
            0,
            0,
            0,
        ]
    )

    assert (
        result.evaluations[
            "evasion_mask"
        ]
        == [
            True,
            True,
            True,
        ]
    )

    assert (
        tmp_path
        .joinpath(
            "network",
            "selected_attack_samples.npy",
        )
        .exists()
    )

    assert (
        tmp_path
        .joinpath(
            "network",
            "adversarial_samples.npy",
        )
        .exists()
    )

    assert (
        tmp_path
        .joinpath(
            "network",
            "simulation_summary.json",
        )
        .exists()
    )


def test_network_simulation_observa_fluxo_reconstruido(
    tmp_path,
    monkeypatch,
):

    context = _context(
        tmp_path,
        observe=True,
    )

    protocol = (
        NetworkSimulationProtocol()
    )

    attacker = FakeAttacker()

    defender = FakeDefender()

    renderer = FakeRenderer()

    backend = FakeBackend()

    capture = FakeCapture()

    extractor = FakeExtractor()

    _patch_core_components(
        monkeypatch,
        protocol,

        attacker=attacker,
        defender=defender,
        renderer=renderer,
        backend=backend,
    )

    monkeypatch.setattr(
        protocol,
        "_create_capture",
        lambda context: capture,
    )

    monkeypatch.setattr(
        protocol,
        "_create_extractor",
        lambda context: extractor,
    )

    result = protocol.run(
        context
    )

    metrics = (
        result.final_metrics
    )

    assert (
        metrics[
            "observation_enabled"
        ]
        is True
    )

    assert capture.started is True
    assert capture.stopped is True

    assert (
        capture.packet_limit
        == 25
    )

    assert (
        metrics[
            "captured_packets"
        ]
        == 1
    )

    assert (
        metrics[
            "reconstructed_flows"
        ]
        == 2
    )

    assert (
        metrics[
            "network_attack_count"
        ]
        == 1
    )

    assert (
        metrics[
            "network_benign_count"
        ]
        == 1
    )

    assert (
        metrics[
            "network_attack_rate"
        ]
        == pytest.approx(
            0.5
        )
    )

    assert (
        result.evaluations[
            "network_predictions"
        ]
        == [
            0,
            1,
        ]
    )

    assert (
        result.evaluations[
            "network_probabilities"
        ]
        == pytest.approx(
            [
                0.20,
                0.80,
            ]
        )
    )

    assert (
        result.evaluations[
            "network_flow_ids"
        ]
        == [
            "flow-benign",
            "flow-attack",
        ]
    )

    assert (
        extractor
        .received_features
        == (
            "feature_1",
            "feature_2",
        )
    )

    assert (
        extractor
        .received_strict
        is False
    )

    # Mesmo no ramo de observação,
    # o backend continua recebendo
    # somente os payloads traduzidos.
    assert (
        len(
            backend.payloads
        )
        == 3
    )


def test_network_simulation_rejeita_execucao_nao_dry_run(
    tmp_path,
):

    context = _context(
        tmp_path,
        dry_run=False,
    )

    protocol = (
        NetworkSimulationProtocol()
    )

    with pytest.raises(
        ValueError,
        match="apenas em dry-run",
    ):
        protocol.run(
            context
        )


def test_network_simulation_rejeita_modo_incorreto(
    tmp_path,
):

    context = _context(
        tmp_path,

        mode=(
            ExperimentMode.TRAIN
        ),
    )

    protocol = (
        NetworkSimulationProtocol()
    )

    with pytest.raises(
        ValueError,
        match="mode=SIMULATE",
    ):
        protocol.run(
            context
        )
