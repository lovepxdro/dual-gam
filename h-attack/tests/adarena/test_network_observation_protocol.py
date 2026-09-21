from types import SimpleNamespace

import json

import numpy as np
import pytest

from adarena.core.config import (
    ComponentSelection,
    ExperimentConfig,
    ExperimentMode,
    NetworkSettings,
)

from adarena.protocols.base import (
    ProtocolContext,
)

from adarena.protocols.network_observation import (
    NetworkObservationProtocol,
)

import adarena.protocols.network_observation as observation_module


class FakeCapture:

    def __init__(self):
        self.calls = []

    def capture(
        self,
        *,
        duration,
        packet_limit=None,
    ):
        self.calls.append(
            {
                "duration": duration,
                "packet_limit": packet_limit,
            }
        )

        return SimpleNamespace(
            packets=(
                object(),
                object(),
                object(),
            ),
            metadata={
                "backend": "fake",
                "mode": "test",
            },
        )


class FakeControlledObservationPipeline:

    def __init__(
        self,
        *,
        inference_pipeline,
        control_pipeline,
    ):
        self.inference_pipeline = inference_pipeline
        self.control_pipeline = control_pipeline

    def process(
        self,
        capture_batch,
    ):
        inference = SimpleNamespace(
            n_flows=2,
            attack_count=1,
            benign_count=1,
            extracted=SimpleNamespace(
                flow_ids=(
                    "flow-benigno",
                    "flow-ataque",
                ),
            ),
            probabilities=np.asarray(
                [0.20, 0.90],
                dtype=np.float32,
            ),
            predictions=np.asarray(
                [0, 1],
                dtype=np.int64,
            ),
        )

        decisions = (
            SimpleNamespace(
                flow_id="flow-benigno",
                prediction=0,
                score=0.20,
                action=SimpleNamespace(
                    value="none"
                ),
                reason="below_threshold",
            ),
            SimpleNamespace(
                flow_id="flow-ataque",
                prediction=1,
                score=0.90,
                action=SimpleNamespace(
                    value="block"
                ),
                reason="threshold_reached",
            ),
        )

        enforcement_results = (
            SimpleNamespace(
                applied=False,
                success=True,
                dry_run=True,
                message="Nenhuma ação necessária",
            ),
            SimpleNamespace(
                applied=False,
                success=True,
                dry_run=True,
                message="BLOCK solicitado em dry-run",
            ),
        )

        control = SimpleNamespace(
            decisions=decisions,
            enforcement_results=(
                enforcement_results
            ),
            block_count=1,
            applied_count=0,
        )

        return SimpleNamespace(
            inference=inference,
            control=control,
        )


def _config(
    tmp_path,
    *,
    mode=ExperimentMode.OBSERVE,
    dry_run=True,
) -> ExperimentConfig:

    return ExperimentConfig(
        attacker=None,

        defender=ComponentSelection(
            component_id="test.defender",
            source=str(
                tmp_path
                / "defender.pth"
            ),
        ),

        attack_dataset=None,

        mode=mode,
        device="cpu",

        output_dir=str(
            tmp_path
        ),

        network=NetworkSettings(
            capture=ComponentSelection(
                component_id="test.capture",
            ),

            extractor=ComponentSelection(
                component_id="test.extractor",
            ),

            preprocessor_source=str(
                tmp_path
                / "preprocessador"
            ),

            capture_duration=2.5,
            packet_limit=10,

            classification_threshold=0.5,

            dry_run=dry_run,
        ),
    )


def _context(
    tmp_path,
    config,
) -> ProtocolContext:

    run_dir = (
        tmp_path
        / "run"
    )

    (
        run_dir
        / "metrics"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    return ProtocolContext(
        config=config,
        registry=SimpleNamespace(),

        run_id="run-test",
        run_dir=run_dir,

        X_train=None,
        X_val=None,
        X_test=None,

        y_train=None,
        y_val=None,
        y_test=None,

        preprocessor=SimpleNamespace(
            feature_names=[
                "f1",
                "f2",
            ],
        ),

        dataset_data=None,
        config_snapshot={},
    )


def test_network_observation_protocol_orquestra_e_salva_artefatos(
    tmp_path,
    monkeypatch,
):

    config = _config(
        tmp_path
    )

    context = _context(
        tmp_path,
        config,
    )

    protocol = (
        NetworkObservationProtocol()
    )

    fake_capture = FakeCapture()

    monkeypatch.setattr(
        protocol,
        "_create_defender",
        lambda context: object(),
    )

    monkeypatch.setattr(
        protocol,
        "_create_capture",
        lambda context: fake_capture,
    )

    monkeypatch.setattr(
        protocol,
        "_create_extractor",
        lambda context: object(),
    )

    monkeypatch.setattr(
        observation_module,
        "ControlledObservationPipeline",
        FakeControlledObservationPipeline,
    )

    result = protocol.run(
        context
    )

    assert fake_capture.calls == [
        {
            "duration": 2.5,
            "packet_limit": 10,
        }
    ]

    assert result.final_metrics[
        "captured_packets"
    ] == 3

    assert result.final_metrics[
        "reconstructed_flows"
    ] == 2

    assert result.final_metrics[
        "network_benign_count"
    ] == 1

    assert result.final_metrics[
        "network_attack_count"
    ] == 1

    assert result.final_metrics[
        "block_decisions"
    ] == 1

    assert result.final_metrics[
        "rules_applied"
    ] == 0

    assert result.final_metrics[
        "dry_run"
    ] is True

    assert result.evaluations[
        "predictions"
    ] == [0, 1]

    assert result.evaluations[
        "decisions"
    ][1][
        "action"
    ] == "block"

    assert result.evaluations[
        "decisions"
    ][1][
        "applied"
    ] is False

    metrics_path = (
        context.run_dir
        / "metrics"
        / "network_observation.json"
    )

    evaluations_path = (
        context.run_dir
        / "metrics"
        / (
            "network_observation_"
            "evaluations.json"
        )
    )

    assert metrics_path.exists()
    assert evaluations_path.exists()

    with open(
        metrics_path,
        encoding="utf-8",
    ) as file:
        persisted_metrics = json.load(file)

    assert persisted_metrics[
        "block_decisions"
    ] == 1

    assert persisted_metrics[
        "rules_applied"
    ] == 0


def test_network_observation_protocol_rejeita_modo_incorreto(
    tmp_path,
):

    config = _config(
        tmp_path,
        mode=ExperimentMode.TRAIN,
    )

    context = _context(
        tmp_path,
        config,
    )

    protocol = (
        NetworkObservationProtocol()
    )

    with pytest.raises(
        ValueError,
        match="mode=OBSERVE",
    ):
        protocol.run(
            context
        )


def test_network_observation_protocol_rejeita_enforcement_real(
    tmp_path,
):

    config = _config(
        tmp_path,
        dry_run=False,
    )

    context = _context(
        tmp_path,
        config,
    )

    protocol = (
        NetworkObservationProtocol()
    )

    with pytest.raises(
        ValueError,
        match="dry_run=True",
    ):
        protocol.run(
            context
        )
