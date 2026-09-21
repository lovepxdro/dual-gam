import numpy as np
import pytest

from adarena.control.base import (
    ControlAction,
)

from adarena.control.enforcer import (
    DryRunRuleEnforcer,
)

from adarena.control.observation import (
    ControlledObservationPipeline,
)

from adarena.control.pipeline import (
    ControlPipeline,
)

from adarena.control.policy import (
    ThresholdDecisionPolicy,
)

from adarena.network.base import (
    FlowFeatureBatch,
)

from adarena.network.inference import (
    BinaryPredictor,
    NetworkInferencePipeline,
)


class FakeExtractor:
    def extract(
        self,
        capture,
        *,
        feature_names,
        strict=True,
    ):
        assert capture == "synthetic-capture"

        return FlowFeatureBatch(
            X=np.asarray(
                [
                    [0.1],
                    [0.9],
                ],
                dtype=np.float32,
            ),
            feature_names=tuple(
                feature_names
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


class FakePreprocessor:
    def __init__(self):
        self.feature_names = [
            "feature_1",
        ]

    def normalizar(
        self,
        X,
    ):
        return np.asarray(
            X,
            dtype=np.float32,
        )


class FakePredictor(
    BinaryPredictor
):
    def predict_proba(
        self,
        X,
    ):
        assert X.shape == (
            2,
            1,
        )

        return np.asarray(
            [
                0.2,
                0.8,
            ],
            dtype=np.float32,
        )


def test_controlled_observation_pipeline_end_to_end():
    inference_pipeline = (
        NetworkInferencePipeline(
            extractor=FakeExtractor(),
            preprocessor=FakePreprocessor(),
            predictor=FakePredictor(),
            threshold=0.5,
        )
    )

    enforcer = DryRunRuleEnforcer()

    control_pipeline = (
        ControlPipeline(
            policy=(
                ThresholdDecisionPolicy(
                    threshold=0.5
                )
            ),
            enforcer=enforcer,
        )
    )

    pipeline = (
        ControlledObservationPipeline(
            inference_pipeline=(
                inference_pipeline
            ),
            control_pipeline=(
                control_pipeline
            ),
        )
    )

    result = pipeline.process(
        "synthetic-capture"
    )

    assert (
        result.inference.n_flows
        == 2
    )

    assert (
        result.inference.benign_count
        == 1
    )

    assert (
        result.inference.attack_count
        == 1
    )

    assert (
        result.inference
        .probabilities
        .tolist()
        == pytest.approx(
            [
                0.2,
                0.8,
            ]
        )
    )

    assert (
        result.control.n_flows
        == 2
    )

    assert (
        result.control.block_count
        == 1
    )

    assert (
        result.control.applied_count
        == 0
    )

    assert (
        result.control
        .decisions[0]
        .action
        == ControlAction.NONE
    )

    assert (
        result.control
        .decisions[1]
        .action
        == ControlAction.BLOCK
    )

    assert len(
        enforcer.history
    ) == 2

    assert all(
        item.dry_run
        for item
        in enforcer.history
    )

    assert all(
        not item.applied
        for item
        in enforcer.history
    )
