import pytest

import numpy as np

from adarena.control.pipeline import (
    ControlPipeline,
)

from adarena.network.base import (
    FlowFeatureBatch,
)

from adarena.network.inference import (
    NetworkInferenceResult,
)

from adarena.control.base import (
    ControlAction,
)

from adarena.control.enforcer import (
    DryRunRuleEnforcer,
)

from adarena.control.policy import (
    ThresholdDecisionPolicy,
)


def test_threshold_policy_returns_none_below_threshold():
    policy = ThresholdDecisionPolicy(
        threshold=0.5
    )

    decision = policy.decide(
        flow_id="flow-benign",
        prediction=0,
        score=0.2,
    )

    assert decision.flow_id == "flow-benign"
    assert decision.prediction == 0
    assert decision.score == 0.2
    assert decision.action == ControlAction.NONE


def test_threshold_policy_returns_block_at_threshold():
    policy = ThresholdDecisionPolicy(
        threshold=0.5
    )

    decision = policy.decide(
        flow_id="flow-attack",
        prediction=1,
        score=0.5,
    )

    assert decision.action == ControlAction.BLOCK


def test_threshold_policy_returns_block_above_threshold():
    policy = ThresholdDecisionPolicy(
        threshold=0.5
    )

    decision = policy.decide(
        flow_id="flow-attack",
        prediction=1,
        score=0.9,
    )

    assert decision.action == ControlAction.BLOCK


@pytest.mark.parametrize(
    "threshold",
    [
        -0.1,
        1.1,
    ],
)
def test_threshold_policy_rejects_invalid_threshold(
    threshold,
):
    with pytest.raises(
        ValueError,
        match="threshold deve estar entre 0 e 1",
    ):
        ThresholdDecisionPolicy(
            threshold=threshold
        )


@pytest.mark.parametrize(
    "score",
    [
        -0.1,
        1.1,
        float("nan"),
        float("inf"),
    ],
)
def test_threshold_policy_rejects_invalid_score(
    score,
):
    policy = ThresholdDecisionPolicy()

    with pytest.raises(ValueError):
        policy.decide(
            flow_id="flow",
            prediction=0,
            score=score,
        )


def test_threshold_policy_rejects_invalid_prediction():
    policy = ThresholdDecisionPolicy()

    with pytest.raises(
        ValueError,
        match="prediction deve ser 0 ou 1",
    ):
        policy.decide(
            flow_id="flow",
            prediction=2,
            score=0.5,
        )


def test_policy_preserves_metadata():
    policy = ThresholdDecisionPolicy()

    decision = policy.decide(
        flow_id="flow-1",
        prediction=0,
        score=0.1,
        metadata={
            "src_ip": "10.0.0.1",
            "dst_ip": "10.0.0.2",
        },
    )

    assert decision.metadata == {
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
    }


def test_dry_run_enforcer_processes_none_decision():
    policy = ThresholdDecisionPolicy()
    enforcer = DryRunRuleEnforcer()

    decision = policy.decide(
        flow_id="flow-benign",
        prediction=0,
        score=0.1,
    )

    result = enforcer.enforce(
        decision
    )

    assert result.success is True
    assert result.applied is False
    assert result.dry_run is True

    assert result.decision is decision

    assert len(enforcer.history) == 1


def test_dry_run_enforcer_processes_block_without_applying():
    policy = ThresholdDecisionPolicy()
    enforcer = DryRunRuleEnforcer()

    decision = policy.decide(
        flow_id="flow-attack",
        prediction=1,
        score=0.9,
    )

    result = enforcer.enforce(
        decision
    )

    assert decision.action == ControlAction.BLOCK

    assert result.success is True
    assert result.applied is False
    assert result.dry_run is True

    assert "não aplicado" in result.message


def test_dry_run_enforcer_preserves_history():
    policy = ThresholdDecisionPolicy()
    enforcer = DryRunRuleEnforcer()

    decisions = [
        policy.decide(
            flow_id="flow-1",
            prediction=0,
            score=0.1,
        ),
        policy.decide(
            flow_id="flow-2",
            prediction=1,
            score=0.9,
        ),
    ]

    for decision in decisions:
        enforcer.enforce(
            decision
        )

    assert len(enforcer.history) == 2

    assert (
        enforcer.history[0]
        .decision
        .flow_id
        == "flow-1"
    )

    assert (
        enforcer.history[1]
        .decision
        .flow_id
        == "flow-2"
    )

def _make_inference_result():
    extracted = FlowFeatureBatch(
        X=np.asarray(
            [
                [0.1],
                [0.9],
            ],
            dtype=np.float32,
        ),
        feature_names=(
            "feature_1",
        ),
        flow_ids=(
            "flow-benign",
            "flow-attack",
        ),
        unsupported_features=(),
        metadata={},
    )

    return NetworkInferenceResult(
        extracted=extracted,
        compatibility=None,
        normalized_features=np.asarray(
            [
                [0.1],
                [0.9],
            ],
            dtype=np.float32,
        ),
        probabilities=np.asarray(
            [
                0.2,
                0.8,
            ],
            dtype=np.float32,
        ),
        predictions=np.asarray(
            [
                0,
                1,
            ],
            dtype=np.int64,
        ),
        threshold=0.5,
    )


def test_control_pipeline_processes_inference_result():
    policy = ThresholdDecisionPolicy(
        threshold=0.5
    )

    enforcer = DryRunRuleEnforcer()

    pipeline = ControlPipeline(
        policy=policy,
        enforcer=enforcer,
    )

    inference = (
        _make_inference_result()
    )

    result = pipeline.process(
        inference
    )

    assert result.inference is inference

    assert result.n_flows == 2
    assert result.block_count == 1
    assert result.applied_count == 0

    assert len(
        result.decisions
    ) == 2

    assert (
        result.decisions[0].action
        == ControlAction.NONE
    )

    assert (
        result.decisions[1].action
        == ControlAction.BLOCK
    )

    assert (
        result.decisions[0].flow_id
        == "flow-benign"
    )

    assert (
        result.decisions[1].flow_id
        == "flow-attack"
    )

    assert len(
        result.enforcement_results
    ) == 2

    assert all(
        enforcement.success
        for enforcement
        in result.enforcement_results
    )

    assert all(
        enforcement.dry_run
        for enforcement
        in result.enforcement_results
    )

    assert all(
        not enforcement.applied
        for enforcement
        in result.enforcement_results
    )


def test_control_pipeline_preserves_inference_threshold():
    pipeline = ControlPipeline(
        policy=ThresholdDecisionPolicy(
            threshold=0.5
        ),
        enforcer=DryRunRuleEnforcer(),
    )

    result = pipeline.process(
        _make_inference_result()
    )

    assert (
        result.decisions[0]
        .metadata[
            "inference_threshold"
        ]
        == 0.5
    )

    assert (
        result.decisions[1]
        .metadata[
            "inference_threshold"
        ]
        == 0.5
    )


def test_control_pipeline_rejects_probability_count_mismatch():
    inference = (
        _make_inference_result()
    )

    inference.probabilities = (
        np.asarray(
            [0.2],
            dtype=np.float32,
        )
    )

    pipeline = ControlPipeline(
        policy=ThresholdDecisionPolicy(),
        enforcer=DryRunRuleEnforcer(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Quantidade de probabilidades "
            "incompatível com flow_ids"
        ),
    ):
        pipeline.process(
            inference
        )


def test_control_pipeline_rejects_prediction_count_mismatch():
    inference = (
        _make_inference_result()
    )

    inference.predictions = (
        np.asarray(
            [0],
            dtype=np.int64,
        )
    )

    pipeline = ControlPipeline(
        policy=ThresholdDecisionPolicy(),
        enforcer=DryRunRuleEnforcer(),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Quantidade de predições "
            "incompatível com flow_ids"
        ),
    ):
        pipeline.process(
            inference
        )
