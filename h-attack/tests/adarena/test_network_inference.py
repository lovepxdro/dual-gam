import numpy as np
import pytest

from adarena.network.base import (
    CaptureBatch,
    CapturedPacket,
    FlowFeatureBatch,
)

from adarena.network.extractor import (
    BasicFlowExtractor,
)

from adarena.network.inference import (
    BinaryPredictor,
    FeatureSchemaValidator,
    IncompatibleFeatureSchemaError,
    NetworkInferencePipeline,
)

from gan.preprocessing import (
    Preprocessador,
)


class FakePredictor(
    BinaryPredictor
):

    def __init__(
        self,
        probability: float = 0.8,
    ):
        self.probability = (
            probability
        )

        self.calls = 0

        self.last_X = None

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:

        self.calls += 1

        self.last_X = (
            np.asarray(X)
            .copy()
        )

        return np.full(
            len(X),

            self.probability,

            dtype=np.float32,
        )


def _capture_batch():

    return CaptureBatch(
        packets=[
            CapturedPacket(
                timestamp=1.0,

                src_ip="172.20.0.2",
                dst_ip="172.20.0.10",

                src_port=50000,
                dst_port=80,

                protocol="TCP",

                length=100,

                tcp_flags="S",
            ),

            CapturedPacket(
                timestamp=1.1,

                src_ip="172.20.0.10",
                dst_ip="172.20.0.2",

                src_port=80,
                dst_port=50000,

                protocol="TCP",

                length=60,

                tcp_flags="SA",
            ),

            CapturedPacket(
                timestamp=1.2,

                src_ip="172.20.0.2",
                dst_ip="172.20.0.10",

                src_port=50000,
                dst_port=80,

                protocol="TCP",

                length=80,

                tcp_flags="A",
            ),
        ],

        started_at=1.0,
        ended_at=1.2,

        interface="test0",
    )


def _preprocessor(
    feature_names,
):

    prep = Preprocessador()

    prep.feature_names = list(
        feature_names
    )

    n_features = len(
        feature_names
    )

    # Ajusta o scaler sem depender
    # de dataset externo.
    prep.scaler.fit(
        np.asarray(
            [
                [
                    0.0
                    for _ in range(
                        n_features
                    )
                ],

                [
                    1.0
                    for _ in range(
                        n_features
                    )
                ],
            ],

            dtype=np.float64,
        )
    )

    return prep


def test_schema_validator_aceita_schema_exato():

    batch = FlowFeatureBatch(
        X=np.asarray(
            [
                [
                    1.0,
                    2.0,
                ]
            ],
            dtype=np.float32,
        ),

        feature_names=(
            "f0",
            "f1",
        ),

        flow_ids=(
            "flow-0",
        ),
    )

    validator = (
        FeatureSchemaValidator()
    )

    report = validator.validate(
        expected_features=(
            "f0",
            "f1",
        ),

        batch=batch,
    )

    assert (
        report.ready_for_model
        is True
    )

    assert (
        report.missing_features
        == ()
    )

    assert (
        report.extra_features
        == ()
    )

    assert (
        report.unsupported_features
        == ()
    )

    assert (
        report.order_matches
        is True
    )


def test_schema_validator_rejeita_ordem_diferente():

    batch = FlowFeatureBatch(
        X=np.asarray(
            [
                [
                    1.0,
                    2.0,
                ]
            ],
            dtype=np.float32,
        ),

        feature_names=(
            "f1",
            "f0",
        ),

        flow_ids=(
            "flow-0",
        ),
    )

    report = (
        FeatureSchemaValidator()
        .validate(
            expected_features=(
                "f0",
                "f1",
            ),

            batch=batch,
        )
    )

    assert (
        report.ready_for_model
        is False
    )

    assert (
        report.order_matches
        is False
    )


def test_pipeline_bloqueia_feature_nao_reconstruida():

    prep = _preprocessor(
        [
            "Flow Duration",

            # BasicFlowExtractor ainda
            # não reconstrói esta feature.
            "Init_Win_bytes_forward",
        ]
    )

    predictor = (
        FakePredictor()
    )

    pipeline = (
        NetworkInferencePipeline(
            extractor=(
                BasicFlowExtractor()
            ),

            preprocessor=prep,

            predictor=predictor,
        )
    )

    with pytest.raises(
        IncompatibleFeatureSchemaError
    ) as exc:
        pipeline.infer(
            _capture_batch()
        )

    report = (
        exc.value.report
    )

    assert (
        report.ready_for_model
        is False
    )

    assert (
        "Init_Win_bytes_forward"
        in report
        .unsupported_features
    )

    # O ponto principal:
    # o Defensor nunca foi chamado.
    assert (
        predictor.calls
        == 0
    )


def test_pipeline_normaliza_e_classifica():

    prep = _preprocessor(
        [
            "Flow Duration",
            "Total Fwd Packets",
        ]
    )

    predictor = FakePredictor(
        probability=0.8
    )

    pipeline = (
        NetworkInferencePipeline(
            extractor=(
                BasicFlowExtractor()
            ),

            preprocessor=prep,

            predictor=predictor,

            threshold=0.5,
        )
    )

    result = pipeline.infer(
        _capture_batch()
    )

    assert (
        result.n_flows
        == 1
    )

    assert (
        result.attack_count
        == 1
    )

    assert (
        result.benign_count
        == 0
    )

    assert (
        result.predictions
        .tolist()
        == [
            1
        ]
    )

    assert (
        result.probabilities
        .tolist()
        == pytest.approx(
            [
                0.8
            ]
        )
    )

    assert (
        result
        .normalized_features
        .shape
        == (
            1,
            2,
        )
    )

    assert np.isfinite(
        result
        .normalized_features
    ).all()

    assert (
        predictor.calls
        == 1
    )

    assert (
        predictor
        .last_X
        .shape
        == (
            1,
            2,
        )
    )
