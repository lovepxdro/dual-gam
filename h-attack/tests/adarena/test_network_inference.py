import numpy as np
import pytest

from adarena.network.base import (
    Capture,
    CaptureBatch,
    CapturedPacket,
    FlowFeatureBatch,
    NetworkBackend,
)

from adarena.network.extractor import (
    BasicFlowExtractor,
)

from adarena.network.inference import (
    BinaryPredictor,
    FeatureSchemaValidator,
    IncompatibleFeatureSchemaError,
    NetworkInferencePipeline,
    NetworkObservationPipeline,
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


class FakeCapture(
    Capture
):

    def __init__(
        self,
        batch: CaptureBatch,
        *,
        events: list[str] | None = None,
    ) -> None:

        self.batch = batch

        self.events = (
            events
            if events is not None
            else []
        )

        self.started = False
        self.stopped = False

        self.packet_limit = None

    def start(
        self,
        *,
        packet_limit: int | None = None,
    ) -> None:

        if self.started:
            raise RuntimeError(
                "Captura já iniciada"
            )

        self.started = True
        self.stopped = False

        self.packet_limit = (
            packet_limit
        )

        self.events.append(
            "capture.start"
        )

    def stop(
        self,
    ) -> CaptureBatch:

        if not self.started:
            raise RuntimeError(
                "Captura não iniciada"
            )

        self.started = False
        self.stopped = True

        self.events.append(
            "capture.stop"
        )

        return self.batch

    def capture(
        self,
        *,
        duration: float,
        packet_limit: int | None = None,
    ) -> CaptureBatch:

        self.start(
            packet_limit=packet_limit
        )

        return self.stop()


class FakeNetworkBackend(
    NetworkBackend
):

    def __init__(
        self,
        *,
        events: list[str] | None = None,
        fail: bool = False,
    ) -> None:

        self.events = (
            events
            if events is not None
            else []
        )

        self.fail = fail

        self.payloads = []

    def execute(
        self,
        payload,
    ):

        self.payloads.append(
            payload
        )

        self.events.append(
            f"backend:{payload}"
        )

        if self.fail:
            raise RuntimeError(
                "Falha simulada no backend"
            )

        return {
            "payload": payload,
            "executed": True,
        }


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


def _inference_pipeline(
    *,
    probability: float = 0.8,
):

    prep = _preprocessor(
        [
            "Flow Duration",
            "Total Fwd Packets",
        ]
    )

    predictor = FakePredictor(
        probability=probability
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

    return (
        pipeline,
        predictor,
    )


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
            "Feature Impossivel",
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
        "Feature Impossivel"
        in report
        .unsupported_features
    )

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


def test_observation_pipeline_executa_backend_durante_captura():

    events = []

    capture = FakeCapture(
        _capture_batch(),
        events=events,
    )

    backend = FakeNetworkBackend(
        events=events,
    )

    (
        inference,
        predictor,
    ) = _inference_pipeline(
        probability=0.8
    )

    pipeline = (
        NetworkObservationPipeline(
            capture=capture,
            backend=backend,
            inference=inference,
        )
    )

    result = pipeline.execute_many(
        [
            "payload-1",
            "payload-2",
        ],

        packet_limit=100,
    )

    assert events == [
        "capture.start",
        "backend:payload-1",
        "backend:payload-2",
        "capture.stop",
    ]

    assert (
        capture.packet_limit
        == 100
    )

    assert (
        capture.stopped
        is True
    )

    assert backend.payloads == [
        "payload-1",
        "payload-2",
    ]

    assert (
        result.n_backend_results
        == 2
    )

    assert (
        result.n_captured_packets
        == 3
    )

    assert (
        result.inference.n_flows
        == 1
    )

    assert (
        result.inference.attack_count
        == 1
    )

    assert (
        predictor.calls
        == 1
    )


def test_observation_pipeline_execute_payload_unico():

    events = []

    capture = FakeCapture(
        _capture_batch(),
        events=events,
    )

    backend = FakeNetworkBackend(
        events=events,
    )

    (
        inference,
        _,
    ) = _inference_pipeline(
        probability=0.2
    )

    pipeline = (
        NetworkObservationPipeline(
            capture=capture,
            backend=backend,
            inference=inference,
        )
    )

    result = pipeline.execute(
        "payload-unico"
    )

    assert backend.payloads == [
        "payload-unico"
    ]

    assert (
        result.n_backend_results
        == 1
    )

    assert (
        result.inference.n_flows
        == 1
    )

    assert (
        result.inference.attack_count
        == 0
    )

    assert (
        result.inference.benign_count
        == 1
    )

    assert events == [
        "capture.start",
        "backend:payload-unico",
        "capture.stop",
    ]


def test_observation_pipeline_encerra_captura_se_backend_falhar():

    events = []

    capture = FakeCapture(
        _capture_batch(),
        events=events,
    )

    backend = FakeNetworkBackend(
        events=events,
        fail=True,
    )

    (
        inference,
        predictor,
    ) = _inference_pipeline()

    pipeline = (
        NetworkObservationPipeline(
            capture=capture,
            backend=backend,
            inference=inference,
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Falha simulada",
    ):
        pipeline.execute(
            "payload-com-erro"
        )

    assert events == [
        "capture.start",
        "backend:payload-com-erro",
        "capture.stop",
    ]

    assert (
        capture.stopped
        is True
    )

    # A execução falhou antes da etapa de
    # inferência.
    assert (
        predictor.calls
        == 0
    )
