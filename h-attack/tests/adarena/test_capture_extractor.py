import numpy as np
import pytest

from scapy.layers.inet import (
    IP,
    TCP,
)

from scapy.packet import Raw


from adarena.builtin import (
    BASIC_FLOW_EXTRACTOR_ID,
    SCAPY_CAPTURE_ID,
    create_default_registry,
)

from adarena.core import (
    ComponentKind,
    DataRepresentation,
)

from adarena.network.base import (
    CaptureBatch,
    CapturedPacket,
)

from adarena.network.capture import (
    ScapyPacketCapture,
)

from adarena.network.extractor import (
    BasicFlowExtractor,
)


def _capture_batch():

    packets = [
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
    ]

    return CaptureBatch(
        packets=packets,

        started_at=1.0,
        ended_at=1.2,

        interface="test0",
    )


def test_registry_expoe_capture_e_extractor():

    registry = (
        create_default_registry()
    )

    capture = registry.spec(
        SCAPY_CAPTURE_ID
    )

    extractor = registry.spec(
        BASIC_FLOW_EXTRACTOR_ID
    )

    assert (
        capture.kind
        == ComponentKind.CAPTURE
    )

    assert (
        capture.output_representation
        == DataRepresentation
        .PACKET_RECORDS
    )

    assert (
        extractor.kind
        == ComponentKind.EXTRACTOR
    )

    assert (
        extractor.input_representation
        == DataRepresentation
        .PACKET_RECORDS
    )

    assert (
        extractor.output_representation
        == DataRepresentation
        .FLOW_FEATURES
    )


def test_scapy_capture_converte_pacote_sem_sniff():

    packet = (
        IP(
            src="172.20.0.2",
            dst="172.20.0.10",
        )
        /
        TCP(
            sport=50000,
            dport=80,
            flags="SA",
        )
        /
        Raw(
            load=b"abc"
        )
    )

    packet.time = 10.5

    record = (
        ScapyPacketCapture
        ._to_record(
            packet
        )
    )

    assert record is not None

    assert (
        record.src_ip
        == "172.20.0.2"
    )

    assert (
        record.dst_ip
        == "172.20.0.10"
    )

    assert (
        record.src_port
        == 50000
    )

    assert (
        record.dst_port
        == 80
    )

    assert (
        record.protocol
        == "TCP"
    )

    assert (
        record.payload_length
        == 3
    )

    assert "S" in (
        record.tcp_flags
    )

    assert "A" in (
        record.tcp_flags
    )


def test_extractor_reconstroi_fluxo_bidirecional():

    extractor = (
        BasicFlowExtractor()
    )

    result = extractor.extract(
        _capture_batch(),

        feature_names=[
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Flow Packets/s",
            "Average Packet Size",
            "SYN Flag Count",
            "ACK Flag Count",
        ],

        strict=True,
    )

    assert (
        result.X.shape
        == (
            1,
            7,
        )
    )

    assert (
        result.ready_for_model
        is True
    )

    vector = result.X[0]

    assert vector[0] == pytest.approx(
        200_000.0
    )

    assert vector[1] == pytest.approx(
        2.0
    )

    assert vector[2] == pytest.approx(
        1.0
    )

    assert vector[3] == pytest.approx(
        15.0
    )

    assert vector[4] == pytest.approx(
        80.0
    )

    assert vector[5] == pytest.approx(
        2.0
    )

    assert vector[6] == pytest.approx(
        2.0
    )


def test_extractor_nao_inventa_feature_desconhecida():

    extractor = (
        BasicFlowExtractor()
    )

    with pytest.raises(
        ValueError,
        match="ainda não suporta",
    ):
        extractor.extract(
            _capture_batch(),

            feature_names=[
                "Flow Duration",
                "Feature Impossivel",
            ],

            strict=True,
        )

    partial = extractor.extract(
        _capture_batch(),

        feature_names=[
            "Flow Duration",
            "Feature Impossivel",
        ],

        strict=False,
    )

    assert (
        partial.ready_for_model
        is False
    )

    assert (
        partial.unsupported_features
        == (
            "Feature Impossivel",
        )
    )

    assert np.isnan(
        partial.X[
            0,
            1,
        ]
    )

def test_extractor_reporta_cobertura_de_features():

    extractor = (
        BasicFlowExtractor()
    )

    report = extractor.support(
        [
            "Flow Duration",
            "Flow Packets/s",
            "SYN Flag Count",
            "Init_Win_bytes_forward",
            "Feature Impossivel",
        ]
    )

    assert (
        report.total
        == 5
    )

    assert (
        report.supported_count
        == 3
    )

    assert (
        report.unsupported_count
        == 2
    )

    assert (
        report.supported_features
        == (
            "Flow Duration",
            "Flow Packets/s",
            "SYN Flag Count",
        )
    )

    assert (
        report.unsupported_features
        == (
            "Init_Win_bytes_forward",
            "Feature Impossivel",
        )
    )

    assert (
        report.coverage
        == pytest.approx(
            3 / 5
        )
    )

    assert (
        report.complete
        is False
    )
