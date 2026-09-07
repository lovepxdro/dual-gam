import numpy as np
import pytest

from adarena.builtin import (
    CICIDS2017_RENDERER_ID,
    SCAPY_NETWORK_BACKEND_ID,
    create_default_registry,
)

from adarena.core import (
    ComponentKind,
    DataRepresentation,
)

from adarena.network.base import (
    NetworkBackend,
    Renderer,
)

from gan.preprocessing import (
    Preprocessador,
)


FEATURES = [
    "flow_duration",
    "flow_packets_per_sec",
    "flow_bytes_per_sec",
    "fwd_packet_length_mean",
    "avg_pkt_size",
    "syn_flag_count",
    "ack_flag_count",
    "fin_flag_count",
    "psh_flag_count",
    "init_fwd_win_bytes",
]


def _preprocessor():

    prep = Preprocessador()

    prep.feature_names = list(
        FEATURES
    )

    # mean = 0, scale = 1
    prep.scaler.fit(
        np.zeros(
            (
                2,
                len(FEATURES),
            ),
            dtype=np.float64,
        )
    )

    return prep


def _sample():

    return np.asarray(
        [
            2_000_000,
            1_000,
            500_000,
            500,
            500,
            10,
            1,
            0,
            0,
            4096,
        ],
        dtype=np.float64,
    )


def test_registry_expoe_renderer_e_backend():

    registry = (
        create_default_registry()
    )

    renderer = registry.spec(
        CICIDS2017_RENDERER_ID
    )

    backend = registry.spec(
        SCAPY_NETWORK_BACKEND_ID
    )

    assert (
        renderer.kind
        == ComponentKind.RENDERER
    )

    assert (
        renderer.input_representation
        == DataRepresentation.FLOW_FEATURES
    )

    assert (
        renderer.output_representation
        == DataRepresentation.ATTACK_PARAMS
    )

    assert (
        backend.kind
        == ComponentKind.NETWORK_BACKEND
    )

    assert (
        backend.input_representation
        == DataRepresentation.ATTACK_PARAMS
    )

    assert (
        backend.output_representation
        == DataRepresentation.NETWORK_RESULT
    )


def test_registry_instancia_interfaces_de_rede():

    registry = (
        create_default_registry()
    )

    renderer = registry.create(
        CICIDS2017_RENDERER_ID,

        preprocessor=(
            _preprocessor()
        ),

        target_ip=(
            "172.20.0.10"
        ),

        target_port=80,
    )

    backend = registry.create(
        SCAPY_NETWORK_BACKEND_ID,

        dry_run=True,
    )

    assert isinstance(
        renderer,
        Renderer,
    )

    assert isinstance(
        backend,
        NetworkBackend,
    )


def test_renderer_produz_parametros_validos():

    registry = (
        create_default_registry()
    )

    renderer = registry.create(
        CICIDS2017_RENDERER_ID,

        preprocessor=(
            _preprocessor()
        ),

        target_ip=(
            "172.20.0.10"
        ),

        target_port=80,
    )

    params = renderer.render(
        _sample(),
        score=0.15,
    )

    assert (
        params.translation_valid
        is True
    )

    assert (
        params.target_ip
        == "172.20.0.10"
    )

    assert (
        params.target_port
        == 80
    )

    assert (
        params.packets_per_second
        == pytest.approx(
            1000
        )
    )

    assert (
        params.packet_size
        == 500
    )

    assert (
        params.evasao_prob
        == pytest.approx(
            0.15
        )
    )


def test_backend_dry_run_executa_saida_do_renderer():

    registry = (
        create_default_registry()
    )

    renderer = registry.create(
        CICIDS2017_RENDERER_ID,

        preprocessor=(
            _preprocessor()
        ),

        target_ip=(
            "172.20.0.10"
        ),

        target_port=80,
    )

    backend = registry.create(
        SCAPY_NETWORK_BACKEND_ID,

        dry_run=True,
    )

    params = renderer.render(
        _sample()
    )

    result = backend.execute(
        params
    )

    assert (
        result.success
        is True
    )

    assert (
        result.dry_run
        is True
    )

    assert (
        result.requested_pps
        == pytest.approx(
            1000
        )
    )

    assert (
        result.packets_sent
        == 2000
    )
