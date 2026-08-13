import numpy as np
import pytest

from gan.preprocessing import Preprocessador
from translator.translator import TranslationError, Translator


CANONICAL_FEATURES = [
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


def _prep(feature_names=None):
    prep = Preprocessador()
    prep.feature_names = list(feature_names or CANONICAL_FEATURES)
    # Fit em zeros => mean=0 e scale=1; assim o vetor normalizado usado
    # no teste também representa diretamente os valores brutos.
    prep.scaler.fit(np.zeros((2, len(prep.feature_names)), dtype=np.float64))
    return prep


def _vector(prep, **values):
    v = np.zeros(len(prep.feature_names), dtype=np.float64)
    normalized = {Translator._normalize_name(name): i for i, name in enumerate(prep.feature_names)}
    aliases = Translator.FEATURE_ALIASES

    for canonical, value in values.items():
        candidates = [canonical, *aliases.get(canonical, [])]
        idx = None
        for candidate in candidates:
            key = Translator._normalize_name(candidate)
            if key in normalized:
                idx = normalized[key]
                break
        if idx is None:
            raise AssertionError(f"Feature de teste não encontrada: {canonical}")
        v[idx] = value
    return v


def test_traducao_consistente_e_auditavel():
    prep = _prep()
    translator = Translator(prep, "172.20.0.10", 80)
    vetor = _vector(
        prep,
        flow_duration=2_000_000,
        flow_packets_per_sec=1_000,
        flow_bytes_per_sec=500_000,
        fwd_packet_length_mean=500,
        avg_pkt_size=500,
        syn_flag_count=10,
        ack_flag_count=1,
        init_fwd_win_bytes=4096,
    )

    params = translator.traduzir(vetor, evasao_prob=0.12)

    assert params.translation_valid is True
    assert params.translation_warnings == []
    assert params.packets_per_second == pytest.approx(1000)
    assert params.packet_size == 500
    assert params.duration_seconds == pytest.approx(2.0)
    assert params.use_tcp is True
    assert params.use_syn_flood is True
    assert "S" in params.tcp_flags
    assert params.window_size == 4096
    assert params.evasao_prob == pytest.approx(0.12)
    assert params.consistency_error == pytest.approx(0.0)


def test_valores_negativos_sao_projetados_sem_abs():
    prep = _prep()
    translator = Translator(prep, "172.20.0.10")
    vetor = _vector(
        prep,
        flow_duration=-2_000_000,
        flow_packets_per_sec=-50,
        flow_bytes_per_sec=-100,
        fwd_packet_length_mean=-200,
        avg_pkt_size=-200,
        init_fwd_win_bytes=-1,
    )

    params = translator.traduzir(vetor)

    assert params.packets_per_second == 1.0
    assert params.packet_size == 40
    assert params.duration_seconds == 0.5
    assert params.window_size == 0
    assert any("valor negativo projetado para 0" in w for w in params.translation_warnings)


def test_nan_e_infinito_sao_rejeitados():
    prep = _prep()
    translator = Translator(prep, "172.20.0.10")

    vetor_nan = np.zeros(len(prep.feature_names), dtype=np.float64)
    vetor_nan[0] = np.nan
    with pytest.raises(TranslationError, match="NaN ou infinito"):
        translator.traduzir(vetor_nan)

    vetor_inf = np.zeros(len(prep.feature_names), dtype=np.float64)
    vetor_inf[0] = np.inf
    with pytest.raises(TranslationError, match="NaN ou infinito"):
        translator.traduzir(vetor_inf)


def test_consistencia_pps_bps_tamanho_marca_vetor_invalido():
    prep = _prep()
    translator = Translator(prep, "172.20.0.10", consistency_tolerance=0.20)
    vetor = _vector(
        prep,
        flow_duration=1_000_000,
        flow_packets_per_sec=1_000,
        flow_bytes_per_sec=10_000,
        fwd_packet_length_mean=500,
        avg_pkt_size=500,
    )

    params = translator.traduzir(vetor)

    assert params.translation_valid is False
    assert params.consistency_error is not None
    assert params.consistency_error > 0.20
    assert any("inconsistência PPS/BPS/tamanho" in w for w in params.translation_warnings)


def test_traduzir_batch_only_valid_filtra_inconsistentes():
    prep = _prep()
    translator = Translator(prep, "172.20.0.10", consistency_tolerance=0.20)

    valido = _vector(
        prep,
        flow_duration=1_000_000,
        flow_packets_per_sec=1000,
        flow_bytes_per_sec=500_000,
        fwd_packet_length_mean=500,
        avg_pkt_size=500,
    )
    invalido = _vector(
        prep,
        flow_duration=1_000_000,
        flow_packets_per_sec=1000,
        flow_bytes_per_sec=1_000,
        fwd_packet_length_mean=500,
        avg_pkt_size=500,
    )

    todos = translator.traduzir_batch(np.stack([valido, invalido]), only_valid=False)
    filtrados = translator.traduzir_batch(np.stack([valido, invalido]), only_valid=True)

    assert len(todos) == 2
    assert [p.translation_valid for p in todos] == [True, False]
    assert len(filtrados) == 1
    assert filtrados[0].translation_valid is True


def test_aliases_de_nomes_de_features_sao_resolvidos():
    aliases = [
        "Flow Duration",
        "Flow Packets/s",
        "Flow Bytes/s",
        "Fwd Packet Length Mean",
        "Average Packet Size",
        "SYN Flag Count",
        "ACK Flag Count",
        "FIN Flag Count",
        "PSH Flag Count",
        "Init Win Bytes Forward",
    ]
    prep = _prep(aliases)
    translator = Translator(prep, "172.20.0.10")

    vetor = _vector(
        prep,
        flow_duration=1_000_000,
        flow_packets_per_sec=100,
        flow_bytes_per_sec=20_000,
        fwd_packet_length_mean=200,
        avg_pkt_size=200,
    )
    params = translator.traduzir(vetor)

    assert params.packets_per_second == pytest.approx(100)
    assert params.packet_size == 200
    assert params.translation_valid is True


def test_dimensao_incompativel_e_rejeitada():
    prep = _prep()
    translator = Translator(prep, "172.20.0.10")

    with pytest.raises(TranslationError, match="Dimensão do vetor incompatível"):
        translator.traduzir(np.zeros(len(prep.feature_names) - 1))
