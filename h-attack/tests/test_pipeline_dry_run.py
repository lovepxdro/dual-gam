import numpy as np

from gan.preprocessing import Preprocessador
from sender.sender import Sender
from translator.translator import Translator


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


def test_preprocessador_translator_sender_dry_run_integrados():
    prep = Preprocessador()
    prep.feature_names = FEATURES.copy()

    # Conjunto sintético usado apenas para estabelecer um scaler real.
    X_fit = np.vstack([
        np.zeros(len(FEATURES)),
        np.ones(len(FEATURES)),
    ])
    prep.scaler.fit(X_fit)

    raw = np.zeros(len(FEATURES), dtype=np.float64)
    idx = {name: i for i, name in enumerate(FEATURES)}
    raw[idx["flow_duration"]] = 2_000_000
    raw[idx["flow_packets_per_sec"]] = 1000
    raw[idx["flow_bytes_per_sec"]] = 500_000
    raw[idx["fwd_packet_length_mean"]] = 500
    raw[idx["avg_pkt_size"]] = 500
    raw[idx["syn_flag_count"]] = 10
    raw[idx["ack_flag_count"]] = 1
    raw[idx["init_fwd_win_bytes"]] = 4096

    normalized = prep.scaler.transform(raw.reshape(1, -1))[0]

    translator = Translator(prep, "172.20.0.10", 80)
    params = translator.traduzir(normalized, evasao_prob=0.10)
    sender = Sender(dry_run=True)
    result = sender.executar(params)

    assert params.translation_valid is True
    assert result.success is True
    assert result.dry_run is True
    assert result.packets_sent == 2000
    assert result.to_dict()["translation"]["valid"] is True
