import pytest

from sender.sender import Sender
from translator.translator import AttackParams


def _params(**overrides):
    base = dict(
        target_ip="172.20.0.10",
        target_port=80,
        packets_per_second=200.0,
        packet_size=100,
        duration_seconds=0.5,
        use_tcp=True,
        tcp_flags="S",
        use_syn_flood=True,
        randomize_src_ip=True,
        randomize_src_port=True,
        window_size=4096,
        translation_valid=True,
        translation_warnings=[],
        consistency_error=0.0,
    )
    base.update(overrides)
    return AttackParams(**base)


def test_dry_run_nao_envia_e_calcula_metricas_exatas():
    sender = Sender(dry_run=True)
    result = sender.executar(_params())

    assert result.success is True
    assert result.dry_run is True
    assert result.requested_packets == 100
    assert result.packets_sent == 100
    assert result.bytes_sent == 10_000
    assert result.pps_real == pytest.approx(200.0)
    assert result.requested_pps == pytest.approx(200.0)
    assert result.duration_actual == pytest.approx(0.5)
    assert result.pps_error_ratio == pytest.approx(0.0)
    assert result.duration_error_ratio == pytest.approx(0.0)
    assert result.packet_delivery_ratio_sender == pytest.approx(1.0)


def test_to_dict_separa_solicitado_observado_e_traducao():
    sender = Sender(dry_run=True)
    params = _params(
        translation_valid=False,
        translation_warnings=["inconsistente"],
        consistency_error=1.25,
    )
    payload = sender.executar(params).to_dict()

    assert payload["requested"]["pps"] == pytest.approx(200.0)
    assert payload["sender_observed"]["pps"] == pytest.approx(200.0)
    assert payload["translation"]["valid"] is False
    assert payload["translation"]["warnings"] == ["inconsistente"]
    assert payload["translation"]["consistency_error"] == pytest.approx(1.25)


def test_target_privado_e_aceito():
    sender = Sender(dry_run=False, require_private_target=True)
    sender._validate_target(_params(target_ip="172.20.0.10"))
    sender._validate_target(_params(target_ip="127.0.0.1"))


def test_target_publico_e_bloqueado_por_padrao():
    sender = Sender(dry_run=False, require_private_target=True)

    with pytest.raises(ValueError, match="target não privado/local"):
        sender._validate_target(_params(target_ip="8.8.8.8"))


def test_target_ip_invalido_e_rejeitado():
    sender = Sender(dry_run=False)

    with pytest.raises(ValueError, match="Target IP inválido"):
        sender._validate_target(_params(target_ip="nao-e-um-ip"))


def test_dry_run_funciona_mesmo_com_traducao_marcada_invalida():
    sender = Sender(dry_run=True)
    result = sender.executar(
        _params(
            translation_valid=False,
            translation_warnings=["teste"],
        )
    )

    assert result.success is True
    assert result.dry_run is True
    assert result.to_dict()["translation"]["valid"] is False
