from __future__ import annotations

import logging
import time

from .base import (
    Capture,
    CaptureBatch,
    CapturedPacket,
)


logger = logging.getLogger(
    __name__
)


class ScapyPacketCapture(Capture):
    """
    Captura passiva utilizada apenas no ambiente
    experimental da ADArena.

    Converte pacotes Scapy imediatamente para
    CapturedPacket, evitando que o restante do Core
    dependa de Scapy.
    """

    def __init__(
        self,
        *,
        iface: str | None = None,
        bpf_filter: str | None = None,
    ) -> None:

        self.iface = iface
        self.bpf_filter = bpf_filter

    def capture(
        self,
        *,
        duration: float,
        packet_limit: int | None = None,
    ) -> CaptureBatch:

        if duration <= 0:
            raise ValueError(
                "duration deve ser > 0"
            )

        if (
            packet_limit is not None
            and packet_limit <= 0
        ):
            raise ValueError(
                "packet_limit deve ser > 0"
            )

        from scapy.all import sniff

        started_at = time.time()

        packets = sniff(
            iface=self.iface,

            filter=self.bpf_filter,

            timeout=duration,

            count=(
                packet_limit
                if packet_limit
                is not None
                else 0
            ),

            store=True,
        )

        ended_at = time.time()

        records = []

        for packet in packets:
            record = self._to_record(
                packet
            )

            if record is not None:
                records.append(
                    record
                )

        logger.info(
            "Captura concluída: "
            "%d pacotes IP em %.3fs",
            len(records),
            ended_at - started_at,
        )

        return CaptureBatch(
            packets=records,

            started_at=started_at,
            ended_at=ended_at,

            interface=self.iface,

            metadata={
                "backend": "scapy",
                "bpf_filter": (
                    self.bpf_filter
                ),
                "captured_raw": (
                    len(packets)
                ),
            },
        )

    @staticmethod
    def _to_record(
        packet,
    ) -> CapturedPacket | None:

        from scapy.layers.inet import (
            IP,
            TCP,
            UDP,
        )

        if IP not in packet:
            return None

        ip = packet[IP]

        src_port = None
        dst_port = None

        tcp_flags = ""

        payload_length = 0

        if TCP in packet:
            transport = packet[TCP]

            protocol = "TCP"

            src_port = int(
                transport.sport
            )

            dst_port = int(
                transport.dport
            )

            tcp_flags = str(
                transport.flags
            )

            payload_length = len(
                bytes(
                    transport.payload
                )
            )

        elif UDP in packet:
            transport = packet[UDP]

            protocol = "UDP"

            src_port = int(
                transport.sport
            )

            dst_port = int(
                transport.dport
            )

            payload_length = len(
                bytes(
                    transport.payload
                )
            )

        else:
            protocol = str(
                int(ip.proto)
            )

        return CapturedPacket(
            timestamp=float(
                packet.time
            ),

            src_ip=str(
                ip.src
            ),

            dst_ip=str(
                ip.dst
            ),

            src_port=src_port,
            dst_port=dst_port,

            protocol=protocol,

            length=int(
                len(packet)
            ),

            tcp_flags=tcp_flags,

            payload_length=(
                payload_length
            ),
        )
