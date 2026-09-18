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

    Suporta dois modos:

    1. síncrono:
       capture(duration=...)

    2. concorrente:
       start()
       ...
       stop()
    """

    def __init__(
        self,
        *,
        iface: str | None = None,
        bpf_filter: str | None = None,
    ) -> None:

        self.iface = iface
        self.bpf_filter = bpf_filter

        self._sniffer = None

        self._started_at: (
            float | None
        ) = None

        self._packet_limit: (
            int | None
        ) = None

    @property
    def running(
        self,
    ) -> bool:

        if self._sniffer is None:
            return False

        return bool(
            getattr(
                self._sniffer,
                "running",
                False,
            )
        )

    def start(
        self,
        *,
        packet_limit: int | None = None,
    ) -> None:
        """
        Inicia captura em background.

        Não bloqueia a execução do chamador.
        """

        if self._sniffer is not None:
            raise RuntimeError(
                "Já existe uma sessão "
                "de captura ativa"
            )

        if (
            packet_limit is not None
            and packet_limit <= 0
        ):
            raise ValueError(
                "packet_limit deve ser > 0"
            )

        from scapy.all import (
            AsyncSniffer,
        )

        self._packet_limit = (
            packet_limit
        )

        self._started_at = (
            time.time()
        )

        self._sniffer = AsyncSniffer(
            iface=self.iface,

            filter=self.bpf_filter,

            count=(
                packet_limit
                if packet_limit
                is not None
                else 0
            ),

            store=True,
        )

        try:
            self._sniffer.start()

        except Exception:
            self._sniffer = None
            self._started_at = None
            self._packet_limit = None

            raise

        logger.info(
            "Captura iniciada "
            "em background "
            "(iface=%s, limit=%s)",
            self.iface,
            packet_limit,
        )

    def stop(
        self,
    ) -> CaptureBatch:
        """
        Encerra a sessão concorrente e retorna
        todos os pacotes observados.
        """

        if self._sniffer is None:
            raise RuntimeError(
                "Nenhuma sessão de captura "
                "foi iniciada"
            )

        sniffer = self._sniffer

        started_at = (
            self._started_at
            if self._started_at
            is not None
            else time.time()
        )

        packet_limit = (
            self._packet_limit
        )

        try:
            # Se o limite de pacotes já tiver sido
            # atingido, o AsyncSniffer pode já ter
            # terminado sozinho.
            if getattr(
                sniffer,
                "running",
                False,
            ):
                packets = (
                    sniffer.stop()
                )

            else:
                packets = getattr(
                    sniffer,
                    "results",
                    None,
                )

            if packets is None:
                packets = []

            ended_at = (
                time.time()
            )

            records = []

            for packet in packets:

                record = (
                    self._to_record(
                        packet
                    )
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

                started_at=(
                    started_at
                ),

                ended_at=(
                    ended_at
                ),

                interface=(
                    self.iface
                ),

                metadata={
                    "backend": (
                        "scapy"
                    ),

                    "mode": (
                        "async"
                    ),

                    "bpf_filter": (
                        self.bpf_filter
                    ),

                    "packet_limit": (
                        packet_limit
                    ),

                    "captured_raw": (
                        len(
                            packets
                        )
                    ),
                },
            )

        finally:
            # A sessão sempre é descartada depois
            # de stop(), mesmo se a conversão de
            # algum pacote falhar.
            self._sniffer = None
            self._started_at = None
            self._packet_limit = None

    def capture(
        self,
        *,
        duration: float,
        packet_limit: int | None = None,
    ) -> CaptureBatch:
        """
        Captura síncrona de conveniência.

        Internamente utiliza start()/stop(),
        garantindo que os dois modos compartilhem
        a mesma implementação.
        """

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

        self.start(
            packet_limit=(
                packet_limit
            )
        )

        deadline = (
            time.monotonic()
            + duration
        )

        try:
            while (
                time.monotonic()
                < deadline
            ):

                # Se count foi atingido,
                # AsyncSniffer encerra sozinho.
                if not self.running:
                    break

                remaining = (
                    deadline
                    - time.monotonic()
                )

                time.sleep(
                    min(
                        0.05,
                        max(
                            remaining,
                            0.0,
                        ),
                    )
                )

            return self.stop()

        except Exception:
            # Se alguma exceção acontecer enquanto
            # aguardamos, tentamos encerrar a captura
            # para não deixar thread pendurada.
            if self._sniffer is not None:

                try:
                    self.stop()

                except Exception:
                    logger.exception(
                        "Falha ao encerrar "
                        "captura após erro"
                    )

            raise

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
        header_length = 0

        tcp_window = None

        if TCP in packet:

            transport = (
                packet[TCP]
            )

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

            tcp_window = int(
                transport.window
            )

            data_offset = (
                transport.dataofs
                if transport.dataofs
                is not None
                else 5
            )

            header_length = int(
                data_offset
                * 4
            )

        elif UDP in packet:

            transport = (
                packet[UDP]
            )

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

            header_length = 8

        else:

            protocol = str(
                int(
                    ip.proto
                )
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
                len(
                    packet
                )
            ),

            tcp_flags=(
                tcp_flags
            ),

            payload_length=(
                payload_length
            ),

            header_length=(
                header_length
            ),

            tcp_window=(
                tcp_window
            ),
        )
