from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .base import (
    CaptureBatch,
    CapturedPacket,
    FlowExtractor,
    FlowFeatureBatch,
)


class BasicFlowExtractor(
    FlowExtractor
):
    """
    Extrator inicial de estatísticas de fluxo.

    A primeira implementação é propositalmente
    conservadora: apenas features reconstruíveis
    diretamente dos pacotes são aceitas.

    Nenhuma feature desconhecida é preenchida
    silenciosamente.
    """

    FEATURE_ALIASES = {
        # duração
        "flowduration": (
            "flow_duration"
        ),

        # quantidade de pacotes
        "totalfwdpackets": (
            "total_fwd_packets"
        ),

        "totalbackwardpackets": (
            "total_bwd_packets"
        ),

        "totalbwdpackets": (
            "total_bwd_packets"
        ),

        # taxas
        "flowpacketspersec": (
            "flow_packets_per_sec"
        ),

        "flowpacketss": (
            "flow_packets_per_sec"
        ),

        "flowbytespersec": (
            "flow_bytes_per_sec"
        ),

        "flowbytess": (
            "flow_bytes_per_sec"
        ),

        "fwdpacketspersec": (
            "fwd_packets_per_sec"
        ),

        "fwdpacketss": (
            "fwd_packets_per_sec"
        ),

        "bwdpacketspersec": (
            "bwd_packets_per_sec"
        ),

        "bwdpacketss": (
            "bwd_packets_per_sec"
        ),

        # tamanho
        "fwdpacketlengthmean": (
            "fwd_packet_length_mean"
        ),

        "bwdpacketlengthmean": (
            "bwd_packet_length_mean"
        ),

        "averagepacketsize": (
            "avg_packet_size"
        ),

        "avgpacketsize": (
            "avg_packet_size"
        ),

        "minpacketlength": (
            "packet_length_min"
        ),

        "maxpacketlength": (
            "packet_length_max"
        ),

        "packetlengthmean": (
            "packet_length_mean"
        ),

        "packetlengthstd": (
            "packet_length_std"
        ),

        # flags TCP
        "synflagcount": (
            "syn_flag_count"
        ),

        "ackflagcount": (
            "ack_flag_count"
        ),

        "finflagcount": (
            "fin_flag_count"
        ),

        "pshflagcount": (
            "psh_flag_count"
        ),

        "rstflagcount": (
            "rst_flag_count"
        ),

        "urgflagcount": (
            "urg_flag_count"
        ),

        # IAT
        "flowiatmean": (
            "flow_iat_mean"
        ),

        "flowiatstd": (
            "flow_iat_std"
        ),

        "flowiatmin": (
            "flow_iat_min"
        ),

        "flowiatmax": (
            "flow_iat_max"
        ),

        # protocolo
        "protocol": (
            "protocol"
        ),
    }

    @staticmethod
    def _normalize_name(
        name: str,
    ) -> str:

        return "".join(
            char.lower()
            for char in name
            if char.isalnum()
        )

    @classmethod
    def _canonical_feature(
        cls,
        name: str,
    ) -> str | None:

        normalized = (
            cls._normalize_name(
                name
            )
        )

        return (
            cls.FEATURE_ALIASES.get(
                normalized
            )
        )

    @staticmethod
    def _endpoint(
        packet: CapturedPacket,
        source: bool,
    ):
        if source:
            return (
                packet.src_ip,
                packet.src_port or 0,
            )

        return (
            packet.dst_ip,
            packet.dst_port or 0,
        )

    @classmethod
    def _group_key(
        cls,
        packet: CapturedPacket,
    ) -> tuple:

        src = cls._endpoint(
            packet,
            True,
        )

        dst = cls._endpoint(
            packet,
            False,
        )

        a, b = sorted(
            (
                src,
                dst,
            )
        )

        return (
            a,
            b,
            packet.protocol,
        )

    def extract(
        self,
        capture: CaptureBatch,
        *,
        feature_names: Sequence[str],
        strict: bool = True,
    ) -> FlowFeatureBatch:

        feature_names = tuple(
            str(name)
            for name in feature_names
        )

        canonical = []

        unsupported = []

        for feature in feature_names:
            resolved = (
                self._canonical_feature(
                    feature
                )
            )

            canonical.append(
                resolved
            )

            if resolved is None:
                unsupported.append(
                    feature
                )

        if (
            strict
            and unsupported
        ):
            raise ValueError(
                "FlowExtractor ainda não "
                "suporta as features: "
                + ", ".join(
                    unsupported
                )
            )

        flows = defaultdict(
            list
        )

        for packet in capture.packets:
            flows[
                self._group_key(
                    packet
                )
            ].append(
                packet
            )

        vectors = []
        flow_ids = []

        for index, packets in enumerate(
            flows.values()
        ):
            packets = sorted(
                packets,
                key=lambda p: p.timestamp,
            )

            values = (
                self._calculate_flow(
                    packets
                )
            )

            vector = []

            for resolved in canonical:
                if resolved is None:
                    vector.append(
                        np.nan
                    )

                else:
                    vector.append(
                        values[resolved]
                    )

            vectors.append(
                vector
            )

            first = packets[0]

            flow_ids.append(
                (
                    f"{index}:"
                    f"{first.src_ip}:"
                    f"{first.src_port}"
                    "->"
                    f"{first.dst_ip}:"
                    f"{first.dst_port}/"
                    f"{first.protocol}"
                )
            )

        if vectors:
            X = np.asarray(
                vectors,
                dtype=np.float32,
            )

        else:
            X = np.empty(
                (
                    0,
                    len(feature_names),
                ),
                dtype=np.float32,
            )

        return FlowFeatureBatch(
            X=X,

            feature_names=(
                feature_names
            ),

            flow_ids=tuple(
                flow_ids
            ),

            unsupported_features=tuple(
                unsupported
            ),

            metadata={
                "flows": len(
                    vectors
                ),

                "packets": len(
                    capture.packets
                ),

                "strict": strict,
            },
        )

    def _calculate_flow(
        self,
        packets: list[
            CapturedPacket
        ],
    ) -> dict[str, float]:

        first = packets[0]

        forward = []
        backward = []

        for packet in packets:
            is_forward = (
                packet.src_ip
                == first.src_ip
                and packet.dst_ip
                == first.dst_ip
                and packet.src_port
                == first.src_port
                and packet.dst_port
                == first.dst_port
            )

            if is_forward:
                forward.append(
                    packet
                )
            else:
                backward.append(
                    packet
                )

        timestamps = np.asarray(
            [
                packet.timestamp
                for packet in packets
            ],
            dtype=np.float64,
        )

        duration_seconds = max(
            float(
                timestamps[-1]
                - timestamps[0]
            ),
            0.0,
        )

        # CIC-IDS2017 representa duração
        # tipicamente em microssegundos.
        duration_us = (
            duration_seconds
            * 1_000_000.0
        )

        lengths = np.asarray(
            [
                packet.length
                for packet in packets
            ],
            dtype=np.float64,
        )

        fwd_lengths = np.asarray(
            [
                packet.length
                for packet in forward
            ],
            dtype=np.float64,
        )

        bwd_lengths = np.asarray(
            [
                packet.length
                for packet in backward
            ],
            dtype=np.float64,
        )

        if len(timestamps) > 1:
            iats = (
                np.diff(
                    timestamps
                )
                * 1_000_000.0
            )
        else:
            iats = np.asarray(
                [0.0]
            )

        def rate(
            count: float,
        ) -> float:

            if duration_seconds <= 0:
                return 0.0

            return (
                count
                / duration_seconds
            )

        def mean_or_zero(
            values,
        ) -> float:

            if len(values) == 0:
                return 0.0

            return float(
                np.mean(
                    values
                )
            )

        def flag_count(
            flag: str,
        ) -> float:

            return float(
                sum(
                    flag
                    in packet.tcp_flags
                    for packet in packets
                )
            )

        protocol_number = {
            "TCP": 6.0,
            "UDP": 17.0,
        }.get(
            first.protocol,
            0.0,
        )

        return {
            "flow_duration": (
                duration_us
            ),

            "total_fwd_packets": float(
                len(forward)
            ),

            "total_bwd_packets": float(
                len(backward)
            ),

            "flow_packets_per_sec": rate(
                len(packets)
            ),

            "flow_bytes_per_sec": rate(
                float(
                    np.sum(
                        lengths
                    )
                )
            ),

            "fwd_packets_per_sec": rate(
                len(forward)
            ),

            "bwd_packets_per_sec": rate(
                len(backward)
            ),

            "fwd_packet_length_mean": (
                mean_or_zero(
                    fwd_lengths
                )
            ),

            "bwd_packet_length_mean": (
                mean_or_zero(
                    bwd_lengths
                )
            ),

            "avg_packet_size": (
                mean_or_zero(
                    lengths
                )
            ),

            "packet_length_min": float(
                np.min(
                    lengths
                )
            ),

            "packet_length_max": float(
                np.max(
                    lengths
                )
            ),

            "packet_length_mean": float(
                np.mean(
                    lengths
                )
            ),

            "packet_length_std": float(
                np.std(
                    lengths
                )
            ),

            "syn_flag_count": (
                flag_count(
                    "S"
                )
            ),

            "ack_flag_count": (
                flag_count(
                    "A"
                )
            ),

            "fin_flag_count": (
                flag_count(
                    "F"
                )
            ),

            "psh_flag_count": (
                flag_count(
                    "P"
                )
            ),

            "rst_flag_count": (
                flag_count(
                    "R"
                )
            ),

            "urg_flag_count": (
                flag_count(
                    "U"
                )
            ),

            "flow_iat_mean": float(
                np.mean(
                    iats
                )
            ),

            "flow_iat_std": float(
                np.std(
                    iats
                )
            ),

            "flow_iat_min": float(
                np.min(
                    iats
                )
            ),

            "flow_iat_max": float(
                np.max(
                    iats
                )
            ),

            "protocol": (
                protocol_number
            ),
        }
