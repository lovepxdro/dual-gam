from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .base import (
    CaptureBatch,
    CapturedPacket,
    FeatureSupportReport,
    FlowExtractor,
    FlowFeatureBatch,
)


class BasicFlowExtractor(
    FlowExtractor
):
    """
    Extrator de estatísticas de fluxo compatíveis
    com o schema utilizado pelo CIC-IDS2017.

    Features desconhecidas nunca são preenchidas
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

        # tamanho por direção
        "fwdpacketslengthtotal": (
            "fwd_packets_length_total"
        ),

        "bwdpacketslengthtotal": (
            "bwd_packets_length_total"
        ),

        "fwdpacketlengthmax": (
            "fwd_packet_length_max"
        ),

        "fwdpacketlengthmin": (
            "fwd_packet_length_min"
        ),

        "fwdpacketlengthstd": (
            "fwd_packet_length_std"
        ),

        "bwdpacketlengthmax": (
            "bwd_packet_length_max"
        ),

        "bwdpacketlengthmin": (
            "bwd_packet_length_min"
        ),

        "bwdpacketlengthstd": (
            "bwd_packet_length_std"
        ),

        # IAT por direção
        "fwdiattotal": (
            "fwd_iat_total"
        ),

        "fwdiatmean": (
            "fwd_iat_mean"
        ),

        "fwdiatstd": (
            "fwd_iat_std"
        ),

        "fwdiatmax": (
            "fwd_iat_max"
        ),

        "fwdiatmin": (
            "fwd_iat_min"
        ),

        "bwdiattotal": (
            "bwd_iat_total"
        ),

        "bwdiatmean": (
            "bwd_iat_mean"
        ),

        "bwdiatstd": (
            "bwd_iat_std"
        ),

        "bwdiatmax": (
            "bwd_iat_max"
        ),

        "bwdiatmin": (
            "bwd_iat_min"
        ),

        # flags por direção
        "fwdpshflags": (
            "fwd_psh_flags"
        ),

        "bwdpshflags": (
            "bwd_psh_flags"
        ),

        "fwdurgflags": (
            "fwd_urg_flags"
        ),

        "bwdurgflags": (
            "bwd_urg_flags"
        ),

        # tamanho global
        "packetlengthmin": (
            "packet_length_min"
        ),

        "packetlengthmax": (
            "packet_length_max"
        ),

        "packetlengthvariance": (
            "packet_length_variance"
        ),

        # flags adicionais
        "cweflagcount": (
            "cwr_flag_count"
        ),

        "cwrflagcount": (
            "cwr_flag_count"
        ),

        "eceflagcount": (
            "ece_flag_count"
        ),

        # razão e segmentos
        "downupratio": (
            "down_up_ratio"
        ),

        "avgfwdsegmentsize": (
            "avg_fwd_segment_size"
        ),

        "avgbwdsegmentsize": (
            "avg_bwd_segment_size"
        ),

        # payload forward
        "fwdactdatapackets": (
            "fwd_act_data_packets"
        ),

        "fwdactdatapkts": (
            "fwd_act_data_packets"
        ),

        "actdatapktfwd": (
            "fwd_act_data_packets"
        ),

        # headers
        "fwdheaderlength": (
            "fwd_header_length"
        ),

        "bwdheaderlength": (
            "bwd_header_length"
        ),

        # janela TCP inicial
        "initfwdwinbytes": (
            "init_fwd_win_bytes"
        ),

        "initbwdwinbytes": (
            "init_bwd_win_bytes"
        ),

        "initwinbytesforward": (
            "init_fwd_win_bytes"
        ),

        "initwinbytesbackward": (
            "init_bwd_win_bytes"
        ),

        # tamanho mínimo do segmento forward
        "fwdsegsizemin": (
            "fwd_seg_size_min"
        ),

        "minsegsizeforward": (
            "fwd_seg_size_min"
        ),

        # bulk forward
        "fwdavgbytesbulk": (
            "fwd_avg_bytes_bulk"
        ),

        "fwdavgpacketsbulk": (
            "fwd_avg_packets_bulk"
        ),

        "fwdavgbulkrate": (
            "fwd_avg_bulk_rate"
        ),

        # bulk backward
        "bwdavgbytesbulk": (
            "bwd_avg_bytes_bulk"
        ),

        "bwdavgpacketsbulk": (
            "bwd_avg_packets_bulk"
        ),

        "bwdavgbulkrate": (
            "bwd_avg_bulk_rate"
        ),

        # subflows
        "subflowfwdpackets": (
            "subflow_fwd_packets"
        ),

        "subflowfwdbytes": (
            "subflow_fwd_bytes"
        ),

        "subflowbwdpackets": (
            "subflow_bwd_packets"
        ),

        "subflowbwdbytes": (
            "subflow_bwd_bytes"
        ),

        # active
        "activemean": (
            "active_mean"
        ),

        "activestd": (
            "active_std"
        ),

        "activemax": (
            "active_max"
        ),

        "activemin": (
            "active_min"
        ),

        # idle
        "idlemean": (
            "idle_mean"
        ),

        "idlestd": (
            "idle_std"
        ),

        "idlemax": (
            "idle_max"
        ),

        "idlemin": (
            "idle_min"
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

    def support(
        self,
        feature_names: Sequence[str],
    ) -> FeatureSupportReport:

        requested = tuple(
            str(feature)
            for feature in feature_names
        )

        supported = []
        unsupported = []

        for feature in requested:

            if (
                self._canonical_feature(
                    feature
                )
                is None
            ):
                unsupported.append(
                    feature
                )

            else:
                supported.append(
                    feature
                )

        return FeatureSupportReport(
            requested_features=requested,

            supported_features=tuple(
                supported
            ),

            unsupported_features=tuple(
                unsupported
            ),
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

        support = self.support(
            feature_names
        )

        unsupported = list(
            support.unsupported_features
        )

        canonical = [
            self._canonical_feature(
                feature
            )
            for feature in feature_names
        ]

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

        duration_us = (
            duration_seconds
            * 1_000_000.0
        )

        # CICFlowMeter trabalha com payload para
        # estatísticas de comprimento/bytes.
        lengths = np.asarray(
            [
                packet.payload_length
                for packet in packets
            ],
            dtype=np.float64,
        )

        fwd_lengths = np.asarray(
            [
                packet.payload_length
                for packet in forward
            ],
            dtype=np.float64,
        )

        bwd_lengths = np.asarray(
            [
                packet.payload_length
                for packet in backward
            ],
            dtype=np.float64,
        )

        def directional_iats(
            direction_packets,
        ) -> np.ndarray:

            if (
                len(direction_packets)
                <= 1
            ):
                return np.asarray(
                    [],
                    dtype=np.float64,
                )

            ts = np.asarray(
                [
                    packet.timestamp
                    for packet
                    in direction_packets
                ],
                dtype=np.float64,
            )

            return (
                np.diff(
                    ts
                )
                * 1_000_000.0
            )

        if len(timestamps) > 1:

            flow_iats = (
                np.diff(
                    timestamps
                )
                * 1_000_000.0
            )

        else:

            flow_iats = np.asarray(
                [],
                dtype=np.float64,
            )

        fwd_iats = directional_iats(
            forward
        )

        bwd_iats = directional_iats(
            backward
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

        def min_or_zero(
            values,
        ) -> float:

            if len(values) == 0:
                return 0.0

            return float(
                np.min(
                    values
                )
            )

        def max_or_zero(
            values,
        ) -> float:

            if len(values) == 0:
                return 0.0

            return float(
                np.max(
                    values
                )
            )

        def sum_or_zero(
            values,
        ) -> float:

            if len(values) == 0:
                return 0.0

            return float(
                np.sum(
                    values
                )
            )

        def std_or_zero(
            values,
        ) -> float:

            if len(values) <= 1:
                return 0.0

            return float(
                np.std(
                    values,
                    ddof=1,
                )
            )

        def variance_or_zero(
            values,
        ) -> float:

            if len(values) <= 1:
                return 0.0

            return float(
                np.var(
                    values,
                    ddof=1,
                )
            )

        def rate(
            value: float,
        ) -> float:

            if duration_seconds <= 0:
                return 0.0

            return (
                value
                / duration_seconds
            )

        def flag_count(
            flag: str,
            selected_packets=None,
        ) -> float:

            source = (
                packets
                if selected_packets is None
                else selected_packets
            )

            return float(
                sum(
                    flag
                    in packet.tcp_flags
                    for packet in source
                )
            )

        def initial_tcp_window(
            direction_packets,
        ) -> float:

            for packet in direction_packets:

                if (
                    packet.tcp_window
                    is not None
                ):
                    return float(
                        packet.tcp_window
                    )

            return 0.0

        protocol_number = {
            "TCP": 6.0,
            "UDP": 17.0,
        }.get(
            first.protocol,
            0.0,
        )

        fwd_count = len(
            forward
        )

        bwd_count = len(
            backward
        )

        down_up_ratio = (
            float(
                bwd_count
                / fwd_count
            )
            if fwd_count > 0
            else 0.0
        )

        fwd_act_data_packets = float(
            sum(
                packet.payload_length
                >= 1
                for packet in forward
            )
        )

        # -----------------------------------------
        # Headers / TCP
        # -----------------------------------------

        fwd_header_length = float(
            sum(
                packet.header_length
                for packet in forward
            )
        )

        bwd_header_length = float(
            sum(
                packet.header_length
                for packet in backward
            )
        )

        init_fwd_win_bytes = (
            initial_tcp_window(
                forward
            )
        )

        init_bwd_win_bytes = (
            initial_tcp_window(
                backward
            )
        )

        fwd_header_sizes = [
            packet.header_length
            for packet in forward
            if packet.header_length > 0
        ]

        fwd_seg_size_min = (
            float(
                min(
                    fwd_header_sizes
                )
            )
            if fwd_header_sizes
            else 0.0
        )

        # -----------------------------------------
        # Subflows
        # -----------------------------------------

        SUBFLOW_GAP_SECONDS = 1.0

        subflow_count = 0
        previous_timestamp = None

        for packet in packets:

            if previous_timestamp is None:

                previous_timestamp = (
                    packet.timestamp
                )

                continue

            gap = (
                packet.timestamp
                - previous_timestamp
            )

            if (
                gap
                > SUBFLOW_GAP_SECONDS
            ):
                subflow_count += 1

            previous_timestamp = (
                packet.timestamp
            )

        if subflow_count > 0:

            subflow_fwd_packets = float(
                fwd_count
                // subflow_count
            )

            subflow_fwd_bytes = float(
                int(
                    np.sum(
                        fwd_lengths
                    )
                )
                // subflow_count
            )

            subflow_bwd_packets = float(
                bwd_count
                // subflow_count
            )

            subflow_bwd_bytes = float(
                int(
                    np.sum(
                        bwd_lengths
                    )
                )
                // subflow_count
            )

        else:

            subflow_fwd_packets = 0.0
            subflow_fwd_bytes = 0.0

            subflow_bwd_packets = 0.0
            subflow_bwd_bytes = 0.0

        # -----------------------------------------
        # Active / Idle
        # -----------------------------------------

        ACTIVITY_TIMEOUT_SECONDS = 5.0

        active_periods = []
        idle_periods = []

        start_active = (
            packets[0].timestamp
        )

        end_active = (
            packets[0].timestamp
        )

        for packet in packets[1:]:

            current = (
                packet.timestamp
            )

            gap = (
                current
                - end_active
            )

            if (
                gap
                > ACTIVITY_TIMEOUT_SECONDS
            ):

                active_duration = (
                    end_active
                    - start_active
                )

                if active_duration > 0:

                    active_periods.append(
                        active_duration
                        * 1_000_000.0
                    )

                idle_periods.append(
                    gap
                    * 1_000_000.0
                )

                start_active = (
                    current
                )

                end_active = (
                    current
                )

            else:

                end_active = (
                    current
                )

        final_active_duration = (
            end_active
            - start_active
        )

        if final_active_duration > 0:

            active_periods.append(
                final_active_duration
                * 1_000_000.0
            )

        active_values = np.asarray(
            active_periods,
            dtype=np.float64,
        )

        idle_values = np.asarray(
            idle_periods,
            dtype=np.float64,
        )

        active_mean = (
            mean_or_zero(
                active_values
            )
        )

        active_std = (
            std_or_zero(
                active_values
            )
        )

        active_max = (
            max_or_zero(
                active_values
            )
        )

        active_min = (
            min_or_zero(
                active_values
            )
        )

        idle_mean = (
            mean_or_zero(
                idle_values
            )
        )

        idle_std = (
            std_or_zero(
                idle_values
            )
        )

        idle_max = (
            max_or_zero(
                idle_values
            )
        )

        idle_min = (
            min_or_zero(
                idle_values
            )
        )

        # -----------------------------------------
        # Bulk
        # -----------------------------------------

        BULK_TIMEOUT_SECONDS = 1.0
        BULK_MIN_PACKETS = 4

        def new_bulk_state():

            return {
                "state_count": 0,
                "packet_count": 0,
                "size_total": 0,
                "duration": 0.0,

                "candidate_count": 0,
                "candidate_size": 0,
                "candidate_start": None,

                "last_timestamp": None,
            }

        fwd_bulk = (
            new_bulk_state()
        )

        bwd_bulk = (
            new_bulk_state()
        )

        def update_bulk(
            state,
            *,
            packet,
            opposite_last_timestamp,
        ) -> None:

            # Tráfego da direção oposta ocorrido
            # após o início do candidato invalida
            # o candidato atual.
            if (
                state[
                    "candidate_start"
                ]
                is not None
                and opposite_last_timestamp
                is not None
                and opposite_last_timestamp
                > state[
                    "candidate_start"
                ]
            ):
                state[
                    "candidate_start"
                ] = None

            size = int(
                packet.payload_length
            )

            # Apenas payload positivo participa
            # da formação de bulk.
            if size <= 0:
                return

            timestamp = float(
                packet.timestamp
            )

            # Primeiro pacote do candidato.
            if (
                state[
                    "candidate_start"
                ]
                is None
            ):

                state[
                    "candidate_start"
                ] = timestamp

                state[
                    "candidate_count"
                ] = 1

                state[
                    "candidate_size"
                ] = size

                state[
                    "last_timestamp"
                ] = timestamp

                return

            last_timestamp = (
                state[
                    "last_timestamp"
                ]
            )

            # Mais de 1 segundo sem pacote
            # reinicia o candidato.
            if (
                last_timestamp
                is not None
                and (
                    timestamp
                    - last_timestamp
                )
                > BULK_TIMEOUT_SECONDS
            ):

                state[
                    "candidate_start"
                ] = timestamp

                state[
                    "candidate_count"
                ] = 1

                state[
                    "candidate_size"
                ] = size

                state[
                    "last_timestamp"
                ] = timestamp

                return

            state[
                "candidate_count"
            ] += 1

            state[
                "candidate_size"
            ] += size

            candidate_count = (
                state[
                    "candidate_count"
                ]
            )

            # Quarto pacote confirma um bulk.
            if (
                candidate_count
                == BULK_MIN_PACKETS
            ):

                state[
                    "state_count"
                ] += 1

                state[
                    "packet_count"
                ] += candidate_count

                state[
                    "size_total"
                ] += state[
                    "candidate_size"
                ]

                state[
                    "duration"
                ] += (
                    timestamp
                    - state[
                        "candidate_start"
                    ]
                )

            # Pacotes posteriores continuam
            # o bulk já confirmado.
            elif (
                candidate_count
                > BULK_MIN_PACKETS
            ):

                state[
                    "packet_count"
                ] += 1

                state[
                    "size_total"
                ] += size

                if (
                    last_timestamp
                    is not None
                ):

                    state[
                        "duration"
                    ] += (
                        timestamp
                        - last_timestamp
                    )

            state[
                "last_timestamp"
            ] = timestamp

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

                update_bulk(
                    fwd_bulk,

                    packet=packet,

                    opposite_last_timestamp=(
                        bwd_bulk[
                            "last_timestamp"
                        ]
                    ),
                )

            else:

                update_bulk(
                    bwd_bulk,

                    packet=packet,

                    opposite_last_timestamp=(
                        fwd_bulk[
                            "last_timestamp"
                        ]
                    ),
                )

        def bulk_features(
            state,
        ) -> tuple[
            float,
            float,
            float,
        ]:

            state_count = int(
                state[
                    "state_count"
                ]
            )

            if state_count > 0:

                avg_bytes = float(
                    int(
                        state[
                            "size_total"
                        ]
                    )
                    // state_count
                )

                avg_packets = float(
                    int(
                        state[
                            "packet_count"
                        ]
                    )
                    // state_count
                )

            else:

                avg_bytes = 0.0
                avg_packets = 0.0

            duration = float(
                state[
                    "duration"
                ]
            )

            if duration > 0:

                avg_rate = float(
                    int(
                        float(
                            state[
                                "size_total"
                            ]
                        )
                        / duration
                    )
                )

            else:

                avg_rate = 0.0

            return (
                avg_bytes,
                avg_packets,
                avg_rate,
            )

        (
            fwd_avg_bytes_bulk,
            fwd_avg_packets_bulk,
            fwd_avg_bulk_rate,
        ) = bulk_features(
            fwd_bulk
        )

        (
            bwd_avg_bytes_bulk,
            bwd_avg_packets_bulk,
            bwd_avg_bulk_rate,
        ) = bulk_features(
            bwd_bulk
        )

        return {
            "protocol": (
                protocol_number
            ),

            "flow_duration": (
                duration_us
            ),

            "total_fwd_packets": float(
                fwd_count
            ),

            "total_bwd_packets": float(
                bwd_count
            ),

            # Comprimentos / bytes
            "fwd_packets_length_total": (
                sum_or_zero(
                    fwd_lengths
                )
            ),

            "bwd_packets_length_total": (
                sum_or_zero(
                    bwd_lengths
                )
            ),

            "fwd_packet_length_max": (
                max_or_zero(
                    fwd_lengths
                )
            ),

            "fwd_packet_length_min": (
                min_or_zero(
                    fwd_lengths
                )
            ),

            "fwd_packet_length_mean": (
                mean_or_zero(
                    fwd_lengths
                )
            ),

            "fwd_packet_length_std": (
                std_or_zero(
                    fwd_lengths
                )
            ),

            "bwd_packet_length_max": (
                max_or_zero(
                    bwd_lengths
                )
            ),

            "bwd_packet_length_min": (
                min_or_zero(
                    bwd_lengths
                )
            ),

            "bwd_packet_length_mean": (
                mean_or_zero(
                    bwd_lengths
                )
            ),

            "bwd_packet_length_std": (
                std_or_zero(
                    bwd_lengths
                )
            ),

            # Taxas
            "flow_packets_per_sec": (
                rate(
                    float(
                        len(
                            packets
                        )
                    )
                )
            ),

            "flow_bytes_per_sec": (
                rate(
                    sum_or_zero(
                        lengths
                    )
                )
            ),

            "fwd_packets_per_sec": (
                rate(
                    float(
                        fwd_count
                    )
                )
            ),

            "bwd_packets_per_sec": (
                rate(
                    float(
                        bwd_count
                    )
                )
            ),

            # Flow IAT
            "flow_iat_mean": (
                mean_or_zero(
                    flow_iats
                )
            ),

            "flow_iat_std": (
                std_or_zero(
                    flow_iats
                )
            ),

            "flow_iat_min": (
                min_or_zero(
                    flow_iats
                )
            ),

            "flow_iat_max": (
                max_or_zero(
                    flow_iats
                )
            ),

            # Forward IAT
            "fwd_iat_total": (
                sum_or_zero(
                    fwd_iats
                )
            ),

            "fwd_iat_mean": (
                mean_or_zero(
                    fwd_iats
                )
            ),

            "fwd_iat_std": (
                std_or_zero(
                    fwd_iats
                )
            ),

            "fwd_iat_max": (
                max_or_zero(
                    fwd_iats
                )
            ),

            "fwd_iat_min": (
                min_or_zero(
                    fwd_iats
                )
            ),

            # Backward IAT
            "bwd_iat_total": (
                sum_or_zero(
                    bwd_iats
                )
            ),

            "bwd_iat_mean": (
                mean_or_zero(
                    bwd_iats
                )
            ),

            "bwd_iat_std": (
                std_or_zero(
                    bwd_iats
                )
            ),

            "bwd_iat_max": (
                max_or_zero(
                    bwd_iats
                )
            ),

            "bwd_iat_min": (
                min_or_zero(
                    bwd_iats
                )
            ),

            # Flags direcionais
            "fwd_psh_flags": (
                flag_count(
                    "P",
                    forward,
                )
            ),

            "bwd_psh_flags": (
                flag_count(
                    "P",
                    backward,
                )
            ),

            "fwd_urg_flags": (
                flag_count(
                    "U",
                    forward,
                )
            ),

            "bwd_urg_flags": (
                flag_count(
                    "U",
                    backward,
                )
            ),

            # Estatísticas globais
            "packet_length_min": (
                min_or_zero(
                    lengths
                )
            ),

            "packet_length_max": (
                max_or_zero(
                    lengths
                )
            ),

            "packet_length_mean": (
                mean_or_zero(
                    lengths
                )
            ),

            "packet_length_std": (
                std_or_zero(
                    lengths
                )
            ),

            "packet_length_variance": (
                variance_or_zero(
                    lengths
                )
            ),

            "avg_packet_size": (
                mean_or_zero(
                    lengths
                )
            ),

            # Flags globais
            "fin_flag_count": (
                flag_count(
                    "F"
                )
            ),

            "syn_flag_count": (
                flag_count(
                    "S"
                )
            ),

            "rst_flag_count": (
                flag_count(
                    "R"
                )
            ),

            "psh_flag_count": (
                flag_count(
                    "P"
                )
            ),

            "ack_flag_count": (
                flag_count(
                    "A"
                )
            ),

            "urg_flag_count": (
                flag_count(
                    "U"
                )
            ),

            "cwr_flag_count": (
                flag_count(
                    "C"
                )
            ),

            "ece_flag_count": (
                flag_count(
                    "E"
                )
            ),

            "down_up_ratio": (
                down_up_ratio
            ),

            "avg_fwd_segment_size": (
                mean_or_zero(
                    fwd_lengths
                )
            ),

            "avg_bwd_segment_size": (
                mean_or_zero(
                    bwd_lengths
                )
            ),

            "fwd_act_data_packets": (
                fwd_act_data_packets
            ),

            # Headers
            "fwd_header_length": (
                fwd_header_length
            ),

            "bwd_header_length": (
                bwd_header_length
            ),

            # Janelas TCP
            "init_fwd_win_bytes": (
                init_fwd_win_bytes
            ),

            "init_bwd_win_bytes": (
                init_bwd_win_bytes
            ),

            "fwd_seg_size_min": (
                fwd_seg_size_min
            ),

            # Bulk forward
            "fwd_avg_bytes_bulk": (
                fwd_avg_bytes_bulk
            ),

            "fwd_avg_packets_bulk": (
                fwd_avg_packets_bulk
            ),

            "fwd_avg_bulk_rate": (
                fwd_avg_bulk_rate
            ),

            # Bulk backward
            "bwd_avg_bytes_bulk": (
                bwd_avg_bytes_bulk
            ),

            "bwd_avg_packets_bulk": (
                bwd_avg_packets_bulk
            ),

            "bwd_avg_bulk_rate": (
                bwd_avg_bulk_rate
            ),

            # Subflows
            "subflow_fwd_packets": (
                subflow_fwd_packets
            ),

            "subflow_fwd_bytes": (
                subflow_fwd_bytes
            ),

            "subflow_bwd_packets": (
                subflow_bwd_packets
            ),

            "subflow_bwd_bytes": (
                subflow_bwd_bytes
            ),

            # Active
            "active_mean": (
                active_mean
            ),

            "active_std": (
                active_std
            ),

            "active_max": (
                active_max
            ),

            "active_min": (
                active_min
            ),

            # Idle
            "idle_mean": (
                idle_mean
            ),

            "idle_std": (
                idle_std
            ),

            "idle_max": (
                idle_max
            ),

            "idle_min": (
                idle_min
            ),
        }
