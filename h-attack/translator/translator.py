"""
Dual-GAM — Translator (GAN → Parâmetros de Rede)

Esta é a peça nova em relação ao experimento inicial.
O Atacante gera vetores no espaço de features do CIC-IDS2017 (77 dimensões, normalizadas).
O Translator converte esses vetores em parâmetros concretos que o Scapy usa para
gerar pacotes reais na rede Docker.

Lógica de mapeamento:
  - Desnormaliza o vetor usando o StandardScaler salvo
  - Extrai as features relevantes para geração de tráfego
  - Aplica clipping/sanitização para que os valores sejam fisicamente válidos
  - Retorna um AttackParams com tudo que o Sender precisa
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from gan.preprocessing import FEATURE_MAP, Preprocessador

logger = logging.getLogger(__name__)


@dataclass
class AttackParams:
    """
    Parâmetros de ataque prontos para o Sender (Scapy).
    Todos os valores já estão na escala real (sem normalização).
    """
    # Alvo
    target_ip: str
    target_port: int

    # Volume
    packets_per_second: float   # taxa de envio
    packet_size: int            # bytes por pacote (payload)
    duration_seconds: float     # quanto tempo manter o ataque

    # Flags TCP/UDP
    use_tcp: bool               # True=TCP, False=UDP
    tcp_flags: str              # ex: "S" (SYN), "SA" (SYN-ACK), "A" (ACK)
    use_syn_flood: bool         # se True, não completa o handshake
    randomize_src_ip: bool      # spoofing de IP fonte
    randomize_src_port: bool    # porta fonte aleatória

    # Janela TCP
    window_size: int            # TCP window size

    # Metadados (para logging e feedback à GAN)
    vetor_original: Optional[np.ndarray] = None
    evasao_prob: Optional[float] = None  # probabilidade que o Defensor deu como benigno

    def __repr__(self) -> str:
        return (
            f"AttackParams(target={self.target_ip}:{self.target_port}, "
            f"pps={self.packets_per_second:.0f}, size={self.packet_size}B, "
            f"dur={self.duration_seconds:.1f}s, "
            f"tcp={self.use_tcp}, flags={self.tcp_flags}, "
            f"syn_flood={self.use_syn_flood})"
        )


class Translator:
    """
    Converte vetores de features do CIC-IDS2017 em AttackParams para o Scapy.

    O mapeamento não é 1-para-1 perfeito — o CIC-IDS2017 descreve
    *fluxos* (estatísticas agregadas), não pacotes individuais. O Translator
    faz a ponte: usa as estatísticas do fluxo para inferir os parâmetros
    de geração de pacotes que produziriam um fluxo semelhante.
    """

    # Limites físicos para sanitização dos valores
    LIMITS = {
        "packets_per_second": (1.0, 100_000.0),
        "packet_size":        (40, 1460),        # min IP+TCP header, max MTU payload
        "duration_seconds":   (0.5, 30.0),
        "window_size":        (0, 65535),
    }

    def __init__(self, preprocessador: Preprocessador, target_ip: str, target_port: int = 80):
        self.prep = preprocessador
        self.target_ip = target_ip
        self.target_port = target_port

    def traduzir(
        self,
        vetor_normalizado: np.ndarray,
        evasao_prob: Optional[float] = None,
    ) -> AttackParams:
        """
        Converte um único vetor de features (shape: [77]) em AttackParams.

        Args:
            vetor_normalizado: saída do Atacante, shape (77,), escala StandardScaler
            evasao_prob: probabilidade atribuída pelo Defensor (para logging)
        """
        # 1. Desnormalizar para a escala original
        vetor = self.prep.desnormalizar(vetor_normalizado.reshape(1, -1)).flatten()

        # 2. Extrair features relevantes usando o mapa do CIC-IDS2017
        pps_raw   = float(vetor[FEATURE_MAP["flow_packets_per_sec"]])
        bps_raw   = float(vetor[FEATURE_MAP["flow_bytes_per_sec"]])
        pkt_mean  = float(vetor[FEATURE_MAP["fwd_packet_length_mean"]])
        syn_count = float(vetor[FEATURE_MAP["syn_flag_count"]])
        ack_count = float(vetor[FEATURE_MAP["ack_flag_count"]])
        fin_count = float(vetor[FEATURE_MAP["fin_flag_count"]])
        psh_count = float(vetor[FEATURE_MAP["psh_flag_count"]])
        win_bytes = float(vetor[FEATURE_MAP["init_fwd_win_bytes"]])
        duration  = float(vetor[FEATURE_MAP["flow_duration"]]) / 1e6  # microsseg → seg
        avg_size  = float(vetor[FEATURE_MAP["avg_pkt_size"]])

        # 3. Derivar parâmetros concretos

        # Taxa de pacotes: usar flow_packets_per_sec diretamente
        pps = self._clamp(abs(pps_raw), *self.LIMITS["packets_per_second"])

        # Tamanho do pacote: média entre fwd_packet_length_mean e avg_pkt_size
        raw_size = (abs(pkt_mean) + abs(avg_size)) / 2
        packet_size = int(self._clamp(raw_size, *self.LIMITS["packet_size"]))

        # Duração: flow_duration (convertido de microssegundos)
        dur = self._clamp(abs(duration), *self.LIMITS["duration_seconds"])

        # TCP window size
        win = int(self._clamp(abs(win_bytes), *self.LIMITS["window_size"]))

        # Tipo de protocolo: DDoS do CIC-IDS2017 é majoritariamente UDP flood,
        # mas o atacante pode aprender a variar. Usa SYN count como sinal.
        use_tcp = syn_count > 0 or ack_count > 0

        # Flags TCP
        tcp_flags = self._inferir_flags(syn_count, ack_count, fin_count, psh_count)

        # SYN flood: muitos SYN, poucos ACK → handshake incompleto
        use_syn_flood = syn_count > ack_count * 2

        logger.debug(
            "Tradução: pps=%.0f, size=%d, dur=%.1f, tcp=%s, flags=%s, syn_flood=%s",
            pps, packet_size, dur, use_tcp, tcp_flags, use_syn_flood,
        )

        return AttackParams(
            target_ip=self.target_ip,
            target_port=self.target_port,
            packets_per_second=pps,
            packet_size=packet_size,
            duration_seconds=dur,
            use_tcp=use_tcp,
            tcp_flags=tcp_flags,
            use_syn_flood=use_syn_flood,
            randomize_src_ip=True,    # sempre spoof para DDoS realista
            randomize_src_port=True,
            window_size=win,
            vetor_original=vetor_normalizado,
            evasao_prob=evasao_prob,
        )

    def traduzir_batch(
        self,
        vetores: np.ndarray,
        evasao_probs: Optional[list[float]] = None,
    ) -> list[AttackParams]:
        """Converte um batch de vetores (shape: [N, 77]) em lista de AttackParams."""
        params = []
        for i, v in enumerate(vetores):
            prob = evasao_probs[i] if evasao_probs else None
            params.append(self.traduzir(v, prob))
        return params

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _inferir_flags(syn: float, ack: float, fin: float, psh: float) -> str:
        """Determina combinação de flags TCP mais provável."""
        flags = ""
        if syn > 0:
            flags += "S"
        if ack > 0:
            flags += "A"
        if psh > 0:
            flags += "P"
        if fin > 0:
            flags += "F"
        return flags if flags else "S"  # default: SYN flood
