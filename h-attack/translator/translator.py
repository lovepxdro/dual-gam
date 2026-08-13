"""
Dual-GAM — Translator (features adversariais → parâmetros de rede)

O Translator é a ponte entre o espaço estatístico do CIC-IDS2017 e a execução
na rede. A tradução não é 1:1: o dataset descreve fluxos agregados, enquanto o
Sender trabalha com parâmetros concretos de geração de pacotes.

Objetivos desta versão:
- resolver features pelos nomes persistidos pelo Preprocessador;
- manter fallback compatível com o FEATURE_MAP legado;
- rejeitar valores não finitos e evitar converter valores fisicamente inválidos
  em válidos apenas com ``abs()``;
- validar relações simples de consistência entre PPS, BPS e tamanho médio;
- registrar correções/projeções aplicadas para auditoria experimental.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gan.preprocessing import FEATURE_MAP, Preprocessador

logger = logging.getLogger(__name__)


@dataclass
class AttackParams:
    """Parâmetros concretos produzidos pelo Translator."""

    target_ip: str
    target_port: int

    packets_per_second: float
    packet_size: int
    duration_seconds: float

    use_tcp: bool
    tcp_flags: str
    use_syn_flood: bool
    randomize_src_ip: bool
    randomize_src_port: bool

    window_size: int

    vetor_original: Optional[np.ndarray] = None
    evasao_prob: Optional[float] = None

    # Auditoria da tradução.
    translation_valid: bool = True
    translation_warnings: list[str] = field(default_factory=list)
    consistency_error: Optional[float] = None
    raw_features: dict[str, float] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"AttackParams(target={self.target_ip}:{self.target_port}, "
            f"pps={self.packets_per_second:.0f}, size={self.packet_size}B, "
            f"dur={self.duration_seconds:.1f}s, "
            f"tcp={self.use_tcp}, flags={self.tcp_flags}, "
            f"syn_flood={self.use_syn_flood}, valid={self.translation_valid})"
        )


class TranslationError(ValueError):
    """Indica que um vetor não pode ser traduzido de forma confiável."""


class Translator:
    """Converte vetores normalizados em parâmetros concretos de rede."""

    LIMITS = {
        "packets_per_second": (1.0, 100_000.0),
        "packet_size": (40.0, 1460.0),
        "duration_seconds": (0.5, 30.0),
        "window_size": (0.0, 65535.0),
    }

    # Nomes encontrados em diferentes versões/exportações do CIC-IDS2017.
    FEATURE_ALIASES = {
        "flow_duration": ["flow_duration", "flow duration"],
        "flow_packets_per_sec": [
            "flow_packets_per_sec",
            "flow packets/s",
            "flow packets per second",
        ],
        "flow_bytes_per_sec": [
            "flow_bytes_per_sec",
            "flow bytes/s",
            "flow bytes per second",
        ],
        "fwd_packet_length_mean": [
            "fwd_packet_length_mean",
            "fwd packet length mean",
        ],
        "avg_pkt_size": [
            "avg_pkt_size",
            "average packet size",
            "avg packet size",
        ],
        "syn_flag_count": ["syn_flag_count", "syn flag count"],
        "ack_flag_count": ["ack_flag_count", "ack flag count"],
        "fin_flag_count": ["fin_flag_count", "fin flag count"],
        "psh_flag_count": ["psh_flag_count", "psh flag count"],
        "init_fwd_win_bytes": [
            "init_fwd_win_bytes",
            "init_win_bytes_forward",
            "init win bytes forward",
        ],
    }

    def __init__(
        self,
        preprocessador: Preprocessador,
        target_ip: str,
        target_port: int = 80,
        consistency_tolerance: float = 0.75,
    ):
        self.prep = preprocessador
        self.target_ip = target_ip
        self.target_port = target_port
        self.consistency_tolerance = consistency_tolerance
        self._feature_index = self._build_feature_index()

    @staticmethod
    def _normalize_name(name: str) -> str:
        name = name.strip().lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        return name.strip("_")

    def _build_feature_index(self) -> dict[str, int]:
        normalized = {
            self._normalize_name(name): idx
            for idx, name in enumerate(self.prep.feature_names)
        }

        resolved: dict[str, int] = {}
        for canonical, aliases in self.FEATURE_ALIASES.items():
            candidates = [canonical, *aliases]
            found = None
            for candidate in candidates:
                key = self._normalize_name(candidate)
                if key in normalized:
                    found = normalized[key]
                    break

            if found is None and canonical in FEATURE_MAP:
                legacy_index = FEATURE_MAP[canonical]
                if legacy_index < len(self.prep.feature_names):
                    found = legacy_index
                    logger.warning(
                        "Translator: feature '%s' não localizada por nome; "
                        "usando índice legado %d",
                        canonical,
                        legacy_index,
                    )

            if found is None:
                raise KeyError(
                    f"Feature necessária não encontrada no preprocessador: {canonical}"
                )
            resolved[canonical] = found

        return resolved

    def _feature(self, vetor: np.ndarray, name: str) -> float:
        value = float(vetor[self._feature_index[name]])
        if not np.isfinite(value):
            raise TranslationError(f"Feature não finita: {name}={value}")
        return value

    @staticmethod
    def _project_nonnegative(value: float, name: str, warnings: list[str]) -> float:
        if value >= 0:
            return value
        warnings.append(f"{name}: valor negativo projetado para 0 ({value:.6g})")
        return 0.0

    @staticmethod
    def _clamp(
        value: float,
        lo: float,
        hi: float,
        name: str,
        warnings: list[str],
    ) -> float:
        projected = max(lo, min(hi, value))
        if projected != value:
            warnings.append(
                f"{name}: {value:.6g} projetado para [{lo:g}, {hi:g}] -> {projected:.6g}"
            )
        return projected

    def _inferir_flags(
        self,
        syn: float,
        ack: float,
        fin: float,
        psh: float,
    ) -> str:
        # Contagens são estatísticas de fluxo. Usamos presença relevante em vez
        # de qualquer ruído positivo minúsculo após uma perturbação adversarial.
        counts = {
            "S": max(0.0, syn),
            "A": max(0.0, ack),
            "F": max(0.0, fin),
            "P": max(0.0, psh),
        }
        max_count = max(counts.values())
        if max_count < 0.5:
            return "S"

        threshold = max(0.5, 0.10 * max_count)
        flags = "".join(flag for flag, count in counts.items() if count >= threshold)
        return flags or "S"

    def _consistency_error(
        self,
        pps: float,
        bps: float,
        packet_size: float,
    ) -> Optional[float]:
        """
        Mede discrepância relativa entre BPS observado no vetor e BPS estimado
        por PPS × tamanho médio. Não é uma igualdade perfeita no CIC-IDS2017,
        portanto a métrica serve como auditoria, não como prova física exata.
        """
        if pps <= 0 or bps <= 0 or packet_size <= 0:
            return None
        expected_bps = pps * packet_size
        return abs(expected_bps - bps) / max(abs(bps), 1.0)

    def traduzir(
        self,
        vetor_normalizado: np.ndarray,
        evasao_prob: Optional[float] = None,
    ) -> AttackParams:
        """Converte um vetor normalizado em ``AttackParams`` auditáveis."""
        vetor_normalizado = np.asarray(vetor_normalizado, dtype=np.float64)
        if vetor_normalizado.ndim != 1:
            raise TranslationError("vetor_normalizado deve possuir shape [features]")
        if len(vetor_normalizado) != len(self.prep.feature_names):
            raise TranslationError(
                "Dimensão do vetor incompatível com o preprocessador: "
                f"{len(vetor_normalizado)} != {len(self.prep.feature_names)}"
            )
        if not np.isfinite(vetor_normalizado).all():
            raise TranslationError("Vetor normalizado contém NaN ou infinito")

        vetor = self.prep.desnormalizar(vetor_normalizado.reshape(1, -1)).flatten()
        warnings: list[str] = []

        raw = {
            name: self._feature(vetor, name)
            for name in self._feature_index
        }

        pps_raw = self._project_nonnegative(
            raw["flow_packets_per_sec"], "flow_packets_per_sec", warnings
        )
        bps_raw = self._project_nonnegative(
            raw["flow_bytes_per_sec"], "flow_bytes_per_sec", warnings
        )
        fwd_mean = self._project_nonnegative(
            raw["fwd_packet_length_mean"], "fwd_packet_length_mean", warnings
        )
        avg_size = self._project_nonnegative(
            raw["avg_pkt_size"], "avg_pkt_size", warnings
        )
        duration = self._project_nonnegative(
            raw["flow_duration"] / 1e6,
            "flow_duration_seconds",
            warnings,
        )
        window = self._project_nonnegative(
            raw["init_fwd_win_bytes"], "init_fwd_win_bytes", warnings
        )

        pps = self._clamp(
            pps_raw,
            *self.LIMITS["packets_per_second"],
            "packets_per_second",
            warnings,
        )

        size_candidates = [v for v in (fwd_mean, avg_size) if v > 0]
        if size_candidates:
            raw_size = float(np.mean(size_candidates))
        elif pps_raw > 0 and bps_raw > 0:
            raw_size = bps_raw / pps_raw
            warnings.append("packet_size derivado de BPS/PPS")
        else:
            raw_size = self.LIMITS["packet_size"][0]
            warnings.append("packet_size sem evidência válida; usando limite mínimo")

        packet_size = int(round(self._clamp(
            raw_size,
            *self.LIMITS["packet_size"],
            "packet_size",
            warnings,
        )))
        dur = self._clamp(
            duration,
            *self.LIMITS["duration_seconds"],
            "duration_seconds",
            warnings,
        )
        win = int(round(self._clamp(
            window,
            *self.LIMITS["window_size"],
            "window_size",
            warnings,
        )))

        syn = raw["syn_flag_count"]
        ack = raw["ack_flag_count"]
        fin = raw["fin_flag_count"]
        psh = raw["psh_flag_count"]

        positive_syn = max(0.0, syn)
        positive_ack = max(0.0, ack)
        use_tcp = max(positive_syn, positive_ack, max(0.0, fin), max(0.0, psh)) >= 0.5
        tcp_flags = self._inferir_flags(syn, ack, fin, psh)
        use_syn_flood = use_tcp and positive_syn >= 0.5 and positive_syn > positive_ack * 2.0

        consistency_error = self._consistency_error(pps_raw, bps_raw, raw_size)
        translation_valid = True
        if consistency_error is not None and consistency_error > self.consistency_tolerance:
            warnings.append(
                "inconsistência PPS/BPS/tamanho acima da tolerância: "
                f"erro_relativo={consistency_error:.3f}"
            )
            translation_valid = False

        logger.debug(
            "Tradução: pps=%.0f size=%d dur=%.2f tcp=%s flags=%s "
            "syn_flood=%s valid=%s warnings=%d",
            pps,
            packet_size,
            dur,
            use_tcp,
            tcp_flags,
            use_syn_flood,
            translation_valid,
            len(warnings),
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
            randomize_src_ip=True,
            randomize_src_port=True,
            window_size=win,
            vetor_original=vetor_normalizado.astype(np.float32, copy=True),
            evasao_prob=evasao_prob,
            translation_valid=translation_valid,
            translation_warnings=warnings,
            consistency_error=consistency_error,
            raw_features=raw,
        )

    def traduzir_batch(
        self,
        vetores: np.ndarray,
        evasao_probs: Optional[list[float]] = None,
        only_valid: bool = False,
    ) -> list[AttackParams]:
        """Converte um batch reutilizando exatamente a tradução unitária."""
        vetores = np.asarray(vetores)
        if vetores.ndim != 2:
            raise TranslationError("vetores deve possuir shape [N, features]")
        if evasao_probs is not None and len(evasao_probs) != len(vetores):
            raise ValueError("evasao_probs deve ter o mesmo tamanho do batch")

        params: list[AttackParams] = []
        for i, vetor in enumerate(vetores):
            prob = evasao_probs[i] if evasao_probs is not None else None
            translated = self.traduzir(vetor, prob)
            if only_valid and not translated.translation_valid:
                continue
            params.append(translated)
        return params
