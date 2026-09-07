from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


class Renderer(ABC):

    @abstractmethod
    def render(
        self,
        sample: Any,
        *,
        score: float | None = None,
    ) -> Any:
        raise NotImplementedError

    @abstractmethod
    def render_batch(
        self,
        samples: Any,
        *,
        scores: list[float] | None = None,
        only_valid: bool = False,
    ) -> list[Any]:
        raise NotImplementedError


class NetworkBackend(ABC):

    @abstractmethod
    def execute(
        self,
        payload: Any,
    ) -> Any:
        raise NotImplementedError

    def execute_many(
        self,
        payloads: list[Any],
    ) -> list[Any]:
        return [
            self.execute(payload)
            for payload in payloads
        ]


@dataclass(frozen=True, slots=True)
class CapturedPacket:
    """
    Representação mínima e independente de biblioteca
    de um pacote observado no ambiente experimental.
    """

    timestamp: float

    src_ip: str
    dst_ip: str

    src_port: int | None
    dst_port: int | None

    protocol: str

    length: int

    tcp_flags: str = ""

    payload_length: int = 0

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class CaptureBatch:
    packets: list[CapturedPacket]

    started_at: float
    ended_at: float

    interface: str | None = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def duration(self) -> float:
        return max(
            0.0,
            self.ended_at
            - self.started_at,
        )

    def __len__(self) -> int:
        return len(
            self.packets
        )


@dataclass(slots=True)
class FlowFeatureBatch:
    """
    Resultado de um FlowExtractor.

    X segue exatamente a ordem de feature_names.

    Features não reconstruíveis podem ser representadas
    por NaN quando strict=False. Nesse caso o batch não
    está pronto para ser enviado ao modelo.
    """

    X: np.ndarray

    feature_names: tuple[str, ...]

    flow_ids: tuple[str, ...]

    unsupported_features: tuple[
        str,
        ...
    ] = ()

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        self.X = np.asarray(
            self.X,
            dtype=np.float32,
        )

        if self.X.ndim != 2:
            raise ValueError(
                "X deve possuir shape "
                "[N, features]"
            )

        if (
            self.X.shape[1]
            != len(self.feature_names)
        ):
            raise ValueError(
                "Quantidade de colunas "
                "incompatível com feature_names"
            )

        if (
            self.X.shape[0]
            != len(self.flow_ids)
        ):
            raise ValueError(
                "Quantidade de fluxos "
                "incompatível com flow_ids"
            )

    @property
    def ready_for_model(self) -> bool:
        return bool(
            not self.unsupported_features
            and np.isfinite(
                self.X
            ).all()
        )


class Capture(ABC):

    @abstractmethod
    def capture(
        self,
        *,
        duration: float,
        packet_limit: int | None = None,
    ) -> CaptureBatch:
        raise NotImplementedError


class FlowExtractor(ABC):

    @abstractmethod
    def extract(
        self,
        capture: CaptureBatch,
        *,
        feature_names: Sequence[str],
        strict: bool = True,
    ) -> FlowFeatureBatch:
        raise NotImplementedError
