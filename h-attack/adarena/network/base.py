from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Renderer(ABC):
    """
    Converte a representação produzida por um modelo
    em uma representação materializável pelo ambiente.

    Exemplo atual:

        FLOW_FEATURES -> ATTACK_PARAMS

    O Core não precisa conhecer AttackParams.
    """

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
    """
    Materializa uma representação no ambiente de rede.

    O Backend é responsável pela execução e pelas métricas
    observadas no lado emissor.

    Captura de tráfego não pertence a este contrato.
    """

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
