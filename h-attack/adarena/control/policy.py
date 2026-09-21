from __future__ import annotations

import math
from typing import Any

from .base import (
    ControlAction,
    Decision,
    DecisionPolicy,
)


class ThresholdDecisionPolicy(DecisionPolicy):
    """
    Política binária baseada na probabilidade de ataque.

    Se o score for maior ou igual ao threshold, produz BLOCK.
    Caso contrário, produz NONE.

    A política não modifica a rede. Ela apenas transforma
    a saída do Defensor em uma decisão explícita.
    """

    def __init__(
        self,
        threshold: float = 0.5,
    ) -> None:

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold deve estar entre 0 e 1"
            )

        self.threshold = float(threshold)

    def decide(
        self,
        *,
        flow_id: str,
        prediction: int,
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> Decision:

        if prediction not in (0, 1):
            raise ValueError(
                "prediction deve ser 0 ou 1"
            )

        if not math.isfinite(score):
            raise ValueError(
                "score deve ser finito"
            )

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                "score deve estar entre 0 e 1"
            )

        if score >= self.threshold:
            action = ControlAction.BLOCK

            reason = (
                f"score de ataque {score:.4f} "
                f">= threshold {self.threshold:.4f}"
            )

        else:
            action = ControlAction.NONE

            reason = (
                f"score de ataque {score:.4f} "
                f"< threshold {self.threshold:.4f}"
            )

        return Decision(
            flow_id=flow_id,
            prediction=prediction,
            score=float(score),
            action=action,
            reason=reason,
            metadata=dict(
                metadata or {}
            ),
        )
