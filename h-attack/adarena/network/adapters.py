from __future__ import annotations

from typing import Any

import numpy as np

from .base import (
    NetworkBackend,
    Renderer,
)


class TranslatorRenderer(Renderer):
    """
    Adapter que transforma o Translator da linha 1.x
    em um Renderer formal da ADArena.

    Atualmente:

        flow features normalizadas
                ↓
            Translator
                ↓
            AttackParams

    O conhecimento específico do CIC-IDS2017 continua
    isolado no Translator atual.
    """

    def __init__(
        self,
        *,
        preprocessor,
        target_ip: str,
        target_port: int = 80,
        consistency_tolerance: float = 0.75,
    ) -> None:

        # Import tardio:
        # o Core não depende do código legado.
        from translator.translator import (
            Translator,
        )

        self.translator = Translator(
            preprocessador=preprocessor,
            target_ip=target_ip,
            target_port=target_port,
            consistency_tolerance=(
                consistency_tolerance
            ),
        )

    def render(
        self,
        sample: Any,
        *,
        score: float | None = None,
    ) -> Any:

        return self.translator.traduzir(
            np.asarray(sample),
            evasao_prob=score,
        )

    def render_batch(
        self,
        samples: Any,
        *,
        scores: list[float] | None = None,
        only_valid: bool = False,
    ) -> list[Any]:

        samples = np.asarray(
            samples
        )

        normalized_scores = (
            list(scores)
            if scores is not None
            else None
        )

        return self.translator.traduzir_batch(
            samples,
            evasao_probs=normalized_scores,
            only_valid=only_valid,
        )


class SenderNetworkBackend(
    NetworkBackend
):
    """
    Adapter do Sender atual.

    O backend apenas materializa AttackParams
    no ambiente controlado e devolve AttackResult.

    Ele não captura o que a rede recebeu.
    Essa responsabilidade será da próxima camada.
    """

    def __init__(
        self,
        *,
        iface: str | None = None,
        dry_run: bool = False,
        require_private_target: bool = True,
    ) -> None:

        from sender.sender import Sender

        self.sender = Sender(
            iface=iface,
            dry_run=dry_run,
            require_private_target=(
                require_private_target
            ),
        )

    def execute(
        self,
        payload: Any,
    ) -> Any:

        return self.sender.executar(
            payload
        )
