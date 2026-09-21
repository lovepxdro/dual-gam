from __future__ import annotations

from dataclasses import dataclass

from adarena.network.base import (
    CaptureBatch,
)

from adarena.network.inference import (
    NetworkInferencePipeline,
    NetworkInferenceResult,
)

from .pipeline import (
    ControlPipeline,
    ControlPipelineResult,
)


@dataclass(slots=True)
class ControlledObservationResult:
    """
    Resultado completo de uma observação seguida
    por decisão e aplicação de controle.
    """

    inference: NetworkInferenceResult
    control: ControlPipelineResult


class ControlledObservationPipeline:
    """
    Combina inferência de rede e plano de controle.

    Fluxo:

        CaptureBatch
            ↓
        NetworkInferencePipeline
            ↓
        ControlPipeline
            ↓
        ControlledObservationResult

    Este componente não captura pacotes diretamente.
    Ele opera sobre uma captura já produzida.
    """

    def __init__(
        self,
        *,
        inference_pipeline: NetworkInferencePipeline,
        control_pipeline: ControlPipeline,
    ) -> None:

        self.inference_pipeline = (
            inference_pipeline
        )

        self.control_pipeline = (
            control_pipeline
        )

    def process(
        self,
        capture: CaptureBatch,
    ) -> ControlledObservationResult:

        inference = (
            self
            .inference_pipeline
            .infer(
                capture
            )
        )

        control = (
            self
            .control_pipeline
            .process(
                inference
            )
        )

        return ControlledObservationResult(
            inference=inference,
            control=control,
        )
