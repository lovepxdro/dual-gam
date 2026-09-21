from __future__ import annotations

from dataclasses import dataclass

from adarena.network.inference import (
    NetworkInferenceResult,
)

from .base import (
    Decision,
    DecisionPolicy,
    EnforcementResult,
    RuleEnforcer,
)


@dataclass(slots=True)
class ControlPipelineResult:
    """
    Resultado da etapa de decisão e aplicação de controle.
    """

    inference: NetworkInferenceResult

    decisions: tuple[
        Decision,
        ...,
    ]

    enforcement_results: tuple[
        EnforcementResult,
        ...,
    ]

    @property
    def n_flows(
        self,
    ) -> int:
        return len(
            self.decisions
        )

    @property
    def block_count(
        self,
    ) -> int:
        from .base import (
            ControlAction,
        )

        return sum(
            decision.action
            == ControlAction.BLOCK
            for decision
            in self.decisions
        )

    @property
    def applied_count(
        self,
    ) -> int:
        return sum(
            result.applied
            for result
            in self.enforcement_results
        )


class ControlPipeline:
    """
    Conecta a saída do Defensor ao plano de controle.

    Fluxo:

        NetworkInferenceResult
                ↓
        DecisionPolicy
                ↓
        RuleEnforcer

    O pipeline não realiza inferência e não conhece
    detalhes de captura, extração ou PyTorch.
    """

    def __init__(
        self,
        *,
        policy: DecisionPolicy,
        enforcer: RuleEnforcer,
    ) -> None:

        self.policy = policy
        self.enforcer = enforcer

    def process(
        self,
        inference: NetworkInferenceResult,
    ) -> ControlPipelineResult:

        flow_ids = (
            inference
            .extracted
            .flow_ids
        )

        probabilities = (
            inference
            .probabilities
        )

        predictions = (
            inference
            .predictions
        )

        n_flows = len(
            flow_ids
        )

        if (
            len(probabilities)
            != n_flows
        ):
            raise ValueError(
                "Quantidade de probabilidades "
                "incompatível com flow_ids"
            )

        if (
            len(predictions)
            != n_flows
        ):
            raise ValueError(
                "Quantidade de predições "
                "incompatível com flow_ids"
            )

        decisions: list[
            Decision
        ] = []

        enforcement_results: list[
            EnforcementResult
        ] = []

        for (
            flow_id,
            prediction,
            probability,
        ) in zip(
            flow_ids,
            predictions,
            probabilities,
            strict=True,
        ):

            decision = (
                self.policy.decide(
                    flow_id=str(
                        flow_id
                    ),
                    prediction=int(
                        prediction
                    ),
                    score=float(
                        probability
                    ),
                    metadata={
                        "inference_threshold": (
                            inference.threshold
                        ),
                    },
                )
            )

            enforcement_result = (
                self.enforcer.enforce(
                    decision
                )
            )

            decisions.append(
                decision
            )

            enforcement_results.append(
                enforcement_result
            )

        return ControlPipelineResult(
            inference=inference,

            decisions=tuple(
                decisions
            ),

            enforcement_results=tuple(
                enforcement_results
            ),
        )
