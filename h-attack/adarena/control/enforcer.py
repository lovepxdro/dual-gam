from __future__ import annotations

import logging

from .base import (
    ControlAction,
    Decision,
    EnforcementResult,
    RuleEnforcer,
)


logger = logging.getLogger(__name__)


class DryRunRuleEnforcer(RuleEnforcer):
    """
    RuleEnforcer que registra decisões sem alterar o plano de dados.

    Serve para:
    - validar o fluxo completo de decisão;
    - produzir artefatos auditáveis;
    - testar políticas sem modificar a rede;
    - servir como implementação segura de referência.
    """

    def __init__(self) -> None:
        self.history: list[EnforcementResult] = []

    def enforce(
        self,
        decision: Decision,
    ) -> EnforcementResult:

        if decision.action == ControlAction.NONE:
            message = (
                f"Nenhuma ação necessária para "
                f"{decision.flow_id}"
            )

        elif decision.action == ControlAction.BLOCK:
            message = (
                f"Bloqueio solicitado para "
                f"{decision.flow_id}; "
                "não aplicado porque o enforcer "
                "está em dry-run"
            )

        else:
            raise ValueError(
                f"Ação não suportada: "
                f"{decision.action}"
            )

        result = EnforcementResult(
            decision=decision,
            success=True,
            applied=False,
            dry_run=True,
            message=message,
        )

        self.history.append(
            result
        )

        logger.debug(
            "Control action | "
            "flow=%s action=%s "
            "success=%s applied=%s dry_run=%s",
            decision.flow_id,
            decision.action.value,
            result.success,
            result.applied,
            result.dry_run,
        )

        return result
