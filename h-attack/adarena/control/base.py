from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ControlAction(str, Enum):
    """
    Ações que podem ser produzidas por uma política de decisão.

    Nesta primeira versão mantemos deliberadamente um conjunto
    mínimo. Novas ações, como rate limiting, podem ser adicionadas
    posteriormente sem alterar o contrato principal.
    """

    NONE = "none"
    BLOCK = "block"


@dataclass(slots=True)
class Decision:
    """
    Decisão produzida a partir da classificação de um fluxo.

    `score` representa, no cenário binário atual, a probabilidade
    atribuída à classe de ataque pelo Defensor.

    `metadata` permite preservar informações adicionais do fluxo
    necessárias para auditoria ou execução posterior da decisão.
    """

    flow_id: str

    prediction: int
    score: float

    action: ControlAction

    reason: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class EnforcementResult:
    """
    Resultado da tentativa de aplicar uma decisão.

    `success` indica que o RuleEnforcer processou a decisão
    corretamente.

    `applied` indica que houve alteração efetiva no plano de dados.

    Essa distinção é importante para dry-run:

        success = True
        applied = False
        dry_run = True
    """

    decision: Decision

    success: bool
    applied: bool

    dry_run: bool = False

    message: str = ""

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class DecisionPolicy(ABC):
    """
    Converte uma classificação do Defensor em uma decisão de controle.

    A política não deve modificar a rede diretamente.
    """

    @abstractmethod
    def decide(
        self,
        *,
        flow_id: str,
        prediction: int,
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> Decision:
        """
        Produz uma decisão para um fluxo observado.
        """

        raise NotImplementedError


class RuleEnforcer(ABC):
    """
    Executa uma decisão produzida por uma DecisionPolicy.

    Implementações concretas podem apenas registrar a ação,
    operar em dry-run ou, futuramente, conversar com um
    controlador SDN.
    """

    @abstractmethod
    def enforce(
        self,
        decision: Decision,
    ) -> EnforcementResult:
        """
        Processa uma decisão de controle.
        """

        raise NotImplementedError
