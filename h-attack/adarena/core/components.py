from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class ComponentKind(str, Enum):
    ATTACKER = "attacker"
    DEFENDER = "defender"
    DATASET = "dataset"
    RENDERER = "renderer"
    EXTRACTOR = "extractor"
    NETWORK_BACKEND = "network_backend"


class DataRepresentation(str, Enum):
    """Representações que podem atravessar as fronteiras da ADArena."""

    FLOW_FEATURES = "flow_features"
    PACKET_SEQUENCE = "packet_sequence"
    RAW_PACKETS = "raw_packets"
    ATTACK_PARAMS = "attack_params"
    BINARY_CLASSIFICATION = "binary_classification"


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """
    Metadados estáveis usados para descoberta e validação de componentes.

    O Core conhece apenas o contrato declarado aqui. Detalhes internos de uma
    implementação ficam fora do registry.
    """

    component_id: str
    kind: ComponentKind
    name: str
    version: str = "1.0"
    description: str = ""
    input_representation: DataRepresentation | None = None
    output_representation: DataRepresentation | None = None
    tags: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.component_id.strip():
            raise ValueError("component_id não pode ser vazio")

        if not self.name.strip():
            raise ValueError("name não pode ser vazio")

        normalized_id = self.component_id.strip().lower()

        allowed = set(
            "abcdefghijklmnopqrstuvwxyz"
            "0123456789._-"
        )

        if any(char not in allowed for char in normalized_id):
            raise ValueError(
                "component_id deve usar apenas letras minúsculas, "
                "números, '.', '_' ou '-'"
            )

        if normalized_id != self.component_id:
            raise ValueError(
                "component_id deve estar normalizado em minúsculas"
            )

        object.__setattr__(
            self,
            "metadata",
            MappingProxyType(dict(self.metadata)),
        )
