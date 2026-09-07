from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .components import (
    ComponentKind,
    ComponentSpec,
    DataRepresentation,
)


ComponentFactory = Callable[..., Any]


@dataclass(frozen=True, slots=True)
class RegisteredComponent:
    spec: ComponentSpec
    factory: ComponentFactory


class ComponentRegistry:
    """Catálogo de componentes disponíveis em uma execução da ADArena."""

    def __init__(self) -> None:
        self._components: dict[str, RegisteredComponent] = {}

    def register(
        self,
        spec: ComponentSpec,
        factory: ComponentFactory,
        *,
        replace: bool = False,
    ) -> None:
        if not callable(factory):
            raise TypeError("factory precisa ser chamável")

        if spec.component_id in self._components and not replace:
            raise ValueError(
                f"Componente já registrado: {spec.component_id}"
            )

        self._components[spec.component_id] = RegisteredComponent(
            spec=spec,
            factory=factory,
        )

    def unregister(self, component_id: str) -> None:
        try:
            del self._components[component_id]
        except KeyError as exc:
            raise KeyError(
                f"Componente não registrado: {component_id}"
            ) from exc

    def contains(self, component_id: str) -> bool:
        return component_id in self._components

    def get(self, component_id: str) -> RegisteredComponent:
        try:
            return self._components[component_id]
        except KeyError as exc:
            raise KeyError(
                f"Componente não registrado: {component_id}"
            ) from exc

    def spec(self, component_id: str) -> ComponentSpec:
        return self.get(component_id).spec

    def create(
        self,
        component_id: str,
        **kwargs: Any,
    ) -> Any:
        return self.get(component_id).factory(**kwargs)

    def list(
        self,
        *,
        kind: ComponentKind | None = None,
        input_representation: DataRepresentation | None = None,
        output_representation: DataRepresentation | None = None,
    ) -> list[ComponentSpec]:
        specs: Iterable[ComponentSpec] = (
            registered.spec
            for registered in self._components.values()
        )

        if kind is not None:
            specs = (
                spec
                for spec in specs
                if spec.kind == kind
            )

        if input_representation is not None:
            specs = (
                spec
                for spec in specs
                if spec.input_representation == input_representation
            )

        if output_representation is not None:
            specs = (
                spec
                for spec in specs
                if spec.output_representation == output_representation
            )

        return sorted(
            specs,
            key=lambda spec: (
                spec.kind.value,
                spec.name.lower(),
            ),
        )
