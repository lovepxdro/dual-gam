"""Primitivas centrais da ADArena."""

from .components import ComponentKind, ComponentSpec, DataRepresentation
from .config import ComponentSelection, ExperimentConfig, ExperimentMode
from .registry import ComponentRegistry, RegisteredComponent

__all__ = [
    "ComponentKind",
    "ComponentSpec",
    "DataRepresentation",
    "ComponentSelection",
    "ExperimentConfig",
    "ExperimentMode",
    "ComponentRegistry",
    "RegisteredComponent",
]
