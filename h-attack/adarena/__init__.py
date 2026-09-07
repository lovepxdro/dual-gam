"""ADArena — núcleo modular da plataforma experimental."""

from .builtin import create_default_registry, register_builtin_components

__all__ = [
    "create_default_registry",
    "register_builtin_components",
]
