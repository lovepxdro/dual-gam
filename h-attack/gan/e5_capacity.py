"""E5-A3: controle de capacidade A2(z), independente da amostra x.

A2(z) = A0(z) + (1-|A0(z)|) * tanh(R(z)).

Mesmo tipo de ramo residual de A1, mas sem condicionamento por x.
Hidden=153 aproxima os parâmetros de A1 (104482 vs. 104538, com D=77,
z_dim=32). Todos os parâmetros do ramo são funcionalmente ativos: não
existem pesos de entradas x preenchidas artificialmente com zero.
"""
from __future__ import annotations

import torch
from torch import nn

from gan.models import Atacante


class AtacanteCapacidade(nn.Module):
    """Comparador de capacidade próxima a A1, com informação somente de z."""

    def __init__(self, noise_dim: int = 32, input_dim: int = 77, hidden_dim: int = 153):
        super().__init__()
        if min(noise_dim, input_dim, hidden_dim) <= 0:
            raise ValueError('Dimensões devem ser positivas')
        self.noise_dim = noise_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.base = Atacante(noise_dim=noise_dim, output_dim=input_dim)
        self.ramo = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
        )
        nn.init.zeros_(self.ramo[-1].weight)
        nn.init.zeros_(self.ramo[-1].bias)

    def carregar_base_a20(self, state: dict[str, torch.Tensor]) -> None:
        self.base.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x entra somente para manter a interface de E5; seu CONTEÚDO nunca é lido.
        if x.ndim != 2 or z.ndim != 2 or x.shape != (len(z), self.input_dim) or z.shape[1] != self.noise_dim:
            raise ValueError(f'Shapes incompatíveis: x={tuple(x.shape)}, z={tuple(z.shape)}')
        if x.device != z.device or x.dtype != z.dtype:
            raise ValueError('x e z devem compartilhar device e dtype')
        if not torch.isfinite(z).all():
            raise ValueError('z contém NaN/Inf')
        delta_base = self.base(z)
        residual = torch.tanh(self.ramo(z))
        return delta_base + (1.0 - delta_base.abs()) * residual
