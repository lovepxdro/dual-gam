"""E5 — gerador condicional residual; não altera o Attacker do E0.

A1(x, z) = A0(z) + (1 - |A0(z)|) * tanh(R(x, z)).

O último Linear de R começa em zero, logo A1(x,z) == A0(z) no
instante inicial. A construção mantém a perturbação em [-1,1] sem clamp.
Entradas x e z devem estar no espaço normalizado do E0.
"""
from __future__ import annotations

import torch
from torch import nn

from gan.models import Atacante


class AtacanteCondicional(nn.Module):
    """Attacker com backbone histórico G(z) + ramo condicionado à amostra.

    Usar ``carregar_base_a20`` com estado salvo de atacante_rodada_20.pth.
    O D20 não faz parte deste módulo e permanece congelado no protocolo E5.
    """

    def __init__(self, noise_dim: int = 32, input_dim: int = 77, hidden_dim: int = 128):
        super().__init__()
        if min(noise_dim, input_dim, hidden_dim) <= 0:
            raise ValueError("Dimensões devem ser positivas")
        self.noise_dim = noise_dim
        self.input_dim = input_dim
        self.base = Atacante(noise_dim=noise_dim, output_dim=input_dim)
        self.ramo = nn.Sequential(
            nn.Linear(input_dim + noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
        )
        # Zero residual no instante inicial: igualdade com A20, não mera proximidade.
        nn.init.zeros_(self.ramo[-1].weight)
        nn.init.zeros_(self.ramo[-1].bias)

    def carregar_base_a20(self, estado: dict[str, torch.Tensor]) -> None:
        """Carga estrita do checkpoint legado (chaves 'rede.*')."""
        self.base.load_state_dict(estado, strict=True)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or z.ndim != 2:
            raise ValueError("x e z devem ter shape [N, D]")
        if x.shape[0] != z.shape[0] or x.shape[1] != self.input_dim or z.shape[1] != self.noise_dim:
            raise ValueError(
                f"Shapes incompatíveis: x={tuple(x.shape)}, z={tuple(z.shape)}; "
                f"esperado [N,{self.input_dim}] e [N,{self.noise_dim}]"
            )
        if x.device != z.device or x.dtype != z.dtype:
            raise ValueError("x e z devem compartilhar device e dtype")
        if not torch.isfinite(x).all() or not torch.isfinite(z).all():
            raise ValueError("Entradas contêm NaN ou Inf")
        delta_base = self.base(z)
        residual = torch.tanh(self.ramo(torch.cat((x, z), dim=1)))
        # Para b em [-1,1] e r em [-1,1], b + (1-|b|)*r permanece em [-1,1].
        return delta_base + (1.0 - delta_base.abs()) * residual

    def perturba(self, x: torch.Tensor, z: torch.Tensor, epsilon: float = 0.3) -> torch.Tensor:
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon deve estar em [0,1]")
        return x + epsilon * self.forward(x, z)
