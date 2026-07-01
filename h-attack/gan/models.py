"""
Dual-GAM — Modelos de rede neural
Adaptado do experimento inicial (notebook Colab) para uso standalone.

Mantém a arquitetura original que funcionou:
- Defensor: MLP 3 camadas (classificador binário benigno/DDoS)
- Atacante: gerador de perturbações adversariais sobre amostras reais de DDoS
"""

import torch
import torch.nn as nn


class Defensor(nn.Module):
    """
    Classificador binário: benigno (0) vs DDoS (1).
    Arquitetura idêntica ao baseline do experimento inicial.
    """

    def __init__(self, input_dim: int = 77):
        super().__init__()
        self.rede = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x).squeeze(1)

    def prob_ddos(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidade de ser DDoS (entre 0 e 1)."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def classifica(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Retorna 0 (benigno) ou 1 (DDoS)."""
        return (self.prob_ddos(x) >= threshold).float()


class Atacante(nn.Module):
    """
    Gerador de perturbações adversariais.

    Recebe ruído aleatório e aprende perturbações que, aplicadas sobre
    amostras reais de DDoS, enganam o Defensor (que as classifica como benigno).

    A perturbação é escalada por EPSILON para manter o tráfego realista:
        amostra_perturbada = amostra_ddos_real + EPSILON * atacante(ruido)
    """

    def __init__(self, noise_dim: int = 32, output_dim: int = 77):
        super().__init__()
        self.noise_dim = noise_dim
        self.rede = nn.Sequential(
            nn.Linear(noise_dim, 64),
            nn.LeakyReLU(0.2),

            nn.Linear(64, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),

            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),

            nn.Linear(256, output_dim),
            nn.Tanh(),  # saída em [-1, 1], compatível com StandardScaler
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.rede(z)

    def perturba(
        self,
        x_ddos: torch.Tensor,
        epsilon: float = 0.3,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Gera e aplica perturbação sobre um batch de amostras reais de DDoS.
        Retorna as amostras perturbadas (prontas para avaliar no Defensor).
        """
        batch_size = x_ddos.shape[0]
        z = torch.randn(batch_size, self.noise_dim).to(device)
        perturbacao = self.forward(z)
        return x_ddos + epsilon * perturbacao
