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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from gan.models import Atacante, Defensor
from gan.preprocessing import Preprocessador
from sender.sender import AttackResult, Sender
from translator.translator import Translator

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
        checkpoint_mode: str = "demo",
        attacker_round: Optional[int] = None,
        defender_round: Optional[int] = None,
    ):
        self.target_ip = target_ip
        self.target_port = target_port
        self.models_root = models_dir
        self.models_dir = self._resolver_experimento(models_dir)
        checkpoints = self.models_dir / "checkpoints"
        self.checkpoint_dir = checkpoints if checkpoints.exists() else self.models_dir
        self.device = torch.device(device)
        self.dry_run = dry_run
        self.checkpoint_mode = checkpoint_mode
        self.attacker_round = attacker_round
        self.defender_round = defender_round

        # Reutiliza os parâmetros efetivos do treinamento para manter
        # a mesma semântica de geração/classificação na execução.
        self.experiment_config = self._carregar_config_execucao()
        treino_cfg = self.experiment_config.get("treinamento", {})
        dados_cfg = self.experiment_config.get("dados", {})

        self.epsilon = float(treino_cfg.get("epsilon", 0.3))
        self.classification_threshold = float(
            treino_cfg.get("classification_threshold", 0.5)
        )
        self.noise_dim = int(dados_cfg.get("noise_dim", 32))
        self.input_dim = int(dados_cfg.get("input_dim", 77))

        # Carregar componentes.
        prep_dir = preprocessador_dir or (self.models_dir / "preprocessador")
        self.prep = Preprocessador.carregar(prep_dir)
        path_atacante, path_defensor = self._resolver_checkpoints()
        self.atacante = self._carregar_atacante(path_atacante)
        self.defensor = self._carregar_defensor(path_defensor)

        self.translator = Translator(self.prep, target_ip, target_port)
        self.sender = Sender(dry_run=dry_run)

        self.historico: list[dict] = []

        # Cada execução de attack/dry-run possui seu próprio histórico.
        # Isso evita que uma execução "final" sobrescreva, por exemplo,
        # o histórico produzido anteriormente pelo modo "demo".
        self._preparar_historico_execucao(
            path_atacante=path_atacante,
            path_defensor=path_defensor,
        )

        # Amostras reais de DDoS para servir de base às perturbações
        self.X_ddos = self._carregar_amostras_ddos(ddos_samples_path)

    def executar_ciclo(
        self,
        n_vetores: int = 100,
        min_evasao_prob: Optional[float] = None,
    ) -> list[AttackResult]:
        """
        Executa um ciclo de ataque completo.

        Args:
            n_vetores: quantos vetores gerar (o atacante tenta N, usa os que evadim)
            min_evasao_prob: threshold opcional. Se omitido, usa exatamente
                             o classification_threshold salvo no treinamento.

        Returns:
            Lista de AttackResult para cada ataque executado
        """
        logger.info("=== Ciclo de ataque | target=%s:%d ===", self.target_ip, self.target_port)

        threshold = (
            self.classification_threshold
            if min_evasao_prob is None
            else min_evasao_prob
        )

        # 1. Gerar vetores perturbados
        vetores, probs = self._gerar_e_avaliar(n_vetores)

        # 2. Filtrar os que evadem usando a mesma fronteira do treinamento.
        mask_evasao = probs < threshold
        n_evasao = mask_evasao.sum()
        logger.info(
            "Gerados: %d | Evadiram: %d (%.1f%%)",
            n_vetores, n_evasao, 100 * n_evasao / n_vetores,
        )

        taxa_evasao = n_evasao / max(n_vetores, 1)

        if n_evasao == 0:
            logger.warning("Nenhum vetor evadiu o Defensor neste ciclo")
            self._registrar_ciclo(
                resultados=[],
                n_vetores=n_vetores,
                n_evasao=0,
                n_traducoes_validas=0,
                n_traducoes_invalidas=0,
                threshold=threshold,
            )
            return []

        vetores_evasao = vetores[mask_evasao]
        probs_evasao = probs[mask_evasao]

        # 3. Traduzir todas as evasões para que plausibilidade seja medida
        #    separadamente da capacidade de enganar o Defensor.
        params_traduzidos = self.translator.traduzir_batch(
            vetores_evasao,
            evasao_probs=probs_evasao.tolist(),
            only_valid=False,
        )

        params_validos = [
            params
            for params in params_traduzidos
            if params.translation_valid
        ]
        params_invalidos = [
            params
            for params in params_traduzidos
            if not params.translation_valid
        ]

        n_validos = len(params_validos)
        n_invalidos = len(params_invalidos)
        taxa_validade = n_validos / max(n_evasao, 1)
        taxa_evasao_plausivel = n_validos / max(n_vetores, 1)

        logger.info(
            "Translator — válidos: %d/%d (%.1f%% dos evadidos) | "
            "inválidos: %d | evasão plausível: %.1f%% dos gerados",
            n_validos,
            n_evasao,
            taxa_validade * 100,
            n_invalidos,
            taxa_evasao_plausivel * 100,
        )

        if params_invalidos:
            logger.info(
                "%d vetor(es) evadiram o Defensor, mas foram rejeitados "
                "pelo Translator e não serão enviados ao Sender.",
                n_invalidos,
            )

        # 4. Executar somente traduções consideradas plausíveis.
        resultados: list[AttackResult] = []
        for i, params in enumerate(params_validos):
            logger.info(
                "Ataque válido %d/%d: %s",
                i + 1,
                len(params_validos),
                params,
            )
            result = self.sender.executar(params)
            resultados.append(result)

            if result.success:
                logger.info(
                    "  ✓ %d pkt | %.2f Mbps | %.1f pps",
                    result.packets_sent,
                    result.mbps_real,
                    result.pps_real,
                )
            else:
                logger.error("  ✗ Erro: %s", result.error)

        if not params_validos:
            logger.warning(
                "Nenhuma evasão produziu uma tradução plausível; "
                "nenhuma execução foi enviada ao Sender neste ciclo."
            )

        self._registrar_ciclo(
            resultados=resultados,
            n_vetores=n_vetores,
            n_evasao=int(n_evasao),
            n_traducoes_validas=n_validos,
            n_traducoes_invalidas=n_invalidos,
            threshold=threshold,
        )
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

        # O histórico já é persistido incrementalmente por executar_ciclo().

    # ── Internos ───────────────────────────────────────────────────────────

    def _resolver_experimento(self, models_dir: Path) -> Path:
        """Resolve /models/latest quando models_dir aponta para a raiz persistente."""
        latest = models_dir / "latest"
        if latest.exists() or latest.is_symlink():
            resolved = latest.resolve()
            logger.info("Experimento ativo: %s", resolved.name)
            return resolved

        # Compatibilidade com execuções antigas, nas quais os artefatos
        # eram salvos diretamente em /models.
        if (models_dir / "config_execucao.json").exists():
            logger.warning(
                "Estrutura legada detectada em %s; usando artefatos da raiz.",
                models_dir,
            )
            return models_dir

        raise FileNotFoundError(
            f"Nenhum experimento encontrado em {models_dir}. "
            "Execute o treinamento antes do attack/dry-run."
        )

    def _carregar_config_execucao(self) -> dict:
        path = self.models_dir / "config_execucao.json"
        if not path.exists():
            logger.warning(
                "Configuração da execução não encontrada em %s; "
                "usando defaults compatíveis.",
                path,
            )
            return {}

        with open(path, encoding="utf-8") as f:
            return json.load(f)

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
                X_base,
                epsilon=self.epsilon,
                device=str(self.device),
            )

            probs_t = self.defensor(amostras_perturbadas)

        vetores = amostras_perturbadas.cpu().numpy()
        probs = probs_t.cpu().numpy()

        return vetores, probs
        
    def _escolher_checkpoint_demonstracao(self) -> tuple[int, int]:
        """
        Seleciona o cenário de maior evasão registrada no treinamento.

        A taxa de evasão da rodada r é medida por A_r contra D_(r-1),
        antes do retreino defensivo daquela rodada.
        """

        historico_path = (
            self.models_dir
            / "historico_treino.json"
        )

        if not historico_path.exists():
            raise FileNotFoundError(
                "Histórico de treinamento não encontrado em "
                f"{historico_path}. O modo 'demo' depende das taxas "
                "de evasão registradas durante o treinamento."
            )

        with open(
            historico_path,
            encoding="utf-8",
        ) as f:
            historico = json.load(f)

        taxas = historico.get(
            "taxa_evasao_pre_adaptacao",
            historico.get("taxa_evasao", []),
        )

        if not taxas:
            raise ValueError(
                "O histórico de treinamento não contém taxas de evasão."
            )

        rodada_atacante = int(np.argmax(taxas)) + 1
        rodada_defensor = rodada_atacante - 1

        logger.info(
            "Checkpoint de demonstração: A%d × D%d | "
            "evasão registrada: %.1f%%",
            rodada_atacante,
            rodada_defensor,
            taxas[rodada_atacante - 1] * 100,
        )

        return rodada_atacante, rodada_defensor

    def _resolver_checkpoints(self) -> tuple[Path, Path]:
        """Resolve os checkpoints conforme o propósito da execução."""

        if self.checkpoint_mode == "demo":
            rodada_atacante, rodada_defensor = (
                self._escolher_checkpoint_demonstracao()
            )

            return (
                self.checkpoint_dir
                / f"atacante_rodada_{rodada_atacante:02d}.pth",
                self.checkpoint_dir
                / f"defensor_rodada_{rodada_defensor:02d}.pth",
            )

        if self.checkpoint_mode == "final":
            logger.info(
                "Checkpoint final selecionado: modelos finais da execução"
            )
            return (
                self.checkpoint_dir / "atacante_final.pth",
                self.checkpoint_dir / "defensor_adaptativo_final.pth",
            )

        if self.checkpoint_mode == "explicit":
            if self.attacker_round is None or self.defender_round is None:
                raise ValueError(
                    "O modo 'explicit' exige --attacker-round e "
                    "--defender-round."
                )

            if self.attacker_round < 1:
                raise ValueError(
                    "--attacker-round deve ser maior ou igual a 1."
                )

            if self.defender_round < 0:
                raise ValueError(
                    "--defender-round deve ser maior ou igual a 0."
                )

            logger.info(
                "Checkpoints explícitos selecionados: A%d × D%d",
                self.attacker_round,
                self.defender_round,
            )

            return (
                self.checkpoint_dir
                / f"atacante_rodada_{self.attacker_round:02d}.pth",
                self.checkpoint_dir
                / f"defensor_rodada_{self.defender_round:02d}.pth",
            )

        raise ValueError(
            f"Modo de checkpoint inválido: {self.checkpoint_mode}"
        )

    def _carregar_atacante(self, path: Path) -> Atacante:
        model = Atacante(
            noise_dim=self.noise_dim,
            output_dim=self.input_dim,
        ).to(self.device)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint do Atacante não encontrado: {path}"
            )

        model.load_state_dict(
            torch.load(
                path,
                map_location=self.device,
            )
        )

        logger.info(
            "Atacante carregado: %s",
            path,
        )

        return model

    def _carregar_defensor(self, path: Path) -> Defensor:
        model = Defensor(
            input_dim=self.input_dim
        ).to(self.device)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint do Defensor não encontrado: {path}"
            )

        model.load_state_dict(
            torch.load(
                path,
                map_location=self.device,
            )
        )

        logger.info(
            "Defensor carregado: %s",
            path,
        )

        return model

    def _preparar_historico_execucao(
        self,
        path_atacante: Path,
        path_defensor: Path,
    ) -> None:
        """Cria um arquivo de histórico exclusivo para esta execução de rede."""
        network_runs_dir = self.models_dir / "network_runs"
        network_runs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        execution_mode = "dry_run" if self.dry_run else "attack"

        checkpoint_label = self.checkpoint_mode
        if self.checkpoint_mode == "explicit":
            checkpoint_label = (
                f"explicit_A{self.attacker_round}_D{self.defender_round}"
            )

        self.network_run_id = (
            f"{execution_mode}_{checkpoint_label}_{timestamp}"
        )
        self.network_history_path = (
            network_runs_dir / f"{self.network_run_id}.json"
        )
        self.network_run_started_at = datetime.now(timezone.utc).isoformat()
        self.network_attacker_checkpoint = path_atacante.name
        self.network_defender_checkpoint = path_defensor.name

        logger.info(
            "Histórico desta execução: %s",
            self.network_history_path,
        )

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

    def _registrar_ciclo(
        self,
        resultados: list[AttackResult],
        n_vetores: int,
        n_evasao: int,
        n_traducoes_validas: int,
        n_traducoes_invalidas: int,
        threshold: float,
    ) -> None:
        """Registra evasão, plausibilidade e execução como etapas distintas."""
        taxa_evasao = n_evasao / max(n_vetores, 1)
        taxa_validade_translator = (
            n_traducoes_validas / max(n_evasao, 1)
            if n_evasao > 0
            else 0.0
        )
        taxa_evasao_plausivel = (
            n_traducoes_validas / max(n_vetores, 1)
        )

        successful = sum(bool(r.success) for r in resultados)
        entry = {
            "timestamp": time.time(),
            "classification_threshold": threshold,

            # Etapa 1 — espaço do modelo.
            "n_vetores_gerados": int(n_vetores),
            "n_evasoes": int(n_evasao),
            "taxa_evasao": taxa_evasao,

            # Etapa 2 — plausibilidade da tradução.
            "n_traducoes_validas": int(n_traducoes_validas),
            "n_traducoes_invalidas": int(n_traducoes_invalidas),
            "taxa_validade_translator": taxa_validade_translator,
            "taxa_evasao_plausivel": taxa_evasao_plausivel,

            # Etapa 3 — execução pelo Sender.
            "n_ataques_executados": len(resultados),
            "n_ataques_sucesso": successful,
            "success_rate": successful / max(len(resultados), 1),
            "packets_sent_total": sum(r.packets_sent for r in resultados),
            "bytes_sent_total": sum(r.bytes_sent for r in resultados),
            "dry_run": self.dry_run,
        }
        self.historico.append(entry)

        logger.info(
            "Resumo do ciclo — gerados=%d | evadiram=%d (%.1f%%) | "
            "plausíveis=%d (%.1f%% dos evadidos; %.1f%% dos gerados) | "
            "executados=%d | sucesso=%d",
            n_vetores,
            n_evasao,
            taxa_evasao * 100,
            n_traducoes_validas,
            taxa_validade_translator * 100,
            taxa_evasao_plausivel * 100,
            len(resultados),
            successful,
        )

        # Persistência incremental: um ciclo concluído não é perdido caso a
        # execução seja interrompida antes do final do loop.
        self._salvar_historico()

    def _salvar_historico(self) -> None:
        # Compatibilidade com testes/uso legado que instanciam o Controller
        # sem passar pelo __init__. Em execuções normais, network_history_path
        # sempre existe e cada run recebe um arquivo próprio.
        path = getattr(
            self,
            "network_history_path",
            self.models_dir / "historico_ataques.json",
        )

        if hasattr(self, "network_history_path"):
            payload = {
                "run_id": self.network_run_id,
                "experiment_id": self.models_dir.name,
                "started_at": self.network_run_started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "mode": "dry-run" if self.dry_run else "attack",
                "checkpoint_mode": self.checkpoint_mode,
                "attacker_checkpoint": self.network_attacker_checkpoint,
                "defender_checkpoint": self.network_defender_checkpoint,
                "target": {
                    "ip": self.target_ip,
                    "port": self.target_port,
                },
                "cycles": self.historico,
            }
        else:
            payload = self.historico

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Histórico salvo em %s", path)
