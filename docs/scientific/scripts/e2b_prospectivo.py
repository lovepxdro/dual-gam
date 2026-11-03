"""E2-B: avaliação prospectiva offline de D20 congelado.

Dois treinamentos pareados de um gerador:
- proxy D19 -> avaliação em D20 (transferência temporal);
- alvo D20 -> avaliação em D20 (estresse adaptativo, NÃO transferência).

Permite inicialização aleatória ou continuação dos pesos históricos de A20.
A continuação de A20 usa um Adam novo: NÃO é uma reprodução exata da rodada 21.

Sem captura de tráfego, sem atualização do Defender e sem utilizar teste reservado.
Execute a partir da raiz do repositório ADArena.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import logging
import random
import tomllib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

LOG = logging.getLogger("e2b")


def fixar_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def hash_arquivo(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def montar_avaliacao(X_val: np.ndarray, y_val: np.ndarray,
                      seed: int, n_eval: int, noise_dim: int):
    """Reproduz exatamente o protocolo fixo de avaliação do E0."""
    ddos = X_val[y_val == 1]
    if len(ddos) == 0:
        raise ValueError("Validação sem amostras DDoS")
    n = min(int(n_eval), len(ddos))
    if n <= 0:
        raise ValueError("n_eval inválido")
    idx = np.random.default_rng(seed).choice(len(ddos), size=n, replace=False)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn(n, noise_dim, generator=generator)
    return torch.from_numpy(ddos[idx].copy()).float(), z


def medir_evasao(defender: nn.Module, x_base: torch.Tensor, z: torch.Tensor,
                 attacker: nn.Module | None, epsilon: float, threshold: float,
                 batch_size: int = 1024) -> dict:
    """Inferência isolada; não altera pesos dos modelos nem usa gradients."""
    defender.eval()
    if attacker is not None:
        attacker.eval()
    n_evasao, total, soma_prob = 0, 0, 0.0
    soma_l1, soma_l2, soma_linf = 0.0, 0.0, 0.0
    with torch.no_grad():
        for start in range(0, len(x_base), batch_size):
            xb, zb = x_base[start:start+batch_size], z[start:start+batch_size]
            perturb = (epsilon * attacker(zb)) if attacker is not None else torch.zeros_like(xb)
            output = defender(xb + perturb).view(-1)
            n_evasao += int((output < threshold).sum().item())
            total += output.numel()
            soma_prob += float(output.sum().item())
            flat = perturb.reshape(len(perturb), -1)
            soma_l1 += float(torch.linalg.vector_norm(flat, ord=1, dim=1).sum().item())
            soma_l2 += float(torch.linalg.vector_norm(flat, ord=2, dim=1).sum().item())
            soma_linf += float(torch.linalg.vector_norm(flat, ord=float("inf"), dim=1).sum().item())
    if total == 0:
        raise ValueError("Conjunto de avaliação vazio")
    return {
        "evasao": n_evasao / total,
        "confianca_ddos_media": soma_prob / total,
        "perturbacao_l1_media": soma_l1 / total,
        "perturbacao_l2_media": soma_l2 / total,
        "perturbacao_linf_media": soma_linf / total,
    }


def ler_estado(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint inválido: {path}")
    return state


def validar_baseline_historico(matriz_path: Path, historico_19: float,
                               historico_20: float, tolerancia: float = 1e-5) -> None:
    """Falha quando split, ruído, arquitetura ou pré-processamento não reproduzem o E0."""
    matriz = json.loads(matriz_path.read_text(encoding="utf-8"))
    valores = matriz["matriz_evasao"]
    if len(valores) != 20 or len(valores[-1]) != 21:
        raise AssertionError("Matriz histórica inesperada")
    esperado_19, esperado_20 = float(valores[-1][19]), float(valores[-1][20])
    if abs(historico_19 - esperado_19) > tolerancia or abs(historico_20 - esperado_20) > tolerancia:
        raise AssertionError(
            "Baseline A20 não reproduziu o E0: "
            f"D19 {historico_19:.6f} vs {esperado_19:.6f}; "
            f"D20 {historico_20:.6f} vs {esperado_20:.6f}. "
            "Confira versão do código, split, scaler, epsilon e checkpoints."
        )


def selecionar_atacante_inicial(modo: str, aleatorio: nn.Module, historico: nn.Module) -> nn.Module:
    """Seleciona pesos de inicialização, sem compartilhar modelo mutável entre braços."""
    if modo not in {"random", "a20"}:
        raise ValueError(f"Inicialização desconhecida: {modo}")
    return copy.deepcopy(historico if modo == "a20" else aleatorio)


def treinar_atacante(defender: nn.Module, X_ddos: torch.Tensor,
                     atacante_inicial: nn.Module, *, seed: int, epochs: int,
                     batch_size: int, noise_dim: int, epsilon: float,
                     lr: float, betas: tuple[float, float],
                     avaliacoes: dict[str, nn.Module], x_eval: torch.Tensor,
                     z_eval: torch.Tensor, threshold: float):
    """Um gerador novo treinado contra um Defender estritamente congelado."""
    if epochs < 1 or batch_size < 2 or len(X_ddos) < 2:
        raise ValueError("Epochs/batch/amostras inválidos")
    fixar_seed(seed)
    atacante = copy.deepcopy(atacante_inicial)
    otimizador = torch.optim.Adam(atacante.parameters(), lr=lr, betas=betas)
    rng_shuffle = torch.Generator(device="cpu").manual_seed(seed + 1)
    loader = DataLoader(TensorDataset(X_ddos), batch_size=batch_size,
                        shuffle=True, generator=rng_shuffle, drop_last=False)
    criterio = nn.BCELoss()
    defender.eval()
    for param in defender.parameters():
        param.requires_grad_(False)
    estado_antes = {k: v.detach().clone() for k, v in defender.state_dict().items()}
    curva = []
    for ep in range(1, epochs + 1):
        atacante.train()
        losses = []
        for (xb,) in loader:
            # BatchNorm da arquitetura atual requer batch com mais de 1 amostra.
            if len(xb) < 2:
                continue
            z = torch.randn(len(xb), noise_dim)
            pred = defender(xb + epsilon * atacante(z))
            alvo = torch.zeros_like(pred)
            loss = criterio(pred, alvo)
            otimizador.zero_grad()
            loss.backward()
            otimizador.step()
            losses.append(float(loss.item()))
        if not losses:
            raise RuntimeError("Nenhum batch válido no treinamento")
        for nome, modelo in avaliacoes.items():
            met = medir_evasao(modelo, x_eval, z_eval, atacante, epsilon, threshold)
            curva.append({"epoch": ep, "defender": nome,
                          "evasao": met["evasao"], "loss_treino": float(np.mean(losses))})
        LOG.info("  epoch %d/%d | loss %.4f | D20 %.2f%%", ep, epochs,
                 np.mean(losses), 100*curva[-1]["evasao"])
    for chave, original in estado_antes.items():
        if not torch.equal(original, defender.state_dict()[chave]):
            raise AssertionError(f"Defender foi modificado durante treinamento: {chave}")
    return atacante, curva


def carregar_dados(raw: dict, seed: int, run_dir: Path):
    from adarena.builtin import create_default_registry
    from gan.preprocessing import Preprocessador
    cfg_dataset = raw["attack_dataset"]
    registry = create_default_registry()
    adapter = registry.create(cfg_dataset["component_id"], **cfg_dataset.get("params", {}))
    dataset = adapter.load(cfg_dataset["source"])
    pre = Preprocessador.carregar(run_dir / "preprocessador")
    features_salvas = list(pre.feature_names)
    if features_salvas != list(dataset.schema.feature_names):
        raise ValueError("Feature order/schema não coincide com o E0")
    split = raw.get("split", {})
    X_train, X_val, _, y_train, y_val, _ = pre.split_e_normalizar(
        dataset.X, dataset.y,
        test_size=float(split.get("test_size", .2)),
        validation_size=float(split.get("validation_size", .1)),
        random_state=seed, fit_scaler=False,
    )
    return X_train, y_train, X_val, y_val


def executar(args) -> tuple[dict, list[dict]]:
    from gan.models import Defensor, Atacante
    from gan.trainer import TrainingConfig

    raw = tomllib.loads(args.config.read_text(encoding="utf-8"))
    seed = int(raw.get("seed", 42))
    run_dir = args.run_dir
    origem = json.loads((run_dir / "config_execucao.json").read_text(encoding="utf-8"))
    if origem["seed"] != seed or origem["components"]["defender"] != raw["defender"]["component_id"] or origem["components"]["attacker"] != raw["attacker"]["component_id"]:
        raise ValueError("Run E0 não corresponde à configuração selecionada")
    cfg = TrainingConfig()
    train = raw.get("training", {})
    epsilon = float(train.get("epsilon", cfg.epsilon))
    threshold = float(train.get("classification_threshold", cfg.classification_threshold))
    noise_dim = int(train.get("noise_dim", cfg.noise_dim))
    n_eval = int(train.get("amostras_avaliacao_adversarial", cfg.amostras_avaliacao_adversarial))
    lr = float(train.get("lr_atacante", cfg.lr_atacante))
    betas = tuple(train.get("adam_betas", cfg.adam_betas))
    bs = int(train.get("batch_size", cfg.batch_size))
    orig_train = origem.get("treinamento", {})
    # Parâmetros de comparação precisam reproduzir o baseline E0.
    for key, current in (("epsilon", epsilon), ("classification_threshold", threshold), ("noise_dim", noise_dim)):
        saved = (origem.get("dados", {}).get("noise_dim") if key == "noise_dim"
                 else orig_train.get(key))
        if saved is not None and float(saved) != float(current):
            raise ValueError(f"Parâmetro {key} difere do E0")
    if args.epochs < 1 or args.max_train_samples < 0:
        raise ValueError("Orçamento inválido")
    X_train, y_train, X_val, y_val = carregar_dados(raw, seed, run_dir)
    input_dim = X_train.shape[1]
    if int(origem.get("dados", {}).get("input_dim", input_dim)) != input_dim:
        raise ValueError("Dimensionamento das features incompatível com o E0")
    estado_19 = ler_estado(run_dir / "checkpoints/defensor_rodada_19.pth")
    estado_20 = ler_estado(run_dir / "checkpoints/defensor_rodada_20.pth")
    estado_a20 = ler_estado(run_dir / "checkpoints/atacante_rodada_20.pth")
    model_d19 = Defensor(input_dim=input_dim)
    model_d20 = Defensor(input_dim=input_dim)
    model_d19.load_state_dict(estado_19, strict=True)
    model_d20.load_state_dict(estado_20, strict=True)
    models = {"D19": model_d19.eval(), "D20": model_d20.eval()}
    for defender in models.values():
        for p in defender.parameters():p.requires_grad_(False)
    a20 = Atacante(noise_dim=noise_dim, output_dim=input_dim)
    a20.load_state_dict(estado_a20, strict=True)
    a20.eval()
    x_eval, z_eval = montar_avaliacao(X_val, y_val, seed, n_eval, noise_dim)
    h19 = medir_evasao(model_d19, x_eval, z_eval, a20, epsilon, threshold)["evasao"]
    h20 = medir_evasao(model_d20, x_eval, z_eval, a20, epsilon, threshold)["evasao"]
    validar_baseline_historico(run_dir / "matriz_checkpoints.json", h19, h20)
    LOG.info("Integridade do E0 confirmada: A20 x D19 %.2f%% | A20 x D20 %.2f%%", 100*h19, 100*h20)

    X_ddos = torch.from_numpy(X_train[y_train == 1].copy()).float()
    if args.max_train_samples and args.max_train_samples < len(X_ddos):
        # Mesmo subconjunto DDoS para ambos os braços, selecionado sem validação.
        indices = np.random.default_rng(args.attacker_seed + 100).choice(
            len(X_ddos), size=args.max_train_samples, replace=False)
        X_ddos = X_ddos[indices]
    if len(X_ddos) < 2:
        raise ValueError("Treino DDoS insuficiente")
    fixar_seed(args.attacker_seed)
    random_init = Atacante(noise_dim=noise_dim, output_dim=input_dim)
    init_attacker = selecionar_atacante_inicial(args.init_attacker, random_init, a20)
    if args.init_attacker == "a20":
        LOG.info("Inicialização A20 histórica: continuação dos pesos com Adam novo; sem estado original do otimizador")
    else:
        LOG.info("Inicialização aleatória: gerador novo")
    estado_inicial = {k: v.detach().clone() for k, v in init_attacker.state_dict().items()}
    row_list = []
    for atacante_nome, attacker in (("limpo", None), ("A20_historico", a20), ("aleatorio", random_init)):
        for defensor_nome, defender in models.items():
            med = medir_evasao(defender, x_eval, z_eval, attacker, epsilon, threshold)
            row_list.append({"atacante": atacante_nome, "defender": defensor_nome,
                             "epoch": 0, **med})
    treino_por_alvo = {}
    for alvo in ("D19", "D20"):
        LOG.info("Treinando Attacker inicializado em %s, alvo %s congelado", args.init_attacker, alvo)
        attacker, curve = treinar_atacante(
            models[alvo], X_ddos, init_attacker, seed=args.attacker_seed,
            epochs=args.epochs, batch_size=bs, noise_dim=noise_dim,
            epsilon=epsilon, lr=lr, betas=betas,
            avaliacoes=models, x_eval=x_eval, z_eval=z_eval,
            threshold=threshold)
        name = "novo_alvo_" + alvo
        for target, defender in models.items():
            med = medir_evasao(defender, x_eval, z_eval, attacker, epsilon, threshold)
            row_list.append({"atacante": name, "defender": target,
                             "epoch": args.epochs, **med})
        treino_por_alvo[alvo] = {"attacker": attacker, "curve": curve}
    for k, v in estado_inicial.items():
        if not torch.equal(v, init_attacker.state_dict()[k]):
            raise AssertionError(f"Estado inicial alterado entre os braços: {k}")
    resultado = {
        "experiment": "E2-B", "status": "piloto", "seed_e0": seed,
        "initialization": args.init_attacker,
        "optimizer_state": "Adam reinicializado, sem estado histórico E0",
        "attacker_seed": args.attacker_seed, "config": str(args.config),
        "run_e0": str(run_dir),
        "budgets": {"epochs": args.epochs, "n_ddos_treino": len(X_ddos), "batch_size": bs,
                    "epsilon": epsilon, "noise_dim": noise_dim,
                    "lr_atacante": lr, "betas": betas, "n_val_ddos": len(x_eval)},
        "integrity": {"a20_d19_evasao": h19, "a20_d20_evasao": h20,
                      "d19_sha256": hash_arquivo(run_dir / "checkpoints/defensor_rodada_19.pth"),
                      "d20_sha256": hash_arquivo(run_dir / "checkpoints/defensor_rodada_20.pth")},
        "protocol": "D19->D20: transferência de gerador novo; D20->D20: estresse adaptativo branco. Avaliação apenas em DDoS de validação fixo; defensor congelado.",
        "metrics": row_list,
        "curves": {k: v["curve"] for k, v in treino_por_alvo.items()},
    }
    return resultado, [(k, v["attacker"].state_dict()) for k, v in treino_por_alvo.items()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attacker-seed", type=int, default=1042)
    parser.add_argument("--init-attacker", choices=("random", "a20"), default="random",
                        help="A20 utiliza os pesos salvos no E0, com um novo otimizador Adam")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-samples", type=int, default=5000,
                        help="0 para utilizar todas as amostras DDoS do treino")
    args = parser.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir deve estar vazio, para não sobrescrever resultados")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result, states = executar(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resultados.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "resultados.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = result["metrics"]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for target, state in states:
        torch.save(state, args.output_dir / f"novo_atacante_alvo_{target.lower()}.pth")
    LOG.info("Resultados gravados em %s", args.output_dir)


if __name__ == "__main__":
    main()
