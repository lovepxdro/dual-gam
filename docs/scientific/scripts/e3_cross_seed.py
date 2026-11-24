"""E3: piloto de transferência cross-seed, somente inferência offline.

Com 2 seeds, usa a MESMA interseção dos testes para diagonal e cross-seed.
Com mais seeds, cada par usa a interseção dos testes de origem e destino,
com referências internas avaliadas na mesma base do par.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from e3_preflight import preparar, resumo, hash_indices

LOG = logging.getLogger("e3-cross")


def noise_for_pair(source: int, target: int, noise_seed: int, n: int, dim: int) -> torch.Tensor:
    """Par ordenado reverso usa o mesmo ruído com o mesmo N na interseção comum."""
    a, b = sorted((source, target))
    raw = f"E3|{a}|{b}|{noise_seed}".encode("ascii")
    seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, dim, generator=gen)


def select_eval_ids(test_a: np.ndarray, test_b: np.ndarray, y: np.ndarray,
                    a: int, b: int, selection_seed: int, max_samples: int) -> np.ndarray:
    if max_samples < 1:
        raise ValueError("max_samples deve ser >= 1")
    common = np.intersect1d(test_a, test_b, assume_unique=True)
    ddos = common[y[common] == 1]
    if len(ddos) == 0:
        raise RuntimeError(f"Sem DDoS em teste comum entre seeds {a} e {b}")
    # Sorteio usando par não ordenado: duas direções usam exatamente a mesma base.
    low, high = sorted((a, b))
    seed = int.from_bytes(hashlib.sha256(f"{low}|{high}|{selection_seed}".encode()).digest()[:8], "little")
    indices = np.random.default_rng(seed).choice(ddos, size=min(max_samples, len(ddos)), replace=False)
    return np.sort(indices)


def converter_perturbacao(x_raw: np.ndarray, delta: np.ndarray, src_scaler, dst_scaler, epsilon: float):
    """Perturbar em z-score da origem -> escala bruta -> z-score do destino.

    As entradas/saídas são float32 para preservar a semântica de inferência do E0;
    cálculos intermediários em float64 evitam perda de precisão na ida/volta.
    """
    x_raw = np.asarray(x_raw, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    if x_raw.shape != delta.shape:
        raise ValueError("Dimensões diferentes entre amostra e perturbação")
    if not np.isfinite(x_raw).all() or not np.isfinite(delta).all():
        raise ValueError("Valores não finitos")
    src_norm = src_scaler.transform(x_raw)
    adv_norm = src_norm + epsilon * delta
    physical = src_scaler.inverse_transform(adv_norm)
    dst_norm = dst_scaler.transform(physical).astype(np.float32)
    if not np.isfinite(dst_norm).all():
        raise ValueError("Transformação produziu valores não finitos")
    return dst_norm, physical


def ler_checkpoint(modelo, file: Path):
    model_state = torch.load(file, map_location="cpu", weights_only=True)
    modelo.load_state_dict(model_state, strict=True)
    modelo.eval()
    for p in modelo.parameters():
        p.requires_grad_(False)
    return modelo


def carregar_modelos(runs):
    from gan.models import Atacante, Defensor
    models = {}
    for seed, run in runs.items():
        params = run.config["treinamento"]
        noise_dim = int(run.config["dados"]["noise_dim"])
        input_dim = int(run.config["dados"]["input_dim"])
        folder = run.path / "checkpoints"
        models[seed] = {
            "A3": ler_checkpoint(Atacante(noise_dim=noise_dim, output_dim=input_dim), folder / "atacante_rodada_03.pth"),
            "A20": ler_checkpoint(Atacante(noise_dim=noise_dim, output_dim=input_dim), folder / "atacante_rodada_20.pth"),
            "D2": ler_checkpoint(Defensor(input_dim=input_dim), folder / "defensor_rodada_02.pth"),
            "D19": ler_checkpoint(Defensor(input_dim=input_dim), folder / "defensor_rodada_19.pth"),
            "D20": ler_checkpoint(Defensor(input_dim=input_dim), folder / "defensor_rodada_20.pth"),
        }
    return models


def avaliar_modelo(attacker, defender, src_scaler, dst_scaler, x_raw: np.ndarray,
                   z: torch.Tensor, epsilon: float, threshold: float, batch: int = 1024) -> dict:
    """Avalia N amostras DDoS limpas e adversariais; taxa condicionada às limpas detectadas."""
    n, n_evasion, n_clean_miss, n_success, n_detected, sum_prob = 0, 0, 0, 0, 0, 0.0
    l1, l2, linf, l1_raw, linf_raw = 0.0, 0.0, 0.0, 0.0, 0.0
    attacker.eval(); defender.eval()
    with torch.no_grad():
        for st in range(0, len(x_raw), batch):
            raw = x_raw[st:st + batch]
            zb = z[st:st + batch]
            delta = attacker(zb).detach().cpu().numpy()
            adv, physical = converter_perturbacao(raw, delta, src_scaler, dst_scaler, epsilon)
            clean = dst_scaler.transform(np.asarray(raw, dtype=np.float64)).astype(np.float32)
            pred_clean = defender(torch.from_numpy(clean)).view(-1).cpu().numpy()
            pred_adv = defender(torch.from_numpy(adv)).view(-1).cpu().numpy()
            correctly = pred_clean >= threshold
            escaped = pred_adv < threshold
            magnitude_normalized = epsilon * delta
            magnitude_physical = physical - np.asarray(raw, dtype=np.float64)
            n += len(raw)
            n_evasion += int(escaped.sum())
            n_clean_miss += int((~correctly).sum())
            n_detected += int(correctly.sum())
            n_success += int(np.sum(correctly & escaped))
            sum_prob += float(np.sum(pred_adv, dtype=np.float64))
            l1 += float(np.linalg.norm(magnitude_normalized, ord=1, axis=1).sum())
            l2 += float(np.linalg.norm(magnitude_normalized, ord=2, axis=1).sum())
            linf += float(np.max(np.abs(magnitude_normalized), axis=1).sum())
            l1_raw += float(np.linalg.norm(magnitude_physical, ord=1, axis=1).sum())
            linf_raw += float(np.max(np.abs(magnitude_physical), axis=1).sum())
    return {
        "n": n, "evasao": n_evasion / n, "evasoes": n_evasion,
        "evasao_limpa": n_clean_miss / n, "sucesso_condicional": n_success / n_detected if n_detected else None,
        "n_detectadas_limpas": n_detected, "n_sucesso": n_success,
        "confianca_ddos_adv": sum_prob / n, "l1_norm_src": l1 / n,
        "l2_norm_src": l2 / n, "linf_norm_src": linf / n,
        "l1_bruto": l1_raw / n, "linf_bruto": linf_raw / n,
    }


def executar(X, y, runs, *, n_eval, noise_seed, selection_seed):
    models = carregar_modelos(runs)
    rows = []
    eps = float(next(iter(runs.values())).config["treinamento"]["epsilon"])
    threshold = float(next(iter(runs.values())).config["treinamento"]["classification_threshold"])
    noise_dim = int(next(iter(runs.values())).config["dados"]["noise_dim"])
    # Com 2 seeds: compartilha a mesma interseção de testes INCLUSIVE na diagonal.
    # Na campanha com mais seeds, cada par dispõe de base própria e das referências pareadas.
    seeds = sorted(runs)
    shared_two = None
    if len(seeds) == 2:
        shared_two = select_eval_ids(runs[seeds[0]].test_ids, runs[seeds[1]].test_ids, y,
                                     seeds[0], seeds[1], selection_seed, n_eval)
        LOG.info("Piloto 2 seeds: %d DDoS em teste comum para TODOS os pares", len(shared_two))
    for source in seeds:
        for target in seeds:
            idx = shared_two if shared_two is not None else select_eval_ids(
                runs[source].test_ids, runs[target].test_ids, y, source, target, selection_seed, n_eval)
            base = np.asarray(X[idx])
            z = noise_for_pair(source, target, noise_seed, len(idx), noise_dim)
            for ar, dr in ((3, 2), (20, 19)):
                attacker = models[source][f"A{ar}"]
                src_pre = runs[source].pre.scaler
                for test_name, defender_seed, defender_name in (
                    ("destino", target, "D20"),
                    ("controle_mesma_trajetoria", source, "D20"),
                    ("efetividade_na_origem", source, f"D{dr}"),
                ):
                    defender = models[defender_seed][defender_name]
                    target_pre = runs[defender_seed].pre.scaler
                    met = avaliar_modelo(attacker, defender, src_pre, target_pre, base, z, eps, threshold)
                    rows.append({"seed_origem": source, "seed_destino": target, "checkpoint_atacante": f"A{ar}",
                                 "avaliacao": test_name, "defender_avaliado": f"D{dr}" if test_name == "efetividade_na_origem" else "D20",
                                 "seed_defender_avaliado": defender_seed,
                                 "base_test_ids_sha256": hash_indices(idx), "noise_seed": noise_seed,
                                 "epsilon": eps, "threshold": threshold, **met})
                    if test_name == "destino":
                        LOG.info("A%d seed %d → D20 seed %d | N=%d evasão=%.2f%% limpa=%.2f%%",
                                 ar, source, target, len(idx), 100*met["evasao"], 100*met["evasao_limpa"])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=Path("models/e0/experiments"))
    p.add_argument("--config-dir", type=Path, default=Path("configs/e0"))
    p.add_argument("--seeds", nargs="+", type=int, default=[20, 42])
    p.add_argument("--n-eval", type=int, default=5000)
    p.add_argument("--noise-seed", type=int, default=2026)
    p.add_argument("--selection-seed", type=int, default=2026)
    p.add_argument("--output-dir", type=Path, default=Path("models/e3/piloto"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    seeds = tuple(sorted(set(args.seeds)))
    if len(seeds) < 2:
        raise ValueError("E3 exige no mínimo duas seeds")
    X, y, features, runs, n_removed = preparar(seeds, args.run_root, args.config_dir)
    info = resumo(X, y, features, runs, n_removed)
    rows = executar(X, y, runs, n_eval=args.n_eval, noise_seed=args.noise_seed,
                    selection_seed=args.selection_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "preflight.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "resultados.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {"experiment": "E3", "status": "piloto" if len(seeds) == 2 else "campanha",
               "seeds": list(seeds), "comparacoes_unicas": len(seeds)**2 * 2,
               "avaliacoes_incluindo_controles": len(rows), "n_eval_max": args.n_eval,
               "noise_seed": args.noise_seed, "selection_seed": args.selection_seed,
               "same_sample_diagonal_for_two_seed_pilot": len(seeds) == 2,
               "metodologia": "Interseção teste/teste, escala bruta intermediária; 3 avaliações por par/checkpoint",
               "results": rows}
    (args.output_dir / "resultados.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    LOG.info("E3 finalizado: %d pares/checkpoints e %d avaliações totais", len(seeds)**2*2, len(rows))
    LOG.info("Resultados em %s", args.output_dir)

if __name__ == "__main__":
    main()
