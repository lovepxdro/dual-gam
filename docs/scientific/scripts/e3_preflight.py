"""E3-P: auditoria reproduzível de splits, scalers e checkpoints do E0.

Não treina modelos e não altera nenhum arquivo do E0.
Execute da raiz da ADArena com `uv run python docs/scientific/scripts/e3_preflight.py ...`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("e3-preflight")
SEEDS_E0 = tuple(range(1, 11)) + (20, 40, 42, 80, 100)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_indices(ids: np.ndarray) -> str:
    return sha256_bytes(np.asarray(ids, dtype="<i8").tobytes())


def split_indices(y: np.ndarray, seed: int, test_size: float, validation_size: float):
    """Reproduz exatamente as duas chamadas train_test_split de Preprocessador."""
    if not (0 < test_size < 1 and 0 < validation_size < 1 and test_size + validation_size < 1):
        raise ValueError("Proporções inválidas")
    ids = np.arange(len(y), dtype=np.int64)
    temp, test = train_test_split(ids, test_size=test_size, random_state=seed, stratify=y)
    train, val = train_test_split(
        temp,
        test_size=validation_size / (1 - test_size),
        random_state=seed,
        stratify=y[temp],
    )
    if len(set(train) & set(val)) or len(set(train) & set(test)) or len(set(val) & set(test)):
        raise AssertionError("Vazamento entre splits")
    return train, val, test


def indices_sem_duplicatas(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mesmo critério (features + rótulo) e keep=first do Preprocessador."""
    df = pd.DataFrame(X)
    df["_label"] = y
    return np.flatnonzero(~df.duplicated(keep="first").to_numpy())


def encontrar_run(root: Path, seed: int) -> Path:
    found = []
    for path in sorted(root.glob(f"run_*_seed{seed}")):
        cfg = path / "config_execucao.json"
        if not cfg.is_file():
            continue
        if int(json.loads(cfg.read_text(encoding="utf-8")).get("seed", -1)) == seed:
            found.append(path)
    if len(found) != 1:
        raise RuntimeError(f"Seed {seed}: esperado 1 run E0, encontrados {len(found)}: {found}")
    return found[0]


def conferir_scaler(scaler, X_train: np.ndarray, *, rtol: float = 3e-6) -> None:
    """Compara todos os atributos aprendidos a um fit independente no split reconstruído."""
    novo = StandardScaler().fit(X_train)
    for attr in ("mean_", "var_", "scale_"):
        a, b = np.asarray(getattr(scaler, attr)), np.asarray(getattr(novo, attr))
        if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=1e-8, equal_nan=False):
            abs_err = float(np.max(np.abs(a - b))) if a.shape == b.shape else float("inf")
            raise AssertionError(f"Scaler não corresponde ao split ({attr}; erro máx {abs_err})")
    if int(scaler.n_samples_seen_) != len(X_train):
        raise AssertionError("Scaler n_samples_seen_ não corresponde ao treino reconstruído")


@dataclass
class RunInfo:
    seed: int
    path: Path
    config: dict[str, Any]
    pre: Any
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray


def carregar_base(config_dir: Path, seeds: tuple[int, ...]):
    import tomllib
    from adarena.builtin import create_default_registry
    if not seeds:
        raise ValueError("Nenhuma seed selecionada")
    raws = {s: tomllib.loads((config_dir / f"seed-{s}.toml").read_text(encoding="utf-8")) for s in seeds}
    sources = {(raw["attack_dataset"]["source"], raw["attack_dataset"]["component_id"],
                json.dumps(raw["attack_dataset"].get("params", {}), sort_keys=True)) for raw in raws.values()}
    if len(sources) != 1:
        raise ValueError("Seeds usam datasets/adaptadores diferentes")
    raw = raws[seeds[0]]["attack_dataset"]
    adapter = create_default_registry().create(raw["component_id"], **raw.get("params", {}))
    base = adapter.load(raw["source"])
    X, y = np.asarray(base.X), np.asarray(base.y)
    if len(X) != len(y) or X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError("Dataset inválido")
    keep = indices_sem_duplicatas(X, y)
    n_removed = len(X) - len(keep)
    X, y = X[keep], y[keep]
    return base, X, y, raws, n_removed


def validar_run(seed: int, root: Path, X: np.ndarray, y: np.ndarray,
                 feature_names: list[str], raw: dict, n_removed: int):
    from gan.preprocessing import Preprocessador
    path = encontrar_run(root, seed)
    cfg = json.loads((path / "config_execucao.json").read_text(encoding="utf-8"))
    if cfg["mode"] != "train" or int(cfg["seed"]) != seed:
        raise ValueError(f"Seed {seed}: origem não é TRAIN E0")
    if cfg["components"]["attacker"] != raw["attacker"]["component_id"] or cfg["components"]["defender"] != raw["defender"]["component_id"]:
        raise ValueError(f"Seed {seed}: IDs de componentes incompatíveis")
    dados = cfg["dados"]
    if list(dados["feature_names"]) != feature_names or int(dados["input_dim"]) != X.shape[1]:
        raise ValueError(f"Seed {seed}: features incompatíveis")
    pre = Preprocessador.carregar(path / "preprocessador")
    if list(pre.feature_names) != feature_names:
        raise ValueError(f"Seed {seed}: ordem de features no preprocessador diverge")
    split = raw.get("split", {})
    ts = float(split.get("test_size", .2)); vs = float(split.get("validation_size", .1))
    if abs(ts - float(dados["test_size"])) > 1e-12 or abs(vs - float(dados["validation_size"])) > 1e-12:
        raise ValueError(f"Seed {seed}: proporção de split difere do E0")
    train, val, test = split_indices(y, seed, ts, vs)
    audit = pre.audit_info
    if not audit or int(audit.get("duplicatas_removidas", -1)) != n_removed:
        raise AssertionError(f"Seed {seed}: auditoria de duplicatas incompatível")
    if any(int(audit["split"][part]) != len(ids) for part, ids in (("treino", train), ("validacao", val), ("teste", test))):
        raise AssertionError(f"Seed {seed}: tamanhos dos splits divergem")
    if int(audit.get("random_state", -1)) != seed:
        raise AssertionError(f"Seed {seed}: random_state divergente")
    conferir_scaler(pre.scaler, X[train])
    saved_train = cfg["treinamento"]
    current = raw.get("training", {})
    for key in ("epsilon", "classification_threshold", "noise_dim"):
        stored = dados["noise_dim"] if key == "noise_dim" else saved_train[key]
        if key in current and float(current[key]) != float(stored):
            raise ValueError(f"Seed {seed}: {key} difere da configuração do E0")
    for name in ("atacante_rodada_03.pth", "atacante_rodada_20.pth", "defensor_rodada_02.pth", "defensor_rodada_19.pth", "defensor_rodada_20.pth"):
        if not (path / "checkpoints" / name).is_file():
            raise FileNotFoundError(path / "checkpoints" / name)
    return RunInfo(seed, path, cfg, pre, train, val, test)


def preparar(seeds: tuple[int, ...], run_root: Path, config_dir: Path):
    base, X, y, raws, n_removed = carregar_base(config_dir, seeds)
    features = list(base.schema.feature_names)
    runs = {s: validar_run(s, run_root, X, y, features, raws[s], n_removed) for s in seeds}
    canonical = {(r.config["treinamento"]["epsilon"], r.config["treinamento"]["classification_threshold"], r.config["dados"]["noise_dim"]) for r in runs.values()}
    if len(canonical) != 1:
        raise ValueError("Treinamentos com eps/limiar/ruído diferentes")
    return X, y, features, runs, n_removed


def resumo(X, y, features, runs, n_removed):
    return {
        "experiment": "E3", "stage": "preflight", "status": "validado", "rows_after_dedup": len(X),
        "dedup_removed": n_removed, "features": len(features), "seeds": sorted(runs),
        "runs": [{"seed": s, "path": str(r.path), "n_train": len(r.train_ids), "n_val": len(r.val_ids),
                  "n_test": len(r.test_ids), "test_ids_sha256": hash_indices(r.test_ids),
                  "scaler_sha256": sha256_file(r.path / "preprocessador" / "scaler.pkl")}
                 for s, r in sorted(runs.items())],
        "pair_overlap_ddos": [
            {"source": s, "target": t, "n": int(np.sum(y[np.intersect1d(a.test_ids, b.test_ids)] == 1))}
            for s, a in sorted(runs.items()) for t, b in sorted(runs.items()) if s != t
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, default=Path("models/e0/experiments"))
    p.add_argument("--config-dir", type=Path, default=Path("configs/e0"))
    p.add_argument("--seeds", type=int, nargs="+", default=[20, 42])
    p.add_argument("--output-dir", type=Path, default=Path("models/e3/piloto"))
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    seeds = tuple(sorted(set(args.seeds)))
    X, y, features, runs, n_removed = preparar(seeds, args.run_root, args.config_dir)
    result = resumo(X, y, features, runs, n_removed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dest = args.output_dir / "preflight.json"
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("E3-P validado para %d seeds | %d features | %d amostras", len(seeds), len(features), len(X))
    for row in result["pair_overlap_ddos"]:
        LOG.info("A seed %d → D seed %d | %d DDoS no teste comum", row["source"], row["target"], row["n"])
    LOG.info("Relatório: %s", dest)


if __name__ == "__main__":
    main()
