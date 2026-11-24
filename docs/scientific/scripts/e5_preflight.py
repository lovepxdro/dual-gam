"""E5-P — auditoria de checkpoints A20/D20, scaler e identidade A0/A1.

Somente leitura dos artefatos E0. Não realiza treinamento ou inferência
sobre dados de teste. Executar na raiz do repositório com `uv run python`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from gan.e5_conditional import AtacanteCondicional
from gan.models import Atacante, Defensor
from gan.preprocessing import Preprocessador


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def localizar_run(runs_root: Path, seed: int) -> Path:
    encontrados = []
    for path in sorted(runs_root.glob(f"run_*_seed{seed}")):
        config_path = path / "config_execucao.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if int(config.get("seed", -1)) == seed:
                encontrados.append(path)
    if len(encontrados) != 1:
        raise RuntimeError(
            f"Seed {seed}: esperado exatamente um E0 em {runs_root}, "
            f"encontrados {len(encontrados)}; confira os runs."
        )
    return encontrados[0]


def conferir_auditoria(pre: Preprocessador) -> dict:
    audit = pre.audit_info
    split = audit.get("split", {})
    for name in ("treino", "validacao", "teste"):
        if int(split.get(name, 0)) <= 0:
            raise AssertionError(f"Split E0 inválido ou ausente: {name}")
    intersecoes = audit.get("intersecoes", {})
    for name in ("treino_validacao", "treino_teste", "validacao_teste"):
        if name not in intersecoes or int(intersecoes[name]) != 0:
            raise AssertionError(f"Auditoria E0 não comprova isolamento: {name}")
    if audit.get("scaler_fit_nesta_execucao") is not True:
        raise AssertionError("E0 não confirma scaler ajustado apenas no treino")
    return {"split": split, "intersecoes": intersecoes}


def auditar_seed(run_dir: Path, seed: int, *, batch: int = 16, epsilon: float = 0.3) -> dict:
    if batch < 2 or epsilon <= 0:
        raise ValueError("Batch deve ser >=2 e epsilon >0")
    config_path = run_dir / "config_execucao.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if int(config.get("seed", -1)) != seed:
        raise AssertionError("Seed não corresponde ao config_execucao.json")
    components = config.get("components", {})
    if components.get("attacker") != "adarena.perturbation_attacker" or components.get("defender") != "adarena.binary_mlp_defender":
        raise AssertionError("Componentes E0 diferentes do protocolo E5")
    dados = config["dados"]
    input_dim = int(dados["input_dim"])
    noise_dim = int(dados["noise_dim"])
    if float(config["treinamento"]["epsilon"]) != float(epsilon):
        raise AssertionError("Epsilon diferente do E0")
    if float(config["treinamento"]["classification_threshold"]) != 0.5:
        raise AssertionError("Limiar diferente do E0")
    pre_dir = run_dir / "preprocessador"
    pre = Preprocessador.carregar(pre_dir)
    if len(pre.feature_names) != input_dim or pre.feature_names != list(dados["feature_names"]):
        raise AssertionError("Nomes/ordem das features incompatíveis")
    scaler = pre.scaler
    means = np.asarray(scaler.mean_)
    scales = np.asarray(scaler.scale_)
    if (len(means) != input_dim or len(scales) != input_dim or
        not np.isfinite(means).all() or not np.isfinite(scales).all() or
        not np.all(scales > 0)):
        raise AssertionError("Scaler E0 inconsistente")
    hist = conferir_auditoria(pre)

    caminho_a = run_dir / "checkpoints" / "atacante_rodada_20.pth"
    caminho_d = run_dir / "checkpoints" / "defensor_rodada_20.pth"
    estado_a = torch.load(caminho_a, map_location="cpu", weights_only=True)
    estado_d = torch.load(caminho_d, map_location="cpu", weights_only=True)
    a0 = Atacante(noise_dim=noise_dim, output_dim=input_dim)
    a0.load_state_dict(estado_a, strict=True)
    a1 = AtacanteCondicional(noise_dim=noise_dim, input_dim=input_dim)
    a1.carregar_base_a20(estado_a)
    d20 = Defensor(input_dim=input_dim)
    d20.load_state_dict(estado_d, strict=True)
    a0.eval()
    a1.eval()
    d20.eval()
    d20.requires_grad_(False)

    # Geradores isolados preservam o RNG global e permitem reprodução em reruns.
    g = torch.Generator(device="cpu").manual_seed(9000 + seed)
    x = torch.randn(batch, input_dim, generator=g)
    z = torch.randn(batch, noise_dim, generator=g)
    with torch.no_grad():
        delta0 = a0(z)
        delta1 = a1(x, z)
        delta1_alt = a1(x + 0.5, z)
        diferenca = float((delta1 - delta0).abs().max())
        if diferenca > 1e-7 or not torch.allclose(delta1, delta1_alt, atol=1e-7, rtol=0):
            raise AssertionError("A1 não reproduziu exatamente A0 na inicialização")
        if not torch.isfinite(delta1).all() or float(delta1.abs().max()) > 1 + 1e-6:
            raise AssertionError("Saída A1 viola [-1,1] ou contém não-finito")
        if float((a1.perturba(x, z, epsilon) - x).abs().max()) > epsilon + 1e-6:
            raise AssertionError("Orçamento epsilon violado")

    # D20 congelado; o grafo ainda precisa fluir até o gerador.
    for param in a1.parameters():
        param.grad = None
    y = d20(x + epsilon * a1(x, z))
    y.mean().backward()
    if any(param.grad is not None for param in d20.parameters()):
        raise AssertionError("Gradientes indevidos no Defender")
    if a1.ramo[-1].weight.grad is None or a1.base.rede[0].weight.grad is None:
        raise AssertionError("Gradiente não alcançou o ramo condicional ou backbone")

    return {
        "seed": seed,
        "run_dir": str(run_dir),
        "run_id": config.get("run_id"),
        "dataset_origem": config.get("dataset"),
        "n_features": input_dim,
        "noise_dim": noise_dim,
        "epsilon": epsilon,
        "limiar": 0.5,
        "split_original": hist["split"],
        "intersecoes_e0": hist["intersecoes"],
        "sha256_a20": sha256(caminho_a),
        "sha256_d20": sha256(caminho_d),
        "sha256_scaler": sha256(pre_dir / "scaler.pkl"),
        "params_a0": sum(p.numel() for p in a0.parameters()),
        "params_a1": sum(p.numel() for p in a1.parameters()),
        "maior_diferenca_inicial_a0_a1": diferenca,
        "max_abs_delta": float(delta1.abs().max()),
        "gradiente_ramo": a1.ramo[-1].weight.grad is not None,
        "gradiente_base": a1.base.rede[0].weight.grad is not None,
        "defender_congelado": True,
        "status": "aprovado",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("models/e0/experiments"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20, 42])
    parser.add_argument("--output-dir", type=Path, default=Path("models/e5/preflight"))
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Seeds duplicadas")
    path_final = args.output_dir / "e5_preflight.json"
    if path_final.exists():
        parser.error(f"Arquivo já existe; não sobrescrever: {path_final}")
    results = []
    for seed in args.seeds:
        run = localizar_run(args.runs_root, seed)
        print(f"[E5-P] Seed {seed}: {run}", flush=True)
        info = auditar_seed(run, seed)
        results.append(info)
        print(f"[E5-P] OK: identidade A0=A1 (erro {info['maior_diferenca_inicial_a0_a1']:.2e}); D20 congelado", flush=True)
    output = {
        "experimento": "E5-P",
        "status": "aprovado",
        "observacao": "Sem treino e sem avaliação de ataque. Auditoria de splits baseada nos registros E0 persistidos, não em reexecução do split.",
        "seeds": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path_final.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[E5-P] Salvo em {path_final}")


if __name__ == "__main__":
    main()
