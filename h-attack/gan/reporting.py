"""Persistência e visualização dos resultados experimentais da Dual-GAM."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logging.getLogger("matplotlib").setLevel(logging.WARNING)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def salvar_metricas_csv(historico: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rodadas = list(range(1, len(historico["loss_atacante"]) + 1))

    defender_path = output_dir / "defender_metrics.csv"
    defender_fields = [
        "rodada", "accuracy", "precision", "recall", "f1", "fpr", "fnr", "roc_auc"
    ]
    with defender_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=defender_fields)
        writer.writeheader()
        for i, rodada in enumerate(rodadas):
            writer.writerow({
                "rodada": rodada,
                "accuracy": historico["acuracia_defensor_validacao"][i],
                "precision": historico["precision_defensor_validacao"][i],
                "recall": historico["recall_defensor_validacao"][i],
                "f1": historico["f1_defensor_validacao"][i],
                "fpr": historico["fpr_defensor_validacao"][i],
                "fnr": historico["fnr_defensor_validacao"][i],
                "roc_auc": historico["roc_auc_defensor_validacao"][i],
            })

    attacker_path = output_dir / "attacker_metrics.csv"
    attacker_fields = [
        "rodada", "evasao_pre", "evasao_pos", "reducao_evasao_pp",
        "confianca_media_pre", "confianca_media_pos", "perturbacao_l1",
        "perturbacao_l2", "perturbacao_linf", "loss_atacante", "loss_defensor",
    ]
    with attacker_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=attacker_fields)
        writer.writeheader()
        for i, rodada in enumerate(rodadas):
            writer.writerow({
                "rodada": rodada,
                "evasao_pre": historico["taxa_evasao_pre_adaptacao"][i],
                "evasao_pos": historico["taxa_evasao_pos_adaptacao"][i],
                "reducao_evasao_pp": historico["reducao_evasao_pp"][i],
                "confianca_media_pre": historico["confianca_media_adv_pre"][i],
                "confianca_media_pos": historico["confianca_media_adv_pos"][i],
                "perturbacao_l1": historico["perturbacao_l1_media"][i],
                "perturbacao_l2": historico["perturbacao_l2_media"][i],
                "perturbacao_linf": historico["perturbacao_linf_media"][i],
                "loss_atacante": historico["loss_atacante"][i],
                "loss_defensor": historico["loss_defensor"][i],
            })

    return {"defender": defender_path, "attacker": attacker_path}


def salvar_summary(config: dict, historico: dict, output_path: Path) -> None:
    teste = historico["metricas_defensor_teste_final"]
    matriz = historico["avaliacao_cruzada_checkpoints"]
    final = matriz["evasao_contra_defensor_final"]
    summary = {
        "run_id": config["run_id"],
        "seed": config["seed"],
        "dataset": config["dataset"],
        "n_rodadas": config["treinamento"]["n_rodadas"],
        "evasao_final_pre_adaptacao": historico["taxa_evasao_pre_adaptacao"][-1],
        "evasao_final_pos_adaptacao": historico["taxa_evasao_pos_adaptacao"][-1],
        "teste_final": teste,
        "robustez_acumulada": {
            "evasao_media_d_final": float(np.mean(final)),
            "pior_evasao_d_final": float(np.max(final)),
            "pior_atacante_d_final": int(np.argmax(final)) + 1,
        },
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def gerar_graficos(historico: dict, plots_dir: Path) -> list[Path]:
    plots_dir.mkdir(parents=True, exist_ok=True)
    rodadas = np.arange(1, len(historico["loss_atacante"]) + 1)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rodadas, np.array(historico["taxa_evasao_pre_adaptacao"]) * 100, marker="o", label="Pré-adaptação")
    ax.plot(rodadas, np.array(historico["taxa_evasao_pos_adaptacao"]) * 100, marker="o", label="Pós-adaptação")
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Taxa de evasão (%)")
    ax.set_title("Evasão adversarial pré e pós-adaptação")
    ax.legend()
    ax.grid(True, alpha=0.25)
    p = plots_dir / "evasao_pre_pos.png"; _save(fig, p); paths.append(p)

    fig, ax = plt.subplots(figsize=(9, 5))
    for key, label in [
        ("acuracia_defensor_validacao", "Accuracy"),
        ("precision_defensor_validacao", "Precision"),
        ("recall_defensor_validacao", "Recall"),
        ("f1_defensor_validacao", "F1"),
    ]:
        ax.plot(rodadas, np.array(historico[key]) * 100, label=label)
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Métrica (%)")
    ax.set_title("Desempenho convencional do Defensor")
    ax.legend()
    ax.grid(True, alpha=0.25)
    p = plots_dir / "metricas_defensor.png"; _save(fig, p); paths.append(p)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rodadas, np.array(historico["fpr_defensor_validacao"]) * 100, label="FPR")
    ax.plot(rodadas, np.array(historico["fnr_defensor_validacao"]) * 100, label="FNR")
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Taxa (%)")
    ax.set_title("Falsos positivos e falsos negativos")
    ax.legend()
    ax.grid(True, alpha=0.25)
    p = plots_dir / "fpr_fnr.png"; _save(fig, p); paths.append(p)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rodadas, historico["loss_atacante"], label="Atacante")
    ax.plot(rodadas, historico["loss_defensor"], label="Defensor")
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Loss")
    ax.set_title("Loss por rodada")
    ax.legend()
    ax.grid(True, alpha=0.25)
    p = plots_dir / "losses.png"; _save(fig, p); paths.append(p)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(rodadas, historico["perturbacao_l1_media"], label="L1")
    ax.plot(rodadas, historico["perturbacao_l2_media"], label="L2")
    ax.plot(rodadas, historico["perturbacao_linf_media"], label="L∞")
    ax.set_xlabel("Rodada")
    ax.set_ylabel("Norma média")
    ax.set_title("Magnitude das perturbações")
    ax.legend()
    ax.grid(True, alpha=0.25)
    p = plots_dir / "perturbacoes.png"; _save(fig, p); paths.append(p)

    cruzada = historico["avaliacao_cruzada_checkpoints"]
    matriz = np.array(cruzada["matriz_evasao"]) * 100
    fig, ax = plt.subplots(figsize=(12, 8))
    image = ax.imshow(matriz, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Defensor")
    ax.set_ylabel("Atacante")
    ax.set_title("Matriz de evasão Aᵢ × Dⱼ (%)")
    ax.set_xticks(np.arange(len(cruzada["colunas_defensores"])))
    ax.set_xticklabels([f"D{x}" for x in cruzada["colunas_defensores"]], rotation=90)
    ax.set_yticks(np.arange(len(cruzada["linhas_atacantes"])))
    ax.set_yticklabels([f"A{x}" for x in cruzada["linhas_atacantes"]])
    fig.colorbar(image, ax=ax, label="Evasão (%)")
    p = plots_dir / "matriz_checkpoints_heatmap.png"; _save(fig, p); paths.append(p)

    return paths
