"""Testes de contrato do E5-P."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "docs/scientific/scripts/e5_preflight.py"
_spec = importlib.util.spec_from_file_location("e5_preflight", SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_localizar_run_usa_seed_exata(tmp_path):
    for seed in (1, 10):
        d = tmp_path / f"run_teste_seed{seed}"
        d.mkdir()
        (d / "config_execucao.json").write_text(json.dumps({"seed": seed}))
    assert mod.localizar_run(tmp_path, 1).name == "run_teste_seed1"
    with pytest.raises(RuntimeError):
        mod.localizar_run(tmp_path, 2)


def test_localizar_run_rejeita_duplicidade(tmp_path):
    for suffix in ("a", "b"):
        d = tmp_path / f"run_{suffix}_seed42"
        d.mkdir()
        (d / "config_execucao.json").write_text('{"seed": 42}')
    with pytest.raises(RuntimeError, match="exatamente um"):
        mod.localizar_run(tmp_path, 42)


def test_auditoria_split_valido():
    pre = SimpleNamespace(audit_info={
        "split": {"treino": 70, "validacao": 10, "teste": 20},
        "intersecoes": {"treino_validacao": 0, "treino_teste": 0, "validacao_teste": 0},
        "scaler_fit_nesta_execucao": True,
    })
    assert mod.conferir_auditoria(pre)["split"]["teste"] == 20


def test_auditoria_isolamento_pendencia_bloqueia():
    pre = SimpleNamespace(audit_info={
        "split": {"treino": 70, "validacao": 10, "teste": 20},
        "intersecoes": {"treino_validacao": 0, "treino_teste": 1, "validacao_teste": 0},
        "scaler_fit_nesta_execucao": True,
    })
    with pytest.raises(AssertionError):
        mod.conferir_auditoria(pre)
