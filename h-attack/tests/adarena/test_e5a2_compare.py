"""Verifica recusas de comparações inválidas e métricas pareadas E5-A2."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'docs' / 'scientific' / 'scripts'))
from e5a2_compare import CONSTRAINTS, compare_dirs


def test_comparison_accepts_identical_protocol_and_computes_difference_of_differences(tmp_path):
    base, log = tmp_path/'old',tmp_path/'new'
    base.mkdir();log.mkdir()
    cfg = {k: 1 for k in CONSTRAINTS}
    cfg.update(experiment='E5-A')
    (base/'config_e5a.json').write_text(json.dumps(cfg))
    new_cfg = {**cfg, 'experiment':'E5-A2', 'loss_mode':'logits'}
    (log/'config_e5a2.json').write_text(json.dumps(new_cfg))
    rows_old, rows_new = [], []
    for arm in ('A0','A1'):
        common = dict(seed_e0=20, replica=1042, braco=arm, seed_defender_transferencia=42,
                      n_treino_ddos=100, n_teste_comum_ddos=50, asr_transferencia=0.,
                      diversidade_l2_alvo=4.)
        rows_old.append({**common, 'asr_alvo':0.01 if arm=='A0' else 0.02})
        rows_new.append({**common, 'asr_alvo':0.04 if arm=='A0' else 0.08})
    summary = {'seeds':[20,42], 'replicas':[1042], 'n_teste_comum_ddos':50,
               'ids_teste_comum_sha256':'fake', 'resultados': rows_old}
    (base/'resumo_e5a.json').write_text(json.dumps(summary))
    (log/'resumo_e5a.json').write_text(json.dumps({**summary, 'resultados':rows_new}))
    result = compare_dirs(base,log,tmp_path/'out')
    assert result['comparabilidade'] == 'aprovada'
    assert result['n_pares_arquitetura_repeticao'] == 2
    assert result['delta_asr_pp_media_por_arquitetura']['A0'] == pytest.approx(3)
    assert result['dif_em_dif_pp_media'] == pytest.approx(3)
    new_cfg['batch_size']=4
    (log/'config_e5a2.json').write_text(json.dumps(new_cfg))
    with pytest.raises(ValueError, match='batch_size'):
        compare_dirs(base,log,tmp_path/'out')
