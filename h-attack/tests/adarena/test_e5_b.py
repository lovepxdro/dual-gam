"""E5-B: seleção test/test, treino resumível e análise pareada."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'docs' / 'scientific' / 'scripts'))

import e5b_campaign as campaign
import e5b_analyze as analysis
from e5_train import Run, split_indices
from gan.models import Atacante, Defensor


def setup_synthetic(tmp_path, monkeypatch):
    rng = np.random.default_rng(2027)
    X = rng.normal(size=(1200, 77))
    y = np.tile([0, 1], 600).astype(int)
    models = {}
    state = Atacante(noise_dim=32, output_dim=77).state_dict()
    for seed in (20, 42, 80):
        tr, val, test = split_indices(y, seed, .4, .1)
        sc = StandardScaler().fit(X[tr])
        d = Defensor(input_dim=77)
        with torch.no_grad():
            d.rede[-2].weight.mul_(.001)
            d.rede[-2].bias.fill_(5.0)
        d.eval().requires_grad_(False)
        models[seed] = Run(seed,tmp_path,
                           {'treinamento':{'lr_atacante':.0002, 'adam_betas_atacante':[.5,.999]}},
                           SimpleNamespace(scaler=sc), tr,val,test,state,d)
    monkeypatch.setattr(campaign,'SEEDS',(20,42,80))
    monkeypatch.setattr(analysis,'SEEDS',(20,42,80))
    monkeypatch.setattr(campaign,'check_preflight',lambda *a,**kw:None)
    monkeypatch.setattr(campaign,'prepare_runs',lambda *a,**kw:(X,y,models,0,'synthetic'))
    p=tmp_path/'preflight.json';p.write_text('{"status":"aprovado"}')
    args=SimpleNamespace(
        runs_root=tmp_path,preflight=p,seeds=[20,42,80],replicas=[1042],
        epochs=1,batch_size=16,max_train_samples=32,val_samples=12,
        test_samples=32,transfer_samples=32,eval_batch=32,device='cpu',
        output_dir=tmp_path/'campanha',stage='train',reuse_previous=False,
        previous_preflight=p,baseline_a2=tmp_path/'old_a2',baseline_a3=tmp_path/'old_a3')
    return args,X,y,models


def test_pares_test_test_nao_contem_treino(tmp_path,monkeypatch):
    args,X,y,runs=setup_synthetic(tmp_path,monkeypatch)
    p1=campaign.pair_test_indices(runs,y,20,42,32)
    p2=campaign.pair_test_indices(runs,y,42,20,32)
    assert np.array_equal(p1,p2)
    assert len(p1)==32
    assert (y[p1]==1).all()
    for s in (20,42):
        assert not np.intersect1d(p1,runs[s].train_ids).size
        assert np.intersect1d(p1,runs[s].test_ids).size==len(p1)


def test_rejeita_seeds_incompletas(tmp_path,monkeypatch):
    args,*_=setup_synthetic(tmp_path,monkeypatch)
    args.seeds=[20,42]
    with pytest.raises(ValueError,match='15 seeds'):
        campaign.validate_args(args)


def test_end_to_end_sintetico_retoma_e_analisa(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    args,_,_,_=setup_synthetic(tmp_path,monkeypatch)
    campaign.execute(args)
    checkpoint = args.output_dir/'seed_20/rep_1042/A1/atacante_final.pth'
    sha = campaign.sha256_file(checkpoint)
    args.stage='evaluate'
    campaign.execute(args)
    summary=campaign.load_json(args.output_dir/'resumo_e5b.json')
    assert summary['n_treinamentos']==9
    assert summary['n_avaliacoes_transferencia']==18
    assert summary['n_pares_nao_direcionados']==3
    res=analysis.analyze(args.output_dir,tmp_path/'analise')
    assert res['validacao_pareamento']=='aprovada'
    assert res['n_treinamentos']==9
    assert res['n_transferencias_direcionadas']==18
    args.stage='all'
    campaign.execute(args)
    assert campaign.sha256_file(checkpoint)==sha


def test_analise_rejeita_ruido_diferente(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    args,_,_,_=setup_synthetic(tmp_path,monkeypatch)
    args.stage='all'
    campaign.execute(args)
    p=args.output_dir/'metricas_alvo.csv'
    text=p.read_text()
    import csv
    with p.open(newline='') as f:
        rows=list(csv.DictReader(f))
    rows[1]['noise_seed']='0'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='pareada'):
        analysis.analyze(args.output_dir,tmp_path/'analise')
