"""E5-B: 15 seeds x 3 repetições x A0/A1/A2; treino pareado e transferência test/test.

Exige E5-P ampliado para as 15 seeds. Executar da raiz do repositório.
Não modifica checkpoints E0. A fase train só usa treino/validação; a fase
 evaluate só lê checkpoints finais e avalia testes uma vez, sem tuning.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from gan.e5_capacity import AtacanteCapacidade
from gan.models import Atacante
from e5_train import (
    EPSILON, THRESHOLD, atomic_json, check_preflight,
    diagnosticar_perdas_iniciais, ensure_finite, evaluate, generate_delta,
    init_attacker, prepare_runs, save_weights, select_ddos, sha256_file,
    sha256_indices, stable_seed, train_one,
)

ARMS = ('A0', 'A1', 'A2')
PARAMS = {'A0': 64013, 'A1': 104538, 'A2': 104482}
SEEDS = (1,2,3,4,5,6,7,8,9,10,20,40,42,80,100)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'Não encontrado: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def expected_config(args) -> dict:
    return {
        'experiment': 'E5-B', 'seeds': sorted(args.seeds),
        'replicas': list(args.replicas), 'arms': list(ARMS),
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'max_train_samples': args.max_train_samples,
        'val_samples': args.val_samples,
        'test_samples': args.test_samples,
        'transfer_samples': args.transfer_samples,
        'eval_batch': args.eval_batch, 'device': args.device,
        'loss_mode': 'logits', 'epsilon': EPSILON, 'threshold': THRESHOLD,
        'init_from': 'A20 E0 (Adam novo)', 'checkpoint_selection': 'epoch final fixo',
        'preflight_hash': sha256_file(args.preflight),
        'runs_root': str(args.runs_root),
        'reuse_previous': bool(args.reuse_previous),
        'previous_preflight': str(args.previous_preflight) if args.reuse_previous else None,
        'baseline_a2': str(args.baseline_a2) if args.reuse_previous else None,
        'baseline_a3': str(args.baseline_a3) if args.reuse_previous else None,
        'training_protocol': 'mesma ordem de batches, z, A20, Adam novo; D20 congelado',
        'evaluation_protocol': 'teste da seed-alvo; transferencia no teste/teste por par; sem reuso das metricas antigas',
    }


def validate_args(args):
    if tuple(sorted(args.seeds)) != SEEDS:
        raise ValueError(f'E5-B exige as 15 seeds E0, sem duplicatas: {SEEDS}')
    if not args.replicas or len(args.replicas) != len(set(args.replicas)):
        raise ValueError('Réplicas vazias ou duplicadas')
    if args.device != 'cpu' or args.epochs < 1 or args.batch_size < 2 or args.eval_batch < 2:
        raise ValueError('E5-B exige CPU, epochs >=1 e batches >=2')
    if min(args.val_samples, args.test_samples, args.transfer_samples) < 1 or args.max_train_samples < 0:
        raise ValueError('Número de amostras inválido')
    if args.reuse_previous and tuple(args.replicas) != (1042,1043,1044):
        raise ValueError('Reutilização E5-A2/A3 exige réplicas 1042,1043,1044')


def ensure_config(args) -> dict:
    config = expected_config(args)
    target = args.output_dir / 'config_e5b.json'
    if target.exists():
        if load_json(target) != config:
            raise ValueError('Configuração no diretório de saída divergiu; não sobrescrever campanha')
    else:
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError('Pasta não vazia sem config_e5b.json')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(target, config)
    return config


def check_old_preflight(args):
    old = load_json(args.previous_preflight)
    new = load_json(args.preflight)
    old_entries = {int(r['seed']): r for r in old['seeds']}
    new_entries = {int(r['seed']): r for r in new['seeds']}
    for seed in (20, 42):
        if seed not in old_entries or seed not in new_entries:
            raise ValueError('Preflight anterior ou novo não contempla seed 20/42')
        for k in ('sha256_a20', 'sha256_d20', 'sha256_scaler', 'run_id'):
            if old_entries[seed][k] != new_entries[seed][k]:
                raise ValueError(f'Checkpoint/preprocessador da seed {seed} mudou: {k}')


def check_legacy_sources(args):
    """Valida proveniência/treino; NÃO reutiliza métricas do teste antigo."""
    check_old_preflight(args)
    c2 = load_json(args.baseline_a2 / 'config_e5a2.json')
    c3 = load_json(args.baseline_a3 / 'config_e5a3.json')
    if c2.get('experiment') != 'E5-A2' or c2.get('loss_mode') != 'logits':
        raise ValueError('Baseline A0/A1 não é E5-A2/logits')
    if c3.get('experiment') != 'E5-A3' or c3.get('loss_mode') != 'logits' or c3.get('a2_ignora_x') is not True:
        raise ValueError('Baseline A2 não é E5-A3/logits válido')
    if c3.get('base_e5a2_config_sha256') != sha256_file(args.baseline_a2 / 'config_e5a2.json'):
        raise ValueError('Config E5-A2 difere da dependência E5-A3')
    if c2.get('preflight_hash') != sha256_file(args.previous_preflight) or c3.get('preflight_hash') != sha256_file(args.previous_preflight):
        raise ValueError('Preflight anterior não é aquele usado nos treinamentos E5-A2/A3')
    for cfg in (c2, c3):
        if cfg.get('seeds') != [20,42] or cfg.get('replicas') != list(args.replicas):
            raise ValueError('Seeds/réplicas dos resultados anteriores diferentes')
        for key in ('epochs','batch_size','max_train_samples','val_samples','eval_batch','device','epsilon','threshold','init_from','checkpoint_selection'):
            value = (EPSILON if key == 'epsilon' else THRESHOLD if key == 'threshold' else
                     'A20 E0 (Adam novo)' if key == 'init_from' else
                     'epoch final fixo' if key == 'checkpoint_selection' else
                     getattr(args, key))
            if cfg.get(key) != value:
                raise ValueError(f'Configuração antiga difere em {key}; não reutilizar pesos')
        if cfg.get('run_root') != str(args.runs_root):
            raise ValueError('runs_root difere da configuração original')


def old_checkpoint(args, seed: int, rep: int, arm: str, *, n_train: int,
                   n_val: int, lr: float, betas: list[float]) -> tuple[Path, dict]:
    root = args.baseline_a3 if arm == 'A2' else args.baseline_a2
    path = root / f'seed_{seed}' / f'rep_{rep}' / arm
    record = load_json(path / 'resultado.json')
    checkpoint = path / 'atacante_final.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Checkpoint E5-A2/A3 ausente: {checkpoint}')
    conditions = {
        'seed_e0': seed, 'replica': rep, 'braco': arm,
        'n_treino_ddos': n_train, 'n_validacao_ddos': n_val,
        'n_parametros': PARAMS[arm], 'epochs': args.epochs,
        'batch_size': args.batch_size, 'epsilon': EPSILON, 'loss_mode': 'logits',
    }
    if any(record.get(k) != v for k, v in conditions.items()):
        raise ValueError(f'Dados de treino/arquitetura de {seed}/{rep}/{arm} incompatíveis')
    opt = record.get('optimizer', {})
    if opt.get('type') != 'Adam novo' or opt.get('lr') != lr or opt.get('betas') != betas:
        raise ValueError(f'Otimizador de {seed}/{rep}/{arm} incompatível')
    if arm == 'A2' and record.get('sha256_checkpoint') != sha256_file(checkpoint):
        raise ValueError(f'Hash de A2 inválido: {checkpoint}')
    return checkpoint, record



def prevalidate_old_checkpoints(args, X, y, runs):
    """Falhar ANTES dos 117 novos treinamentos se algum legado estiver ausente."""
    for seed in (20, 42):
        info = runs[seed]
        ntrain = len(select_train(info, X, y, seed=seed, cap=args.max_train_samples))
        nval = len(select_ddos(info.val_ids, y, args.val_samples,
                               stable_seed('e5-val-selection', seed)))
        tr = info.config['treinamento']
        for rep in args.replicas:
            for arm in ARMS:
                checkpoint, _ = old_checkpoint(
                    args, seed, rep, arm, n_train=ntrain, n_val=nval,
                    lr=float(tr['lr_atacante']),
                    betas=[float(v) for v in tr['adam_betas_atacante']])
                model = create_attacker(arm, info.checkpoint_attacker,
                                        input_dim=X.shape[1], noise_dim=32,
                                        seed=seed, rep=rep)
                model.load_state_dict(torch.load(checkpoint, map_location='cpu',
                                                 weights_only=True), strict=True)
    print('[E5-B] Os 18 checkpoints E5-A2/A3 foram verificados antes do treinamento', flush=True)


def create_attacker(arm: str, state: dict, *, input_dim: int, noise_dim: int, seed: int, rep: int):
    if arm in ('A0', 'A1'):
        return init_attacker(arm, state, input_dim=input_dim, noise_dim=noise_dim,
                             init_seed=stable_seed('e5-init', seed, rep))
    from e5a3_capacity import build_a2
    return build_a2(state, input_dim=input_dim, noise_dim=noise_dim,
                    init_seed=stable_seed('e5-init', seed, rep))


def check_initial_identity(attacker, arm, state, x):
    reference = Atacante(noise_dim=32, output_dim=77)
    reference.load_state_dict(state, strict=True)
    reference.eval()
    attacker.eval()
    with torch.no_grad():
        z = torch.randn(32, 32, generator=torch.Generator().manual_seed(2026))
        original = reference(z)
        actual = generate_delta(attacker, arm, x[:32], z)
        delta = float((original - actual).abs().max())
        if delta > 1e-7:
            raise AssertionError(f'Identidade A20/A{arm} falhou: {delta}')
    return delta


def select_train(info, X, y, *, seed: int, cap: int):
    subset = X[info.train_ids]
    labels = y[info.train_ids]
    attack = subset[labels == 1]
    if 0 < cap < len(attack):
        chosen = np.random.default_rng(stable_seed('e5-train-subset', seed)).choice(len(attack), size=cap, replace=False)
        attack = attack[np.sort(chosen)]
    return torch.from_numpy(info.pre.scaler.transform(attack).astype(np.float32))


def save_completed(path: Path, value: dict):
    atomic_json(path, value)


def train_campaign(args, X, y, runs):
    total_new = total_reused = 0
    for seed in sorted(runs):
        info = runs[seed]
        x_train = select_train(info, X, y, seed=seed, cap=args.max_train_samples)
        if len(x_train) < 32:
            raise ValueError('Treino exige 32 ou mais amostras para a identidade inicial')
        val_ids = select_ddos(info.val_ids, y, args.val_samples, stable_seed('e5-val-selection', seed))
        x_val = info.pre.scaler.transform(X[val_ids]).astype(np.float32)
        tcfg = info.config['treinamento']
        lr = float(tcfg['lr_atacante'])
        betas = [float(v) for v in tcfg['adam_betas_atacante']]
        for rep in args.replicas:
            for arm in ARMS:
                dest = args.output_dir / f'seed_{seed}' / f'rep_{rep}' / arm
                dest.mkdir(parents=True, exist_ok=True)
                result_file = dest / 'treinamento.json'
                weights_file = dest / 'atacante_final.pth'
                if result_file.exists() or weights_file.exists():
                    if not (result_file.is_file() and weights_file.is_file()):
                        raise RuntimeError(f'Treino parcial em {dest}; corrigir antes de retomar')
                    prior = load_json(result_file)
                    if (prior['seed_e0'],prior['replica'],prior['braco']) != (seed,rep,arm) or prior['sha256_checkpoint'] != sha256_file(weights_file):
                        raise RuntimeError(f'Treino antigo ou checkpoint alterado em {dest}')
                    if prior['n_treino_ddos'] != len(x_train) or prior['n_validacao_ddos'] != len(x_val):
                        raise RuntimeError('Treino anterior divergente nas amostras')
                    print(f'[E5-B/train] Resume seed={seed} rep={rep} {arm}', flush=True)
                    continue
                if args.reuse_previous and seed in (20,42):
                    src, old_result = old_checkpoint(args, seed, rep, arm, n_train=len(x_train),
                                                       n_val=len(x_val), lr=lr, betas=betas)
                    # Copy byte-for-byte; never re-save torch model, preserving original SHA.
                    tmp = weights_file.with_suffix('.pth.tmp')
                    shutil.copyfile(src, tmp)
                    tmp.replace(weights_file)
                    result = {
                        'seed_e0': seed, 'replica': rep, 'braco': arm, 'fonte': 'reutilizado',
                        'origem': str(src), 'sha256_checkpoint': sha256_file(weights_file),
                        'n_treino_ddos': len(x_train), 'n_validacao_ddos': len(x_val),
                        'epochs': args.epochs, 'batch_size': args.batch_size, 'loss_mode': 'logits',
                        'optimizer': {'type': 'Adam novo', 'lr': lr, 'betas': betas},
                        'curva_validacao': old_result['curva_validacao'],
                        'tempo_treinamento_segundos': old_result['tempo_treinamento_segundos'],
                        'n_parametros': PARAMS[arm],
                    }
                    save_completed(result_file, result)
                    total_reused += 1
                    print(f'[E5-B/train] Reutilizado {seed}/{rep}/{arm}', flush=True)
                    continue
                model = create_attacker(arm, info.checkpoint_attacker, input_dim=X.shape[1],
                                        noise_dim=32, seed=seed, rep=rep)
                identity = check_initial_identity(model, arm, info.checkpoint_attacker, x_train)
                diagnostic = diagnosticar_perdas_iniciais(
                    model, arm, info.defender, x_train, seed_e0=seed, rep_seed=rep,
                    noise_dim=32, epsilon=EPSILON)
                curves, elapsed = train_one(
                    model, arm, info.defender, x_train, x_val,
                    seed_e0=seed, rep_seed=rep, epochs=args.epochs,
                    batch_size=args.batch_size, noise_dim=32, epsilon=EPSILON,
                    lr=lr, betas=tuple(betas), eval_batch=args.eval_batch, loss_mode='logits')
                save_weights(weights_file, model.state_dict())
                record = {
                    'seed_e0': seed, 'replica': rep, 'braco': arm, 'fonte': 'novo',
                    'sha256_checkpoint': sha256_file(weights_file),
                    'n_treino_ddos': len(x_train), 'n_validacao_ddos': len(x_val),
                    'epochs': args.epochs, 'batch_size': args.batch_size,
                    'loss_mode': 'logits', 'optimizer': {'type': 'Adam novo', 'lr': lr, 'betas': betas},
                    'curva_validacao': curves, 'tempo_treinamento_segundos': elapsed,
                    'n_parametros': sum(p.numel() for p in model.parameters()),
                    'identidade_inicial_max_diff': identity,
                    'diagnostico_pre_treino': diagnostic,
                }
                save_completed(result_file, record)
                total_new += 1
                print(f'[E5-B/train] Salvo seed={seed} rep={rep} {arm}', flush=True)
    print(f'[E5-B/train] Pronto: {total_new} novos, {total_reused} reutilizados, {len(runs)*len(args.replicas)*len(ARMS)} disponíveis.', flush=True)


def verify_trained(args, runs):
    for seed in runs:
        for rep in args.replicas:
            for arm in ARMS:
                root = args.output_dir / f'seed_{seed}' / f'rep_{rep}' / arm
                r = load_json(root / 'treinamento.json')
                w = root / 'atacante_final.pth'
                if r['sha256_checkpoint'] != sha256_file(w) or (r['seed_e0'],r['replica'],r['braco']) != (seed,rep,arm):
                    raise RuntimeError(f'Checkpoint/cadastro inválido: {root}')
    print('[E5-B/eval] Todos os 135 checkpoints encontrados e verificados', flush=True)


def pair_test_indices(runs: dict, y: np.ndarray, seed: int, other: int, cap: int) -> np.ndarray:
    """Interseção TESTE/TESTE: evita dados de treino de origem e destino."""
    a, b = sorted((seed, other))
    common = np.intersect1d(runs[a].test_ids, runs[b].test_ids, assume_unique=True)
    chosen = select_ddos(common, y, cap, stable_seed('e5b-pair', a, b))
    if len(chosen) < 32:
        raise ValueError(f'Par {a}/{b} com menos de 32 DDoS em test/test')
    return chosen


def evaluation_metadata(indices: np.ndarray, noise: int):
    return {'n_amostras': len(indices), 'ids_sha256': sha256_indices(indices), 'noise_seed': noise}


def evaluate_campaign(args, X, y, runs):
    verify_trained(args, runs)
    seeds = tuple(sorted(runs))
    target_ids = {s: select_ddos(runs[s].test_ids, y, args.test_samples,
                                 stable_seed('e5b-target-test', s)) for s in seeds}
    pairs = {(a,b): pair_test_indices(runs, y, a, b, args.transfer_samples)
             for i,a in enumerate(seeds) for b in seeds[i+1:]}
    # For each source, choose target test and pairwise transfer test; same indices
    # and z for all 3 arms and 3 replications; pair test never leaks to training.
    for seed in seeds:
        info = runs[seed]
        idx_target = target_ids[seed]
        if np.intersect1d(idx_target, info.train_ids).size:
            raise AssertionError(f'Teste local vazou para treino da seed {seed}')
        x_target = info.pre.scaler.transform(X[idx_target]).astype(np.float32)
        other_inputs = {}
        for target in seeds:
            if target == seed:
                continue
            pair = tuple(sorted((seed,target)))
            ids = pairs[pair]
            if (np.intersect1d(ids, info.train_ids).size or
                np.intersect1d(ids, runs[target].train_ids).size):
                raise AssertionError(f'Vazamento entre {seed} e {target}')
            raw = np.asarray(X[ids], dtype=np.float64)
            x_src = info.pre.scaler.transform(raw).astype(np.float32)
            ensure_finite(x_src, 'Fonte transfer')
            other_inputs[target] = (ids, raw, x_src)
        for rep in args.replicas:
            noise_target = stable_seed('e5b-target-z', seed, rep)
            for arm in ARMS:
                dest = args.output_dir / f'seed_{seed}' / f'rep_{rep}' / arm
                checkpoint_sha = load_json(dest / 'treinamento.json')['sha256_checkpoint']
                model = create_attacker(arm, info.checkpoint_attacker, input_dim=X.shape[1],
                                        noise_dim=32, seed=seed, rep=rep)
                state = torch.load(dest / 'atacante_final.pth', map_location='cpu', weights_only=True)
                model.load_state_dict(state, strict=True)
                model.eval()
                if sum(p.numel() for p in model.parameters()) != PARAMS[arm]:
                    raise AssertionError(f'Parâmetros inesperados {seed}/{rep}/{arm}')
                local_file = dest / 'avaliacao_alvo.json'
                local_meta = {**evaluation_metadata(idx_target, noise_target),
                              'sha256_checkpoint': checkpoint_sha}
                if local_file.is_file():
                    row = load_json(local_file)
                    if (row['meta'] != local_meta or
                        (row['seed_origem'],row['replica'],row['braco']) != (seed,rep,arm)):
                        raise ValueError(f'Avaliação local incomparável em {local_file}')
                else:
                    metric = evaluate(model, arm, info.defender, x_target, noise_dim=32,
                                      noise_seed=noise_target, epsilon=EPSILON,
                                      batch_size=args.eval_batch)
                    atomic_json(local_file, {'seed_origem': seed, 'replica': rep,
                                            'braco': arm, 'meta': local_meta,
                                            'metricas': metric})
                for target in seeds:
                    if target == seed:
                        continue
                    ids, raw, x_src = other_inputs[target]
                    pair = tuple(sorted((seed,target)))
                    noise = stable_seed('e5b-pair-z', *pair, rep)
                    meta = {**evaluation_metadata(ids, noise),
                            'sha256_checkpoint': checkpoint_sha}
                    path = dest / f'transferencia_para_{target}.json'
                    if path.is_file():
                        row = load_json(path)
                        if (row['meta'] != meta or row['seed_destino'] != target or
                            (row['seed_origem'],row['replica'],row['braco']) != (seed,rep,arm)):
                            raise ValueError(f'Transferência incomparável em {path}')
                        continue
                    result = evaluate(model, arm, runs[target].defender, x_src,
                                      noise_dim=32, noise_seed=noise, epsilon=EPSILON,
                                      batch_size=args.eval_batch,
                                      source_scaler=info.pre.scaler,
                                      target_scaler=runs[target].pre.scaler,
                                      x_raw=raw)
                    atomic_json(path, {'seed_origem': seed, 'seed_destino': target,
                                       'replica': rep, 'braco': arm, 'meta': meta,
                                       'metricas': result})
                print(f'[E5-B/eval] seed={seed} rep={rep} {arm}: alvo + 14 transferências OK', flush=True)
    summarize_evaluation(args)


def summarize_evaluation(args):
    rows_target, rows_transfer = [], []
    for seed in sorted(args.seeds):
        for rep in args.replicas:
            for arm in ARMS:
                dest = args.output_dir / f'seed_{seed}' / f'rep_{rep}' / arm
                tr = load_json(dest / 'treinamento.json')
                tar = load_json(dest / 'avaliacao_alvo.json')
                rows_target.append({
                    'seed': seed, 'replica': rep, 'braco': arm,
                    'n': tar['meta']['n_amostras'], 'ids_sha256': tar['meta']['ids_sha256'],
                    'noise_seed': tar['meta']['noise_seed'],
                    'detectados_limpos': tar['metricas']['n_detectados_limpos'],
                    'sucessos': tar['metricas']['n_sucesso'],
                    'asr': tar['metricas']['asr_condicional'],
                    'diversidade_l2': tar['metricas']['diversidade_l2_mesma_amostra'],
                    'diversidade_l2_eficaz': tar['metricas']['diversidade_l2_pares_eficazes'],
                    'tempo_treino_s': tr['tempo_treinamento_segundos'], 'fonte': tr['fonte'],
                })
                for other in sorted(args.seeds):
                    if other == seed:
                        continue
                    res = load_json(dest / f'transferencia_para_{other}.json')
                    rows_transfer.append({
                        'origem': seed, 'destino': other, 'replica': rep, 'braco': arm,
                        'n': res['meta']['n_amostras'], 'ids_sha256': res['meta']['ids_sha256'],
                        'noise_seed': res['meta']['noise_seed'],
                        'detectados_limpos': res['metricas']['n_detectados_limpos'],
                        'sucessos': res['metricas']['n_sucesso'],
                        'asr': res['metricas']['asr_condicional'],
                        'diversidade_l2': res['metricas']['diversidade_l2_mesma_amostra'],
                    })
    for filename, rows in [('metricas_alvo.csv',rows_target),('metricas_transferencia.csv',rows_transfer)]:
        with (args.output_dir / filename).open('w',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary = {
        'experiment':'E5-B', 'status':'avaliacao_concluida',
        'n_seeds':len(args.seeds), 'n_replicas':len(args.replicas),
        'n_treinamentos':len(rows_target), 'n_avaliacoes_transferencia':len(rows_transfer),
        'n_pares_nao_direcionados':len(args.seeds)*(len(args.seeds)-1)//2,
        'n_treinamentos_reutilizados':sum(r['fonte']=='reutilizado' for r in rows_target),
        'nota': 'ASR condicional ao D20 detectar fluxo limpo; teste/teste por par; estatística por seed, não por pares independentes',
    }
    atomic_json(args.output_dir / 'resumo_e5b.json',summary)
    print('[E5-B/eval] Campanha completa. metricas_alvo.csv + metricas_transferencia.csv',flush=True)
    return summary


def execute(args):
    validate_args(args)
    ensure_config(args)
    check_preflight(args.preflight, args.runs_root, tuple(sorted(args.seeds)))
    if args.reuse_previous:
        check_legacy_sources(args)
    X,y,runs,_,_ = prepare_runs(args.runs_root,args.preflight,tuple(sorted(args.seeds)))
    if args.reuse_previous:
        prevalidate_old_checkpoints(args,X,y,runs)
    if args.stage in ('train','all'):
        train_campaign(args,X,y,runs)
    if args.stage in ('evaluate','all'):
        evaluate_campaign(args,X,y,runs)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',type=Path,default=Path('models/e0/experiments'))
    p.add_argument('--preflight',type=Path,default=Path('models/e5/preflight_15/e5_preflight.json'))
    p.add_argument('--seeds',type=int,nargs='+',default=list(SEEDS))
    p.add_argument('--replicas',type=int,nargs='+',default=[1042,1043,1044])
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--max-train-samples',type=int,default=0)
    p.add_argument('--val-samples',type=int,default=4096)
    p.add_argument('--test-samples',type=int,default=5000)
    p.add_argument('--transfer-samples',type=int,default=1024)
    p.add_argument('--eval-batch',type=int,default=1024)
    p.add_argument('--device',choices=['cpu'],default='cpu')
    p.add_argument('--output-dir',type=Path,default=Path('models/e5/b_campanha'))
    p.add_argument('--stage',choices=['train','evaluate','all'],default='all')
    p.add_argument('--reuse-previous',action='store_true',help='Copiar checkpoints E5-A2/A3 seeds 20/42 após verificar todos os controles')
    p.add_argument('--previous-preflight',type=Path,default=Path('models/e5/preflight/e5_preflight.json'))
    p.add_argument('--baseline-a2',type=Path,default=Path('models/e5/a2_logits'))
    p.add_argument('--baseline-a3',type=Path,default=Path('models/e5/a3_capacidade'))
    return p.parse_args(argv)


if __name__=='__main__':
    execute(parse_args())
