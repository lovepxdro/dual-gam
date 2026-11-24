"""E5-A3: seis treinamentos A2(z), com capacidade próxima a A1(x,z).

Executar na raiz do repositório. Reutiliza protocolo E5-A2 (BCEWithLogits,
D20 congelado, splits E0 e avaliação pareada) sem alterar código E5-A2.
Exige os 12 resultados E5-A2 completos; treina somente o braço A2.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from gan.e5_capacity import AtacanteCapacidade
from gan.models import Atacante
from e5_train import (EPSILON, THRESHOLD, atomic_json, check_preflight,
                      diagnosticar_perdas_iniciais, evaluate, generate_delta,
                      prepare_runs, save_weights, seed_everything, select_ddos,
                      sha256_file, sha256_indices, stable_seed, train_one)


COMPARABLE = (
    'seeds', 'replicas', 'epochs', 'batch_size', 'max_train_samples',
    'val_samples', 'test_samples', 'eval_batch', 'device', 'epsilon',
    'threshold', 'init_from', 'checkpoint_selection', 'preflight_hash',
    'run_root', 'dataset',
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f'Arquivo obrigatório não encontrado: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def checked_baseline(path: Path, expected: dict) -> tuple[dict, dict]:
    cfg = load_json(path / 'config_e5a2.json')
    if cfg.get('experiment') != 'E5-A2' or cfg.get('loss_mode') != 'logits':
        raise ValueError('Baseline inválido: esperado E5-A2 com loss_mode=logits')
    bad = [k for k in COMPARABLE if cfg.get(k) != expected.get(k)]
    if bad:
        raise ValueError(f'E5-A2 e E5-A3 usam configurações diferentes: {bad}')
    summary = load_json(path / 'resumo_e5a.json')
    if summary.get('experiment') != 'E5-A2' or summary.get('loss_mode') != 'logits':
        raise ValueError('Resumo E5-A2 inválido')
    if summary.get('seeds') != expected['seeds'] or summary.get('replicas') != expected['replicas']:
        raise ValueError('Seeds ou réplicas do baseline diferentes')
    rows = summary.get('resultados', [])
    keys = [(r['seed_e0'], r['replica'], r['braco']) for r in rows]
    expected_keys = {(s, rep, arm) for s in expected['seeds'] for rep in expected['replicas'] for arm in ('A0', 'A1')}
    if len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError('Baseline E5-A2 incompleto ou duplicado: são necessários 12 pares A0/A1')
    if any(r['n_teste_comum_ddos'] != summary['n_teste_comum_ddos'] for r in rows):
        raise ValueError('Baseline com contagens de teste inconsistentes')
    return cfg, summary


def expected_config(args) -> dict:
    return {
        'experiment': 'E5-A3', 'seeds': sorted(args.seeds), 'replicas': list(args.replicas),
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'max_train_samples': args.max_train_samples,
        'val_samples': args.val_samples, 'test_samples': args.test_samples,
        'eval_batch': args.eval_batch, 'device': args.device,
        'epsilon': EPSILON, 'threshold': THRESHOLD,
        'init_from': 'A20 E0 (Adam novo)', 'checkpoint_selection': 'epoch final fixo',
        'preflight_hash': sha256_file(args.preflight), 'run_root': str(args.runs_root),
        'dataset': 'DDoS-Friday (origem config E0)',
        'loss_mode': 'logits', 'bracos': ['A2'],
        'base_e5a2': str(args.baseline_dir),
        'base_e5a2_config_sha256': sha256_file(args.baseline_dir / 'config_e5a2.json'),
        'a2_ramo': 'z-only, MLP 153-153, residual zero, backbone inicializado de A20 e treinável; D20 congelado',
        'a2_ignora_x': True,
    }


def build_a2(state: dict, *, input_dim: int, noise_dim: int, init_seed: int):
    seed_everything(init_seed)
    model = AtacanteCapacidade(input_dim=input_dim, noise_dim=noise_dim)
    model.carregar_base_a20(state)
    return model


def execute(args) -> dict:
    if len(set(args.seeds)) != 2 or len(args.seeds) != 2:
        raise ValueError('E5-A3 exige exatamente duas seeds para transferência pareada')
    if len(args.replicas) != len(set(args.replicas)) or not args.replicas:
        raise ValueError('Réplicas devem ser distintas e não vazias')
    if args.device != 'cpu':
        raise ValueError('E5-A3 atual usa CPU para controle de RNG')
    if args.epochs < 1 or args.batch_size < 2 or args.eval_batch < 2 or min(args.val_samples, args.test_samples) < 1 or args.max_train_samples < 0:
        raise ValueError('Orçamento inválido')
    check_preflight(args.preflight, args.runs_root, tuple(sorted(args.seeds)))
    cfg = expected_config(args)
    _, baseline = checked_baseline(args.baseline_dir, cfg)
    summary_ids = baseline['ids_teste_comum_sha256']
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError('Diretório de saída não vazio; use --resume para continuar')
        if load_json(args.output_dir / 'config_e5a3.json') != cfg:
            raise ValueError('Configuração incompatível com --resume')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_dir / 'config_e5a3.json', cfg)

    seeds = tuple(sorted(args.seeds))
    X, y, runs, n_dedup, source = prepare_runs(args.runs_root, args.preflight, seeds)
    common = np.intersect1d(runs[seeds[0]].test_ids, runs[seeds[1]].test_ids, assume_unique=True)
    shared_ddos = select_ddos(common, y, args.test_samples, stable_seed('e5-shared-test', *seeds))
    test_sha = sha256_indices(shared_ddos)
    if test_sha != summary_ids or len(shared_ddos) != baseline['n_teste_comum_ddos']:
        raise AssertionError('Amostras de teste E5-A3 diferem do baseline E5-A2')
    records = []
    for seed in seeds:
        info = runs[seed]
        other = next(s for s in seeds if s != seed)
        target = runs[other]
        x_train = X[info.train_ids]
        y_train = y[info.train_ids]
        train_ddos = x_train[y_train == 1]
        if args.max_train_samples and args.max_train_samples < len(train_ddos):
            indices = np.random.default_rng(stable_seed('e5-train-subset', seed)).choice(
                len(train_ddos), size=args.max_train_samples, replace=False)
            train_ddos = train_ddos[np.sort(indices)]
        x_train_t = torch.from_numpy(info.pre.scaler.transform(train_ddos).astype(np.float32))
        val_ids = select_ddos(info.val_ids, y, args.val_samples, stable_seed('e5-val-selection', seed))
        x_val = info.pre.scaler.transform(X[val_ids]).astype(np.float32)
        raw_shared = np.asarray(X[shared_ddos], dtype=np.float64)
        x_src_shared = info.pre.scaler.transform(raw_shared).astype(np.float32)
        if len(np.intersect1d(val_ids, info.train_ids)) or len(np.intersect1d(shared_ddos, info.train_ids)):
            raise AssertionError('Vazamento de índices em treino/val/teste')
        for rep in args.replicas:
            reference = load_json(args.baseline_dir / f'seed_{seed}' / f'rep_{rep}' / 'A1' / 'resultado.json')
            ref_a0 = load_json(args.baseline_dir / f'seed_{seed}' / f'rep_{rep}' / 'A0' / 'resultado.json')
            for field in ('n_treino_ddos', 'n_validacao_ddos', 'ids_teste_comum_sha256',
                          'noise_seed_teste', 'optimizer', 'epochs', 'batch_size', 'epsilon'):
                if reference.get(field) != ref_a0.get(field):
                    raise AssertionError(f'Baseline A0/A1 inconsistente: {seed}/{rep}, {field}')
            if (reference.get('n_parametros') != 104538 or
                ref_a0.get('n_parametros') != 64013 or
                reference.get('loss_mode') != 'logits' or ref_a0.get('loss_mode') != 'logits'):
                raise AssertionError('Configuração/arquitetura real dos braços A0/A1 divergente')
            if (reference['n_treino_ddos'] != len(x_train_t) or
                reference['n_validacao_ddos'] != len(x_val) or
                reference['ids_teste_comum_sha256'] != test_sha or
                reference['noise_seed_teste'] != stable_seed('e5-test', *seeds, rep) or
                reference['defender_transferencia'] != other or
                reference['optimizer']['lr'] != float(info.config['treinamento']['lr_atacante']) or
                reference['optimizer']['betas'] != list(info.config['treinamento']['adam_betas_atacante']) or
                reference['epochs'] != args.epochs or reference['batch_size'] != args.batch_size or
                reference['epsilon'] != EPSILON):
                raise AssertionError(f'Condições de avaliação diferentes para seed={seed} réplica={rep}')
            out = args.output_dir / f'seed_{seed}' / f'rep_{rep}' / 'A2'
            out.mkdir(parents=True, exist_ok=True)
            result_file = out / 'resultado.json'
            weights_file = out / 'atacante_final.pth'
            if args.resume and result_file.exists() and weights_file.exists():
                prior = load_json(result_file)
                if (prior['seed_e0'], prior['replica'], prior['braco']) != (seed, rep, 'A2'):
                    raise ValueError('Resultado para --resume com chave inválida')
                if prior.get('sha256_checkpoint') != sha256_file(weights_file):
                    raise ValueError('Checkpoint de --resume alterado')
                records.append(prior)
                print(f'[E5-A3] Reaproveitado seed={seed}, réplica={rep}', flush=True)
                continue
            model = build_a2(info.checkpoint_attacker, input_dim=X.shape[1], noise_dim=32,
                             init_seed=stable_seed('e5-init', seed, rep))
            model.eval()
            with torch.no_grad():
                g = torch.Generator(device='cpu').manual_seed(stable_seed('e5-id', seed, rep))
                z = torch.randn(32, 32, generator=g)
                x = x_train_t[:32]
                if len(x) != 32:
                    raise ValueError('Treino exige pelo menos 32 exemplos DDoS')
                a0 = Atacante(noise_dim=32, output_dim=X.shape[1])
                a0.load_state_dict(info.checkpoint_attacker, strict=True)
                a0.eval()
                initial_diff = float((model(x, z) - a0(z)).abs().max())
                x_synthetic = torch.full_like(x, 123.0)
                x_invariance = float((model(x, z) - model(x_synthetic, z)).abs().max())
                if initial_diff > 1e-7 or x_invariance != 0.0:
                    raise AssertionError('Controle A2 não reproduz A20 ou depende de x')
            diagnostic = diagnosticar_perdas_iniciais(
                model, 'A2', info.defender, x_train_t,
                seed_e0=seed, rep_seed=rep, noise_dim=32, epsilon=EPSILON)
            curves, elapsed = train_one(
                model, 'A2', info.defender, x_train_t, x_val,
                seed_e0=seed, rep_seed=rep, epochs=args.epochs,
                batch_size=args.batch_size, noise_dim=32, epsilon=EPSILON,
                lr=float(info.config['treinamento']['lr_atacante']),
                betas=tuple(float(v) for v in info.config['treinamento']['adam_betas_atacante']),
                eval_batch=args.eval_batch, loss_mode='logits')
            # Avaliação somente após todos os epochs, sem escolha de checkpoint por teste.
            noise_seed = stable_seed('e5-test', *seeds, rep)
            metric_target = evaluate(model, 'A2', info.defender, x_src_shared, noise_dim=32,
                                     noise_seed=noise_seed, epsilon=EPSILON, batch_size=args.eval_batch)
            metric_transfer = evaluate(model, 'A2', target.defender, x_src_shared, noise_dim=32,
                                       noise_seed=noise_seed, epsilon=EPSILON, batch_size=args.eval_batch,
                                       source_scaler=info.pre.scaler, target_scaler=target.pre.scaler,
                                       x_raw=raw_shared)
            row = {
                'experiment': 'E5-A3', 'seed_e0': seed, 'replica': rep, 'braco': 'A2',
                'defender_alvo': seed, 'defender_transferencia': other,
                'n_treino_ddos': len(x_train_t), 'n_validacao_ddos': len(x_val),
                'n_teste_comum_ddos': len(shared_ddos), 'ids_teste_comum_sha256': test_sha,
                'noise_seed_teste': noise_seed,
                'init_a0_a2_max_abs_diff': initial_diff,
                'invariancia_x_max_abs_diff': x_invariance,
                'n_parametros': sum(p.numel() for p in model.parameters()),
                'optimizer': {'type': 'Adam novo',
                              'lr': float(info.config['treinamento']['lr_atacante']),
                              'betas': list(info.config['treinamento']['adam_betas_atacante'])},
                'epochs': args.epochs, 'batch_size': args.batch_size,
                'epsilon': EPSILON, 'loss_mode': 'logits',
                'tempo_treinamento_segundos': elapsed,
                'curva_validacao': curves,
                'diagnostico_pre_treino': diagnostic,
                'eficacia_alvo_teste': metric_target,
                'transferencia_teste': metric_transfer,
            }
            save_weights(weights_file, model.state_dict())
            row['sha256_checkpoint'] = sha256_file(weights_file)
            atomic_json(result_file, row)
            print(f'[E5-A3] seed={seed} réplica={rep} ASR alvo={metric_target["asr_condicional"]} '
                  f'ASR transfer={metric_transfer["asr_condicional"]}', flush=True)
            records.append(row)

    flat = []
    for rec in records:
        flat.append({
            'seed_e0': rec['seed_e0'], 'replica': rec['replica'], 'braco': 'A2',
            'seed_defender_transferencia': rec['defender_transferencia'],
            'n_treino_ddos': rec['n_treino_ddos'],
            'n_teste_comum_ddos': rec['n_teste_comum_ddos'],
            'asr_alvo': rec['eficacia_alvo_teste']['asr_condicional'],
            'asr_transferencia': rec['transferencia_teste']['asr_condicional'],
            'diversidade_l2_alvo': rec['eficacia_alvo_teste']['diversidade_l2_mesma_amostra'],
            'diversidade_l2_eficaz_alvo': rec['eficacia_alvo_teste']['diversidade_l2_pares_eficazes'],
            'n_pares_eficazes': rec['eficacia_alvo_teste']['pares_ambos_sucesso'],
            'tempo_treino_segundos': rec['tempo_treinamento_segundos'],
        })
    flat.sort(key=lambda row: (row['seed_e0'], row['replica']))
    with (args.output_dir / 'resultados_e5a3.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    summary = {
        'experiment': 'E5-A3', 'status': 'treinamentos_concluidos',
        'dataset': source, 'seeds': list(seeds), 'replicas': list(args.replicas),
        'n_treinamentos': len(records), 'n_teste_comum_ddos': len(shared_ddos),
        'ids_teste_comum_sha256': test_sha, 'deduplicatas_originais': n_dedup,
        'parametros_a2': records[0]['n_parametros'],
        'limites': 'N=2 seeds, três réplicas; x não entra no Attacker A2; estruturas residuais diferem em largura',
        'resultados': flat,
    }
    atomic_json(args.output_dir / 'resumo_e5a3.json', summary)
    print(f'[E5-A3] Salvos {len(records)} treinamentos em {args.output_dir}', flush=True)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, default=Path('models/e0/experiments'))
    p.add_argument('--preflight', type=Path, default=Path('models/e5/preflight/e5_preflight.json'))
    p.add_argument('--baseline-dir', type=Path, default=Path('models/e5/a2_logits'))
    p.add_argument('--seeds', nargs='+', type=int, default=[20, 42])
    p.add_argument('--replicas', nargs='+', type=int, default=[1042, 1043, 1044])
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--max-train-samples', type=int, default=0)
    p.add_argument('--val-samples', type=int, default=4096)
    p.add_argument('--test-samples', type=int, default=5000)
    p.add_argument('--eval-batch', type=int, default=1024)
    p.add_argument('--device', choices=['cpu'], default='cpu')
    p.add_argument('--output-dir', type=Path, default=Path('models/e5/a3_capacidade'))
    p.add_argument('--resume', action='store_true')
    return p.parse_args(argv)


if __name__ == '__main__':
    execute(parse_args())
