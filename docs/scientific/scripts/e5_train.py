"""E5-A: comparação pareada de Attackers A0(z) e A1(x,z) contra D20 congelado.

Executar na raiz do dual-gam. Não modifica modelos ou preprocessadores do E0.
Preflight E5-P deve ter sido aprovado. Dados e checkpoints do E0 são reutilizados;
nenhuma seleção de hiperparâmetros usa teste. Teste é avaliado somente no epoch final.

Saídas por braço são persistidas após concluir o treinamento, permitindo --resume.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from gan.e5_conditional import AtacanteCondicional
from gan.models import Atacante, Defensor
from gan.preprocessing import Preprocessador


THRESHOLD = 0.5
EPSILON = 0.3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_indices(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype='<i8').tobytes()).hexdigest()


def stable_seed(*parts: object) -> int:
    data = '|'.join(str(v) for v in parts).encode('utf-8')
    return int.from_bytes(hashlib.sha256(data).digest()[:8], 'little') % (2**63 - 1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def ensure_finite(matrix: np.ndarray, label: str) -> None:
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f'{label}: matriz não finita ou com shape inválido')


def deduplicate_indices(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exatamente o critério do E0: features + label e keep=first."""
    df = pd.DataFrame(x)
    df['_label'] = y
    return np.flatnonzero(~df.duplicated(keep='first').to_numpy())


def split_indices(y: np.ndarray, seed: int, test_size: float, validation_size: float):
    """Mesmo par de train_test_split estratificados usado pelo E0."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError('Split inválido')
    ids = np.arange(len(y), dtype=np.int64)
    temporary, test = train_test_split(ids, test_size=test_size, random_state=seed, stratify=y)
    train, val = train_test_split(temporary, test_size=validation_size / (1 - test_size),
                                  random_state=seed, stratify=y[temporary])
    if len(np.intersect1d(train, val)) or len(np.intersect1d(train, test)) or len(np.intersect1d(val, test)):
        raise AssertionError('Vazamento de índices entre splits')
    return train, val, test


def select_ddos(indices: np.ndarray, y: np.ndarray, cap: int, seed: int) -> np.ndarray:
    candidates = np.asarray(indices, dtype=np.int64)
    candidates = candidates[y[candidates] == 1]
    if len(candidates) == 0:
        raise ValueError('Seleção de avaliação sem DDoS')
    if cap > 0 and cap < len(candidates):
        candidates = np.random.default_rng(seed).choice(candidates, size=cap, replace=False)
    return np.sort(candidates)


def frozen_defender(state: dict, input_dim: int) -> Defensor:
    model = Defensor(input_dim=input_dim)
    model.load_state_dict(state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model


def init_attacker(arm: str, state: dict, *, input_dim: int, noise_dim: int, init_seed: int):
    seed_everything(init_seed)
    if arm == 'A0':
        model = Atacante(noise_dim=noise_dim, output_dim=input_dim)
        model.load_state_dict(state, strict=True)
    elif arm == 'A1':
        model = AtacanteCondicional(noise_dim=noise_dim, input_dim=input_dim)
        model.carregar_base_a20(state)
    else:
        raise ValueError(f'Braço desconhecido: {arm}')
    return model


def generate_delta(model: nn.Module, arm: str, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return model(z) if arm == 'A0' else model(x, z)


def logits_defender(defender: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Logits reais de D20 antes da última Sigmoid (sem logit(prob) e sem clamp).

    A arquitetura do E0 termina em nn.Sigmoid. Avaliar o prefixo de `rede`
    mantém os pesos congelados, mas evita a saturação numérica da Sigmoid.
    Abortamos se houver uma arquitetura inesperada em vez de fabricar logits.
    """
    rede = getattr(defender, 'rede', None)
    if not isinstance(rede, nn.Sequential) or len(rede) < 2 or not isinstance(rede[-1], nn.Sigmoid):
        raise TypeError('E5-A2 exige Defender com rede Sequential terminando em Sigmoid')
    logits = rede[:-1](x)
    if logits.ndim != 2 or logits.shape[1] != 1:
        raise AssertionError(f'Saída pré-sigmoid esperada [N,1], recebida {tuple(logits.shape)}')
    return logits.squeeze(1)


def perda_e5(defender: nn.Module, adv: torch.Tensor, mode: str):
    """BCE histórica e BCE numérica estável: mesma tarefa/labels-alvo benignos."""
    if mode == 'probs':
        pred = defender(adv)
        return nn.functional.binary_cross_entropy(pred, torch.zeros_like(pred)), pred.detach(), None
    if mode == 'logits':
        logits = logits_defender(defender, adv)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits))
        return loss, torch.sigmoid(logits.detach()), logits.detach()
    raise ValueError(f'Modo de perda desconhecido: {mode}')


def norma_gradiente(model: nn.Module) -> tuple[float, float | None]:
    """Norma L2 global e do ramo condicional, antes do passo do otimizador."""
    total_sq = branch_sq = 0.0
    has_branch = False
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            raise FloatingPointError(f'Gradiente não finito: {name}')
        squared = float(grad.double().square().sum())
        total_sq += squared
        if name.startswith('ramo.'):
            has_branch = True
            branch_sq += squared
    return math.sqrt(total_sq), math.sqrt(branch_sq) if has_branch else None


def diagnosticar_perdas_iniciais(model: nn.Module, arm: str, defender: nn.Module,
                                 x_ddos: torch.Tensor, *, seed_e0: int, rep_seed: int,
                                 noise_dim: int, epsilon: float, n_samples: int = 256) -> dict:
    """Compara gradientes das duas perdas no MESMO A20, x e z, sem atualizar pesos.

    Executa o gerador em eval para impedir mudança dos running stats do BatchNorm.
    O gerador volta ao modo anterior. Nenhum otimizador é criado nem há seleção de
    hiperparâmetros baseada em teste. Usa somente amostras de treinamento.
    """
    if n_samples < 2 or len(x_ddos) < 2:
        raise ValueError('Diagnóstico exige ao menos duas amostras')
    original_train_mode = model.training
    model.eval()
    defender.eval()
    n = min(n_samples, len(x_ddos))
    xb = x_ddos[:n]
    generator = torch.Generator(device='cpu').manual_seed(stable_seed('e5a2-diagnostico-z', seed_e0, rep_seed))
    z = torch.randn(n, noise_dim, generator=generator)
    output = {'n_amostras_treino': n}
    try:
        for mode in ('probs', 'logits'):
            model.zero_grad(set_to_none=True)
            delta = generate_delta(model, arm, xb, z)
            loss, probs, logits = perda_e5(defender, xb + epsilon * delta, mode)
            loss.backward()
            grad_total, grad_ramo = norma_gradiente(model)
            output[mode] = {
                'loss': float(loss.detach()),
                'gradiente_l2': grad_total,
                'gradiente_ramo_l2': grad_ramo,
                'fracao_prob_exatamente_um': float((probs == 1).float().mean()),
                'fracao_prob_exatamente_zero': float((probs == 0).float().mean()),
                'media_logit': float(logits.mean()) if logits is not None else None,
            }
    finally:
        model.zero_grad(set_to_none=True)
        model.train(original_train_mode)
    return output


def train_one(model: nn.Module, arm: str, defender: nn.Module, x_ddos: torch.Tensor,
              x_val: np.ndarray, *, seed_e0: int, rep_seed: int, epochs: int,
              batch_size: int, noise_dim: int, epsilon: float, lr: float,
              betas: tuple[float, float], eval_batch: int, loss_mode: str = 'probs'):
    """A ordem dos batches e z são idênticos entre braços para a mesma réplica."""
    if len(x_ddos) < 2 or batch_size < 2 or epochs < 1:
        raise ValueError('Orçamento de treino insuficiente')
    model.train()
    defender.eval()
    defender.requires_grad_(False)
    before_d = {k: v.detach().clone() for k, v in defender.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    order_rng = np.random.default_rng(stable_seed('e5-order', seed_e0, rep_seed))
    z_rng = torch.Generator(device='cpu').manual_seed(stable_seed('e5-train-z', seed_e0, rep_seed))
    curves = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = order_rng.permutation(len(x_ddos))
        losses = []
        normas_grad = []
        normas_ramo = []
        saturados_um = saturados_zero = total_probs = 0
        logit_sum = logit_n = 0.0
        for st in range(0, len(order), batch_size):
            ids = order[st:st + batch_size]
            if len(ids) < 2:  # BatchNorm original exige pelo menos 2 amostras.
                continue
            xb = x_ddos[ids]
            z = torch.randn(len(xb), noise_dim, generator=z_rng)
            delta = generate_delta(model, arm, xb, z)
            if not torch.isfinite(delta).all() or float(delta.detach().abs().max()) > 1.00001:
                raise FloatingPointError('Perturbação inválida durante o treinamento')
            loss, probs, logits = perda_e5(defender, xb + epsilon * delta, loss_mode)
            if not torch.isfinite(loss):
                raise FloatingPointError('Loss não finita')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if loss_mode == 'logits':
                grad_norm, branch_norm = norma_gradiente(model)
                normas_grad.append(grad_norm)
                if branch_norm is not None:
                    normas_ramo.append(branch_norm)
                saturados_um += int((probs == 1).sum())
                saturados_zero += int((probs == 0).sum())
                total_probs += probs.numel()
                logit_sum += float(logits.sum())
                logit_n += logits.numel()
            optimizer.step()
            losses.append(float(loss.detach()))
        if not losses:
            raise RuntimeError('Nenhum batch foi treinado')
        val = evaluate(model, arm, defender, x_val, noise_dim=noise_dim,
                       noise_seed=stable_seed('e5-val', seed_e0, rep_seed),
                       epsilon=epsilon, batch_size=eval_batch)
        epoch_record = {
            'epoch': epoch, 'loss_media': float(np.mean(losses)),
            'asr_condicional_validacao': val['asr_condicional'],
            'n_detectados_validacao': val['n_detectados_limpos'],
            'n_sucesso_validacao': val['n_sucesso'],
        }
        if loss_mode == 'logits':
            epoch_record.update({
                'grad_l2_media': float(np.mean(normas_grad)),
                'grad_l2_min': float(np.min(normas_grad)),
                'grad_l2_max': float(np.max(normas_grad)),
                'fracao_batches_grad_zero': float(np.mean(np.asarray(normas_grad) == 0)),
                'grad_ramo_l2_media': float(np.mean(normas_ramo)) if normas_ramo else None,
                'fracao_prob_exatamente_um': saturados_um / total_probs,
                'fracao_prob_exatamente_zero': saturados_zero / total_probs,
                'media_logit': logit_sum / logit_n,
            })
        curves.append(epoch_record)
        marker = '[E5-A2]' if loss_mode == 'logits' else '[E5-A]'
        detail = f' grad_l2={epoch_record["grad_l2_media"]:.4g}' if loss_mode == 'logits' else ''
        print(f'{marker} seed={seed_e0} replica={rep_seed} {arm} '
              f'epoch={epoch}/{epochs} loss={curves[-1]["loss_media"]:.5f} '
              f'ASR_val={val["asr_condicional"]}{detail}', flush=True)
    for k, tensor in before_d.items():
        if not torch.equal(tensor, defender.state_dict()[k]):
            raise AssertionError(f'Defender modificado durante E5-A: {k}')
    return curves, time.perf_counter() - t0


def _frozen_source_attack(model: nn.Module, arm: str, x_src: torch.Tensor,
                          z: torch.Tensor, epsilon: float):
    delta = generate_delta(model, arm, x_src, z)
    if not torch.isfinite(delta).all() or float(delta.abs().max()) > 1.00001:
        raise FloatingPointError('Delta não-finito ou fora de [-1,1]')
    return delta, x_src + epsilon * delta


def evaluate(model: nn.Module, arm: str, defender: nn.Module, x_source: np.ndarray,
             *, noise_dim: int, noise_seed: int, epsilon: float = EPSILON,
             batch_size: int = 1024, source_scaler=None, target_scaler=None,
             x_raw: np.ndarray | None = None) -> dict:
    """ASR condicionado a D20 identificar corretamente o DDoS limpo.

    Em transferência, x_raw são os dados brutos compartilhados. As perturbações
    são geradas no espaço da origem e convertidas para escala do destino.
    """
    ensure_finite(x_source, 'X de origem')
    if x_raw is not None and (source_scaler is None or target_scaler is None):
        raise ValueError('Transferência exige os dois scalers')
    if x_raw is not None and x_raw.shape != x_source.shape:
        raise ValueError('X_raw e X_source com shapes diferentes')
    model.eval(); defender.eval()
    gen1 = torch.Generator(device='cpu').manual_seed(noise_seed)
    gen2 = torch.Generator(device='cpu').manual_seed(stable_seed('e5-div', noise_seed))
    n = n_detected = n_success = n_clean_miss = n_adv_evasion = 0
    sum_conf = sum_l1 = sum_l2 = sum_linf = sum_div = sum_div_success = 0.0
    successful_pairs = saturated = 0
    source_n = 0
    with torch.no_grad():
        for st in range(0, len(x_source), batch_size):
            xb = torch.from_numpy(np.asarray(x_source[st:st + batch_size], dtype=np.float32))
            z1 = torch.randn(len(xb), noise_dim, generator=gen1)
            z2 = torch.randn(len(xb), noise_dim, generator=gen2)
            delta1, adv_src1 = _frozen_source_attack(model, arm, xb, z1, epsilon)
            delta2, adv_src2 = _frozen_source_attack(model, arm, xb, z2, epsilon)
            if x_raw is None:
                clean = xb
                adv1, adv2 = adv_src1, adv_src2
            else:
                raw = np.asarray(x_raw[st:st + batch_size], dtype=np.float64)
                # Reconstrução em float64 para evitar perda na ida/volta do scaler.
                clean_np = target_scaler.transform(raw)
                src_np = np.asarray(x_source[st:st + batch_size], dtype=np.float64)
                sc = np.asarray(source_scaler.scale_, dtype=np.float64)
                mn = np.asarray(source_scaler.mean_, dtype=np.float64)
                dest_mean = np.asarray(target_scaler.mean_, dtype=np.float64)
                dest_scale = np.asarray(target_scaler.scale_, dtype=np.float64)
                raw_adv1 = (src_np + epsilon * delta1.numpy().astype(np.float64)) * sc + mn
                raw_adv2 = (src_np + epsilon * delta2.numpy().astype(np.float64)) * sc + mn
                clean = torch.from_numpy(clean_np.astype(np.float32))
                adv1 = torch.from_numpy(((raw_adv1 - dest_mean) / dest_scale).astype(np.float32))
                adv2 = torch.from_numpy(((raw_adv2 - dest_mean) / dest_scale).astype(np.float32))
            p_clean = defender(clean).view(-1)
            p1 = defender(adv1).view(-1)
            p2 = defender(adv2).view(-1)
            detected = p_clean >= THRESHOLD
            success1 = detected & (p1 < THRESHOLD)
            success2 = detected & (p2 < THRESHOLD)
            pairs = success1 & success2
            pair_distance = torch.linalg.vector_norm(delta1 - delta2, dim=1)
            move = epsilon * delta1
            n += len(xb)
            n_detected += int(detected.sum())
            n_success += int(success1.sum())
            n_clean_miss += int((~detected).sum())
            n_adv_evasion += int((p1 < THRESHOLD).sum())
            successful_pairs += int(pairs.sum())
            saturated += int((delta1.abs() >= 0.999).sum())
            source_n += delta1.numel()
            sum_conf += float(p1.sum())
            sum_l1 += float(torch.linalg.vector_norm(move, ord=1, dim=1).sum())
            sum_l2 += float(torch.linalg.vector_norm(move, ord=2, dim=1).sum())
            sum_linf += float(torch.linalg.vector_norm(move, ord=float('inf'), dim=1).sum())
            sum_div += float(pair_distance.sum())
            sum_div_success += float(pair_distance[pairs].sum())
    if n == 0:
        raise ValueError('Avaliação vazia')
    return {
        'n': n, 'n_detectados_limpos': n_detected, 'n_sucesso': n_success,
        'n_erro_limpo': n_clean_miss,
        'asr_condicional': n_success / n_detected if n_detected else None,
        'erro_limpo': n_clean_miss / n,
        'evasao_nao_condicional': n_adv_evasion / n,
        'confianca_adv_media': sum_conf / n,
        'delta_l1_media': sum_l1 / n, 'delta_l2_media': sum_l2 / n,
        'delta_linf_media': sum_linf / n,
        'diversidade_l2_mesma_amostra': sum_div / n,
        'pares_ambos_sucesso': successful_pairs,
        'diversidade_l2_pares_eficazes': sum_div_success / successful_pairs if successful_pairs else None,
        'fracao_componentes_saturados': saturated / source_n,
    }


@dataclass
class Run:
    seed: int
    path: Path
    config: dict
    pre: Preprocessador
    train_ids: np.ndarray
    val_ids: np.ndarray
    test_ids: np.ndarray
    checkpoint_attacker: dict
    defender: nn.Module


def locate_run(root: Path, seed: int) -> Path:
    found = []
    for path in sorted(root.glob(f'run_*_seed{seed}')):
        config = path / 'config_execucao.json'
        if config.is_file() and json.loads(config.read_text(encoding='utf8')).get('seed') == seed:
            found.append(path)
    if len(found) != 1:
        raise ValueError(f'Seed {seed}: esperado exatamente um run E0; encontrados {len(found)}')
    return found[0]


def check_preflight(report_path: Path, runs_root: Path, seeds: tuple[int, ...]) -> dict:
    info = json.loads(report_path.read_text(encoding='utf-8'))
    if info.get('status') != 'aprovado' or info.get('experimento') != 'E5-P':
        raise AssertionError('E5-P não está aprovado')
    entries = {int(item['seed']): item for item in info['seeds']}
    for seed in seeds:
        if seed not in entries or entries[seed]['status'] != 'aprovado':
            raise AssertionError(f'E5-P não aprovado para a seed {seed}')
        path = locate_run(runs_root, seed)
        for name, relative in [('sha256_a20', 'checkpoints/atacante_rodada_20.pth'),
                               ('sha256_d20', 'checkpoints/defensor_rodada_20.pth'),
                               ('sha256_scaler', 'preprocessador/scaler.pkl')]:
            if sha256_file(path / relative) != entries[seed][name]:
                raise AssertionError(f'Seed {seed}: hash {name} divergiu do E5-P')
        if entries[seed]['maior_diferenca_inicial_a0_a1'] > 1e-7 or not entries[seed]['defender_congelado']:
            raise AssertionError(f'Seed {seed}: equivalência/freeze E5-P falhou')
    return entries


def prepare_runs(runs_root: Path, preflight: Path, seeds: tuple[int, ...]):
    from adarena.builtin import create_default_registry
    checked = check_preflight(preflight, runs_root, seeds)
    paths = {s: locate_run(runs_root, s) for s in seeds}
    configs = {s: json.loads((p / 'config_execucao.json').read_text(encoding='utf8')) for s, p in paths.items()}
    dataset_specs = [(c['dataset_config']['component_id'], c['dataset_config']['source'],
                      json.dumps(c['dataset_config'].get('params', {}), sort_keys=True)) for c in configs.values()]
    if len(set(dataset_specs)) != 1:
        raise AssertionError('E0 das seeds utiliza fontes/adapters diferentes')
    name, source, _ = dataset_specs[0]
    dataset = create_default_registry().create(name, **configs[seeds[0]]['dataset_config'].get('params', {})).load(source)
    X, y = np.asarray(dataset.X), np.asarray(dataset.y)
    features = list(dataset.schema.feature_names)
    ensure_finite(X, 'dataset E0 bruto')
    keep = deduplicate_indices(X, y)
    removed = len(X) - len(keep)
    X, y = X[keep], y[keep]
    runs = {}
    for seed in seeds:
        path = paths[seed]
        cfg = configs[seed]
        p = Preprocessador.carregar(path / 'preprocessador')
        data_cfg = cfg['dados']
        if cfg['components']['defender'] != 'adarena.binary_mlp_defender' or cfg['components']['attacker'] != 'adarena.perturbation_attacker':
            raise AssertionError('Componentes E0 incompatíveis')
        if data_cfg['feature_names'] != features or p.feature_names != features:
            raise AssertionError(f'Seed {seed}: ordem de features diferente do E0')
        if int(data_cfg['noise_dim']) != 32 or int(data_cfg['input_dim']) != X.shape[1]:
            raise AssertionError('Dimensão do E0 inesperada')
        tr = cfg['treinamento']
        if float(tr['epsilon']) != EPSILON or float(tr['classification_threshold']) != THRESHOLD:
            raise AssertionError('Orçamento ou limiar inesperados')
        train_ids, val_ids, test_ids = split_indices(y, seed, float(data_cfg['test_size']), float(data_cfg['validation_size']))
        audit = p.audit_info
        if (int(audit.get('duplicatas_removidas', -1)) != removed or
            any(int(audit['split'][k]) != len(ids) for k, ids in [('treino', train_ids), ('validacao', val_ids), ('teste', test_ids)]) or
            int(audit['random_state']) != seed or
            any(int(audit['intersecoes'][k]) != 0 for k in ['treino_validacao', 'treino_teste', 'validacao_teste'])):
            raise AssertionError(f'Seed {seed}: splits/dedup E0 divergentes')
        check_scaler = StandardScaler().fit(X[train_ids])
        for key in ('mean_', 'var_', 'scale_'):
            expected, actual = np.asarray(getattr(check_scaler, key)), np.asarray(getattr(p.scaler, key))
            if not np.allclose(expected, actual, rtol=3e-6, atol=1e-8):
                raise AssertionError(f'Seed {seed}: scaler {key} não corresponde ao split')
        state_a = torch.load(path / 'checkpoints/atacante_rodada_20.pth', map_location='cpu', weights_only=True)
        state_d = torch.load(path / 'checkpoints/defensor_rodada_20.pth', map_location='cpu', weights_only=True)
        d = frozen_defender(state_d, input_dim=X.shape[1])
        runs[seed] = Run(seed, path, cfg, p, train_ids, val_ids, test_ids, state_a, d)
        print(f'[E5] Seed {seed}: split E0 reproduzido, scaler confirmado, D20 congelado', flush=True)
    return X, y, runs, removed, source


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def save_weights(path: Path, state: dict):
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, tmp)
    tmp.replace(path)


def run(args):
    loss_mode = getattr(args, 'loss_mode', 'probs')
    if loss_mode not in ('probs', 'logits'):
        raise ValueError('loss_mode deve ser probs ou logits')
    experiment = 'E5-A2' if loss_mode == 'logits' else 'E5-A'
    seeds = tuple(sorted(set(args.seeds)))
    replicas = tuple(args.replicas)
    if len(seeds) != len(args.seeds) or len(replicas) != len(set(replicas)):
        raise ValueError('Seeds e réplicas devem ser distintas')
    if len(seeds) != 2:
        raise ValueError('E5-A exige exatamente duas seeds E0 para transferência pareada')
    if args.epochs < 1 or args.batch_size < 2 or args.val_samples < 1 or args.test_samples < 1 or args.max_train_samples < 0:
        raise ValueError('Orçamento inválido')
    if args.device != 'cpu':
        raise ValueError('E5-A atual só suporta CPU para manter controle de RNG pareado')
    config = {
        'experiment': 'E5-A', 'seeds': list(seeds), 'replicas': list(replicas),
        'epochs': args.epochs, 'batch_size': args.batch_size,
        'max_train_samples': args.max_train_samples,
        'val_samples': args.val_samples, 'test_samples': args.test_samples,
        'eval_batch': args.eval_batch, 'device': args.device,
        'epsilon': EPSILON, 'threshold': THRESHOLD,
        'init_from': 'A20 E0 (Adam novo)', 'checkpoint_selection': 'epoch final fixo',
        'preflight_hash': sha256_file(args.preflight),
        'run_root': str(args.runs_root),
        'dataset': 'DDoS-Friday (origem config E0)',
    }
    if loss_mode == 'logits':
        config.update({
            'experiment': experiment,
            'loss_mode': 'logits',
            'loss': 'BCEWithLogitsLoss sobre saida pre-Sigmoid de D20',
            'diagnostico_inicial': 'BCE probabilidades vs logits no mesmo batch de treino',
        })
    config_path = args.output_dir / ('config_e5a2.json' if loss_mode == 'logits' else 'config_e5a.json')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.resume:
            raise FileExistsError('Pasta de saída não vazia; use --resume para continuar')
        if not config_path.exists() or json.loads(config_path.read_text(encoding='utf8')) != config:
            raise RuntimeError('Configuração existente diferente; --resume recusado')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(config_path, config)

    X, y, runs, dedup, source = prepare_runs(args.runs_root, args.preflight, seeds)
    test_shared = np.intersect1d(runs[seeds[0]].test_ids, runs[seeds[1]].test_ids, assume_unique=True)
    shared_ddos = select_ddos(test_shared, y, args.test_samples,
                              stable_seed('e5-shared-test', *seeds))
    print(f'[E5-A] Teste comum: {len(shared_ddos)} DDoS em interseção test/test. '
          'Transferência nos dois sentidos com mesma amostragem.', flush=True)
    rows = []
    for seed in seeds:
        info = runs[seed]
        target_seed = next(s for s in seeds if s != seed)
        target = runs[target_seed]
        # Registros da mesma base nunca usados para treinar esse gerador.
        x_train = X[info.train_ids]
        y_train = y[info.train_ids]
        train_ddos = x_train[y_train == 1]
        if args.max_train_samples and args.max_train_samples < len(train_ddos):
            ids = np.random.default_rng(stable_seed('e5-train-subset', seed)).choice(
                len(train_ddos), size=args.max_train_samples, replace=False)
            train_ddos = train_ddos[np.sort(ids)]
        x_ddos_t = torch.from_numpy(info.pre.scaler.transform(train_ddos).astype(np.float32))
        val_ids = select_ddos(info.val_ids, y, args.val_samples, stable_seed('e5-val-selection', seed))
        x_val = info.pre.scaler.transform(X[val_ids]).astype(np.float32)
        x_pair_raw = np.asarray(X[shared_ddos], dtype=np.float64)
        x_pair_source = info.pre.scaler.transform(x_pair_raw).astype(np.float32)
        # Memória: validar que nenhum ID de avaliação participa do treino.
        if len(np.intersect1d(shared_ddos, info.train_ids)) or len(np.intersect1d(val_ids, info.train_ids)):
            raise AssertionError('Leakage E5-A: IDs do treino presentes na avaliação')
        for rep in replicas:
            tcfg = info.config['treinamento']
            lr = float(tcfg['lr_atacante'])
            betas = tuple(float(v) for v in tcfg['adam_betas_atacante'])
            if len(betas) != 2:
                raise AssertionError('Adam betas inválidos')
            for arm in ('A0', 'A1'):
                destination = args.output_dir / f'seed_{seed}' / f'rep_{rep}' / arm
                destination.mkdir(parents=True, exist_ok=True)
                result_path = destination / 'resultado.json'
                weight_path = destination / 'atacante_final.pth'
                if args.resume and result_path.exists() and weight_path.exists():
                    existing = json.loads(result_path.read_text(encoding='utf8'))
                    if existing['seed_e0'] != seed or existing['replica'] != rep or existing['braco'] != arm:
                        raise RuntimeError(f'Resultado anterior inválido: {result_path}')
                    print(f'[E5-A] Reaproveitando {seed}/{rep}/{arm}', flush=True)
                    rows.append(existing)
                    continue
                init_seed = stable_seed('e5-init', seed, rep)
                attacker = init_attacker(arm, info.checkpoint_attacker, input_dim=X.shape[1],
                                         noise_dim=32, init_seed=init_seed)
                # A0/A1 começam com a mesma saída em eval, mesmo z e batch.
                attacker.eval()
                with torch.no_grad():
                    gen0 = torch.Generator(device='cpu').manual_seed(stable_seed('e5-id', seed, rep))
                    zcheck = torch.randn(32, 32, generator=gen0)
                    xcheck = x_ddos_t[:32]
                    if len(xcheck) != 32:
                        raise ValueError('Treino precisa de pelo menos 32 amostras para controle')
                    check_a0 = Atacante(noise_dim=32, output_dim=X.shape[1])
                    check_a0.load_state_dict(info.checkpoint_attacker, strict=True)
                    check_a0.eval()
                    diff = float((generate_delta(attacker, arm, xcheck, zcheck) - check_a0(zcheck)).abs().max())
                    if diff > 1e-7:
                        raise AssertionError(f'Identidade inicial A0/A1 falhou: {diff}')
                diagnostico = (diagnosticar_perdas_iniciais(
                    attacker, arm, info.defender, x_ddos_t,
                    seed_e0=seed, rep_seed=rep, noise_dim=32, epsilon=EPSILON)
                    if loss_mode == 'logits' else None)
                curves, elapsed = train_one(
                    attacker, arm, info.defender, x_ddos_t, x_val,
                    seed_e0=seed, rep_seed=rep, epochs=args.epochs,
                    batch_size=args.batch_size, noise_dim=32, epsilon=EPSILON,
                    lr=lr, betas=betas, eval_batch=args.eval_batch, loss_mode=loss_mode)
                # O teste só é acessado após todos os epochs; não seleciona modelo.
                result_src = evaluate(
                    attacker, arm, info.defender, x_pair_source, noise_dim=32,
                    noise_seed=stable_seed('e5-test', *seeds, rep),
                    epsilon=EPSILON, batch_size=args.eval_batch)
                result_dst = evaluate(
                    attacker, arm, target.defender, x_pair_source, noise_dim=32,
                    noise_seed=stable_seed('e5-test', *seeds, rep), epsilon=EPSILON,
                    batch_size=args.eval_batch, source_scaler=info.pre.scaler,
                    target_scaler=target.pre.scaler, x_raw=x_pair_raw)
                result = {
                    'experiment': experiment, 'seed_e0': seed, 'replica': rep, 'braco': arm,
                    'defender_alvo': seed, 'defender_transferencia': target_seed,
                    'n_treino_ddos': len(x_ddos_t), 'n_validacao_ddos': len(x_val),
                    'n_teste_comum_ddos': len(shared_ddos),
                    'ids_teste_comum_sha256': sha256_indices(shared_ddos),
                    'noise_seed_teste': stable_seed('e5-test', *seeds, rep),
                    'init_a0_a1_max_abs_diff': diff,
                    'n_parametros': sum(p.numel() for p in attacker.parameters()),
                    'optimizer': {'type': 'Adam novo', 'lr': lr, 'betas': list(betas)},
                    'epochs': args.epochs, 'batch_size': args.batch_size,
                    'epsilon': EPSILON, 'tempo_treinamento_segundos': elapsed,
                    'curva_validacao': curves,
                    'eficacia_alvo_teste': result_src,
                    'transferencia_teste': result_dst,
                }
                if loss_mode == 'logits':
                    result['loss_mode'] = loss_mode
                    result['diagnostico_pre_treino'] = diagnostico
                save_weights(weight_path, attacker.state_dict())
                atomic_json(result_path, result)
                print(f'[E5-A] Salvo: {result_path} | ASR alvo={result_src["asr_condicional"]} '
                      f'| ASR transferência={result_dst["asr_condicional"]}', flush=True)
                rows.append(result)
    # Resumo são apenas métricas efetivamente produzidas; sem alegar resultados aqui.
    flat = []
    for record in rows:
        flat.append({
            'seed_e0': record['seed_e0'], 'replica': record['replica'], 'braco': record['braco'],
            'seed_defender_transferencia': record['defender_transferencia'],
            'n_treino_ddos': record['n_treino_ddos'],
            'n_teste_comum_ddos': record['n_teste_comum_ddos'],
            'asr_alvo': record['eficacia_alvo_teste']['asr_condicional'],
            'asr_transferencia': record['transferencia_teste']['asr_condicional'],
            'diversidade_l2_alvo': record['eficacia_alvo_teste']['diversidade_l2_mesma_amostra'],
            'diversidade_l2_eficaz_alvo': record['eficacia_alvo_teste']['diversidade_l2_pares_eficazes'],
            'n_pares_eficazes': record['eficacia_alvo_teste']['pares_ambos_sucesso'],
            'tempo_treino_segundos': record['tempo_treinamento_segundos'],
        })
    with (args.output_dir / 'resultados_e5a.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(sorted(flat, key=lambda r: (r['seed_e0'], r['replica'], r['braco'])))
    atomic_json(args.output_dir / 'resumo_e5a.json', {
        'experiment': experiment, 'status': 'treinamentos_concluidos',
        'dataset': source, 'seeds': list(seeds), 'replicas': list(replicas),
        'n_treinamentos': len(rows), 'deduplicatas_originais': dedup,
        'n_teste_comum_ddos': len(shared_ddos),
        'ids_teste_comum_sha256': sha256_indices(shared_ddos),
        'reproducao': 'Splits e scaler E0 auditados; A0/A1 inicializados de A20; D20 congelado',
        'critica': 'Comparação de arquitetura+condicionamento (capacidades diferentes); perturbações em espaço normalizado, não tráfego físico',
        'resultados': sorted(flat, key=lambda r: (r['seed_e0'], r['replica'], r['braco'])),
        **({'loss_mode': loss_mode, 'controle': 'Mesmo protocolo E5-A, mudando somente a formulação da loss',
            'diagnosticos_iniciais': [
                {'seed_e0': r['seed_e0'], 'replica': r['replica'], 'braco': r['braco'],
                 **r['diagnostico_pre_treino']} for r in rows
            ]} if loss_mode == 'logits' else {}),
    })
    print(f'[{experiment}] Finalizado: {len(rows)} treinamentos. Resumo: {args.output_dir}/resumo_e5a.json', flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, default=Path('models/e0/experiments'))
    p.add_argument('--preflight', type=Path, default=Path('models/e5/preflight/e5_preflight.json'))
    p.add_argument('--seeds', nargs='+', type=int, default=[20, 42])
    p.add_argument('--replicas', nargs='+', type=int, default=[1042, 1043, 1044])
    p.add_argument('--loss-mode', choices=['probs', 'logits'], default='probs',
                   help='probs: reprodução E5-A; logits: diagnóstico E5-A2 com BCEWithLogitsLoss')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--max-train-samples', type=int, default=0, help='0: treino DDoS completo; >0: subconjunto fixo só para teste rápido')
    p.add_argument('--val-samples', type=int, default=4096)
    p.add_argument('--test-samples', type=int, default=5000)
    p.add_argument('--eval-batch', type=int, default=1024)
    p.add_argument('--device', choices=['cpu'], default='cpu')
    p.add_argument('--output-dir', type=Path, default=Path('models/e5/piloto'))
    p.add_argument('--resume', action='store_true', help='Continuar apenas resultados de mesma configuração, preservando trabalhos completos')
    return p.parse_args(argv)


if __name__ == '__main__':
    run(parse_args())
