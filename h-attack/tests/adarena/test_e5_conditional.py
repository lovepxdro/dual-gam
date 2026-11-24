"""Testes unitários do gerador A1 (não usam dataset nem checkpoint E0)."""
import pytest
import torch

from gan.e5_conditional import AtacanteCondicional
from gan.models import Atacante, Defensor


def modelos(d=7, n=5):
    torch.manual_seed(31)
    antigo = Atacante(noise_dim=n, output_dim=d).eval()
    novo = AtacanteCondicional(noise_dim=n, input_dim=d).eval()
    novo.carregar_base_a20(antigo.state_dict())
    return antigo, novo


def test_identidade_inicial_exacta_e_invariancia_a_x():
    a0, a1 = modelos()
    x = torch.randn(9, 7)
    z = torch.randn(9, 5)
    with torch.no_grad():
        assert torch.equal(a0(z), a1(x, z))
        assert torch.equal(a0(z), a1(x + 100, z))


def test_limites_e_orcamento_apos_modificar_ramo():
    _, a1 = modelos()
    with torch.no_grad():
        a1.ramo[-1].weight.normal_(mean=0, std=0.1)
        a1.ramo[-1].bias.fill_(0.1)
    x = torch.randn(24, 7)
    z = torch.randn(24, 5)
    delta = a1(x, z)
    assert delta.shape == x.shape
    assert torch.isfinite(delta).all()
    assert delta.abs().max() <= 1 + 1e-6
    assert (a1.perturba(x, z, epsilon=0.3) - x).abs().max() <= 0.300001
    assert not torch.equal(delta, a1(x + 1.0, z))


def test_backprop_defender_congelado():
    _, a1 = modelos()
    d20 = Defensor(input_dim=7).eval().requires_grad_(False)
    x, z = torch.randn(12, 7), torch.randn(12, 5)
    y = d20(x + .3 * a1(x, z))
    y.mean().backward()
    assert a1.ramo[-1].weight.grad is not None
    assert a1.base.rede[0].weight.grad is not None
    assert all(p.grad is None for p in d20.parameters())


def test_entrada_invalida_rejeitada():
    _, a1 = modelos()
    with pytest.raises(ValueError, match="Shapes"):
        a1(torch.zeros(2, 7), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="NaN"):
        a1(torch.full((2, 7), float("nan")), torch.zeros(2, 5))
    with pytest.raises(ValueError, match="epsilon"):
        a1.perturba(torch.zeros(2, 7), torch.zeros(2, 5), -0.1)


def test_carregamento_estrito_legado():
    _, a1 = modelos()
    with pytest.raises(RuntimeError):
        a1.carregar_base_a20({"invalido": torch.zeros(1)})


def test_gradientes_do_ramo_mesmo_inicializacao_zero():
    _, a1 = modelos()
    x, z = torch.randn(10, 7), torch.randn(10, 5)
    a1(x, z).square().sum().backward()
    assert a1.ramo[-1].weight.grad.abs().sum() > 0
