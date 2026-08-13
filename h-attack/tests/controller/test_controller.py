import numpy as np
import pytest

from controller.controller import AttackController
from sender.sender import AttackResult
from translator.translator import AttackParams


def _params(valid: bool) -> AttackParams:
    return AttackParams(
        target_ip="172.20.0.10",
        target_port=80,
        packets_per_second=200.0,
        packet_size=100,
        duration_seconds=0.5,
        use_tcp=True,
        tcp_flags="A",
        use_syn_flood=False,
        randomize_src_ip=True,
        randomize_src_port=True,
        window_size=4096,
        translation_valid=valid,
        translation_warnings=[] if valid else ["inconsistente"],
        consistency_error=0.0 if valid else 2.0,
    )


class FakeTranslator:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = 0

    def traduzir_batch(self, vetores, evasao_probs=None, only_valid=False):
        self.calls += 1
        assert only_valid is False
        return self.outputs


class FakeSender:
    def __init__(self):
        self.params = []

    def executar(self, params):
        self.params.append(params)
        return AttackResult(
            params=params,
            packets_sent=100,
            bytes_sent=10_000,
            duration_actual=0.5,
            success=True,
            dry_run=True,
        )


def _controller(tmp_path, probs, translated):
    controller = AttackController.__new__(AttackController)
    controller.target_ip = "172.20.0.10"
    controller.target_port = 80
    controller.classification_threshold = 0.5
    controller.dry_run = True
    controller.models_dir = tmp_path
    controller.historico = []
    controller.translator = FakeTranslator(translated)
    controller.sender = FakeSender()

    vetores = np.arange(len(probs) * 3, dtype=float).reshape(len(probs), 3)
    controller._gerar_e_avaliar = lambda n: (
        vetores[:n],
        np.asarray(probs[:n], dtype=float),
    )
    return controller


def test_controller_envia_apenas_traducoes_validas(tmp_path):
    controller = _controller(
        tmp_path,
        probs=[0.1, 0.2, 0.8],
        translated=[_params(False), _params(True)],
    )

    results = controller.executar_ciclo(n_vetores=3)

    assert len(results) == 1
    assert len(controller.sender.params) == 1
    assert controller.sender.params[0].translation_valid is True

    entry = controller.historico[-1]
    assert entry["n_vetores_gerados"] == 3
    assert entry["n_evasoes"] == 2
    assert entry["n_traducoes_validas"] == 1
    assert entry["n_traducoes_invalidas"] == 1
    assert entry["n_ataques_executados"] == 1
    assert entry["taxa_evasao"] == pytest.approx(2 / 3)
    assert entry["taxa_validade_translator"] == pytest.approx(0.5)
    assert entry["taxa_evasao_plausivel"] == pytest.approx(1 / 3)


def test_controller_zero_plausiveis_nao_chama_sender(tmp_path):
    controller = _controller(
        tmp_path,
        probs=[0.1, 0.2],
        translated=[_params(False), _params(False)],
    )

    results = controller.executar_ciclo(n_vetores=2)

    assert results == []
    assert controller.sender.params == []
    entry = controller.historico[-1]
    assert entry["n_evasoes"] == 2
    assert entry["n_traducoes_validas"] == 0
    assert entry["n_traducoes_invalidas"] == 2
    assert entry["n_ataques_executados"] == 0
    assert entry["taxa_evasao_plausivel"] == 0.0


def test_controller_zero_evasoes_registra_ciclo_sem_translator(tmp_path):
    controller = _controller(
        tmp_path,
        probs=[0.7, 0.9],
        translated=[],
    )

    results = controller.executar_ciclo(n_vetores=2)

    assert results == []
    assert controller.translator.calls == 0
    assert controller.sender.params == []
    entry = controller.historico[-1]
    assert entry["n_evasoes"] == 0
    assert entry["n_traducoes_validas"] == 0
    assert entry["taxa_evasao"] == 0.0


def test_historico_e_persistido_apos_cada_ciclo(tmp_path):
    controller = _controller(
        tmp_path,
        probs=[0.1],
        translated=[_params(True)],
    )

    controller.executar_ciclo(n_vetores=1)

    path = tmp_path / "historico_ataques.json"
    assert path.exists()
    assert '"n_traducoes_validas": 1' in path.read_text()


def test_historicos_de_execucoes_distintas_nao_sobrescrevem(tmp_path):
    from pathlib import Path

    def novo_controller(mode):
        controller = AttackController.__new__(AttackController)
        controller.models_dir = tmp_path
        controller.dry_run = True
        controller.checkpoint_mode = mode
        controller.attacker_round = None
        controller.defender_round = None
        controller.target_ip = "172.20.0.10"
        controller.target_port = 80
        controller.historico = [{"n_vetores_gerados": 1}]
        controller._preparar_historico_execucao(
            Path("atacante.pth"),
            Path("defensor.pth"),
        )
        controller._salvar_historico()
        return controller.network_history_path

    demo_path = novo_controller("demo")
    final_path = novo_controller("final")

    assert demo_path.exists()
    assert final_path.exists()
    assert demo_path != final_path
    assert demo_path.parent == tmp_path / "network_runs"
    assert final_path.parent == tmp_path / "network_runs"
