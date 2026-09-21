from __future__ import annotations

import logging

from typer.testing import CliRunner

from adarena.cli.app import app
from adarena.logging_config import (
    configure_console_logging,
)


runner = CliRunner()


def _adarena_console_handlers():
    return [
        handler
        for handler
        in logging.getLogger().handlers
        if getattr(
            handler,
            "_adarena_console_handler",
            False,
        )
    ]


def test_cli_version():
    result = runner.invoke(
        app,
        ["version"],
    )

    assert result.exit_code == 0
    assert "ADArena 2.4.0" in result.stdout


def test_cli_components_lista_registry():
    result = runner.invoke(
        app,
        ["components"],
    )

    assert result.exit_code == 0

    assert (
        "adarena.binary_mlp_defender"
        in result.stdout
    )

    assert (
        "adarena.scapy_capture"
        in result.stdout
    )

    assert (
        "adarena.network_observation"
        in result.stdout
    )


def test_cli_components_filtra_por_kind():
    result = runner.invoke(
        app,
        [
            "components",
            "--kind",
            "defender",
        ],
    )

    assert result.exit_code == 0

    assert (
        "adarena.binary_mlp_defender"
        in result.stdout
    )

    assert (
        "adarena.perturbation_attacker"
        not in result.stdout
    )


def test_cli_components_rejeita_kind_invalido():
    result = runner.invoke(
        app,
        [
            "components",
            "--kind",
            "banana",
        ],
    )

    assert result.exit_code != 0

    assert (
        "tipo de componente inválido"
        in result.output
    )


def test_console_logging_normal_usa_info():
    configure_console_logging(
        verbose=False
    )

    handlers = (
        _adarena_console_handlers()
    )

    assert len(handlers) == 1
    assert handlers[0].level == logging.INFO


def test_console_logging_verbose_usa_debug():
    configure_console_logging(
        verbose=True
    )

    handlers = (
        _adarena_console_handlers()
    )

    assert len(handlers) == 1
    assert handlers[0].level == logging.DEBUG



def test_cli_config_validate_observe(
    tmp_path,
):
    config_path = (
        tmp_path
        / "observe.toml"
    )

    config_path.write_text(
        """
mode = "observe"
output_dir = "models"

[defender]
component_id = "adarena.binary_mlp_defender"
source = "defender.pth"

[protocol]
component_id = "adarena.network_observation"

[network]
preprocessor_source = "preprocessador"
capture_duration = 5.0
packet_limit = 100
classification_threshold = 0.5
dry_run = true

[network.capture]
component_id = "adarena.scapy_capture"

[network.capture.params]
iface = "test0"
bpf_filter = "ip"

[network.extractor]
component_id = "adarena.basic_flow_extractor"
""".strip(),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "config",
            "validate",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert (
        "Configuração válida"
        in result.stdout
    )
    assert (
        "Mode: observe"
        in result.stdout
    )


def test_cli_config_validate_rejeita_componente(
    tmp_path,
):
    config_path = (
        tmp_path
        / "invalid.toml"
    )

    config_path.write_text(
        """
mode = "observe"

[defender]
component_id = "adarena.nao_existe"
source = "defender.pth"

[protocol]
component_id = "adarena.network_observation"

[network]
preprocessor_source = "preprocessador"
dry_run = true

[network.capture]
component_id = "adarena.scapy_capture"

[network.extractor]
component_id = "adarena.basic_flow_extractor"
""".strip(),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "config",
            "validate",
            str(config_path),
        ],
    )

    assert result.exit_code == 2
    assert (
        "Configuração inválida"
        in result.stdout
    )



def test_cli_observe_executa_config(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    import adarena.cli.app as cli_app

    from adarena.core.config import (
        ExperimentMode,
    )

    config_path = (
        tmp_path
        / "observe.toml"
    )

    config_path.write_text(
        'mode = "observe"\n',
        encoding="utf-8",
    )

    fake_result = SimpleNamespace(
        run_id="run_test",
        run_dir=(
            tmp_path
            / "experiments"
            / "run_test"
        ),
        final_metrics={
            "captured_packets": 180,
            "reconstructed_flows": 15,
            "network_benign_count": 15,
            "network_attack_count": 0,
            "block_decisions": 0,
            "rules_applied": 0,
        },
    )

    observed = {}

    def fake_execute(
        path,
        *,
        expected_mode,
    ):
        observed["path"] = path
        observed["mode"] = expected_mode
        return fake_result

    monkeypatch.setattr(
        cli_app,
        "execute_config",
        fake_execute,
    )

    result = runner.invoke(
        app,
        [
            "observe",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert "Experimento concluído" in result.stdout
    assert "run_test" in result.stdout
    assert "180" in result.stdout
    assert "15" in result.stdout

    assert (
        observed["mode"]
        == ExperimentMode.OBSERVE
    )


def test_cli_observe_rejeita_mode_incorreto(
    tmp_path,
    monkeypatch,
):
    import adarena.cli.app as cli_app

    from adarena.application import (
        ModeMismatchError,
    )

    config_path = (
        tmp_path
        / "train.toml"
    )

    config_path.write_text(
        'mode = "train"\n',
        encoding="utf-8",
    )

    def fail_execute(
        path,
        *,
        expected_mode,
    ):
        raise ModeMismatchError(
            'o comando exige mode = "observe", '
            'mas o arquivo define mode = "train"'
        )

    monkeypatch.setattr(
        cli_app,
        "execute_config",
        fail_execute,
    )

    result = runner.invoke(
        app,
        [
            "observe",
            "-c",
            str(config_path),
        ],
    )

    assert result.exit_code == 2
    assert (
        "Não foi possível iniciar"
        in result.stdout
    )



def test_cli_observe_mostra_preflight_sem_traceback(
    tmp_path,
    monkeypatch,
):
    import adarena.cli.app as cli_app

    from adarena.application import (
        PreflightError,
    )

    config_path = (
        tmp_path
        / "observe.toml"
    )

    config_path.write_text(
        'mode = "observe"\n',
        encoding="utf-8",
    )

    def fail_execute(
        path,
        *,
        expected_mode,
    ):
        raise PreflightError(
            "checkpoint do Defender não encontrado"
        )

    monkeypatch.setattr(
        cli_app,
        "execute_config",
        fail_execute,
    )

    result = runner.invoke(
        app,
        [
            "observe",
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 2
    assert (
        "checkpoint do Defender não encontrado"
        in result.stdout
    )
    assert "Traceback" not in result.stdout
