from __future__ import annotations

import asyncio
import json

from types import SimpleNamespace

from textual.widgets import (
    Input,
    Select,
)

from adarena.core.config import (
    ExperimentMode,
)
from adarena.tui.app import (
    ADArenaTUI,
)


def _make_train_run(
    root,
):
    run_dir = (
        root
        / "train-output"
        / "experiments"
        / "run_20260922_122111_seed42"
    )

    checkpoints = (
        run_dir
        / "checkpoints"
    )

    checkpoints.mkdir(
        parents=True
    )

    (
        run_dir
        / "preprocessador"
    ).mkdir()

    payload = {
        "run_id": (
            "run_20260922_122111_seed42"
        ),
        "created_at": (
            "2026-09-22T12:21:11"
        ),
        "mode": "train",
        "seed": 42,
        "metadata": {
            "purpose": (
                "tui-polish-test"
            ),
        },
        "components": {
            "attacker": (
                "adarena.perturbation_attacker"
            ),
            "defender": (
                "adarena.binary_mlp_defender"
            ),
        },
    }

    (
        run_dir
        / "config_execucao.json"
    ).write_text(
        json.dumps(
            payload
        ),
        encoding="utf-8",
    )

    for name in (
        "atacante_final.pth",
        "atacante_rodada_01.pth",
        "atacante_rodada_02.pth",
        "defensor_adaptativo_final.pth",
        "defensor_rodada_00.pth",
        "defensor_rodada_01.pth",
        "defensor_rodada_02.pth",
    ):
        (
            checkpoints
            / name
        ).write_bytes(
            b"test"
        )

    return run_dir


def test_tui_run_atualiza_checkpoints_e_preprocessador(
    tmp_path,
):
    run_dir = _make_train_run(
        tmp_path
    )

    async def scenario():
        app = ADArenaTUI(
            models_root=tmp_path,
            configs_root=(
                tmp_path
                / "configs"
            ),
        )

        async with app.run_test():
            mode = app.query_one(
                "#builder-mode",
                Select,
            )

            mode.value = (
                ExperimentMode
                .SIMULATE
                .value
            )

            app._sync_source_run()

            source_run = (
                app.query_one(
                    "#builder-source-run",
                    Select,
                )
            )

            attacker = (
                app.query_one(
                    "#builder-attacker-source",
                    Select,
                )
            )

            defender = (
                app.query_one(
                    "#builder-defender-source",
                    Select,
                )
            )

            preprocessor = (
                app.query_one(
                    "#builder-preprocessor-source",
                    Input,
                )
            )

            assert (
                str(source_run.value)
                == str(run_dir)
            )

            assert attacker.disabled is False
            assert defender.disabled is False

            assert (
                str(attacker.value)
                == str(
                    run_dir
                    / "checkpoints"
                    / "atacante_final.pth"
                )
            )

            assert (
                str(defender.value)
                == str(
                    run_dir
                    / "checkpoints"
                    / "defensor_adaptativo_final.pth"
                )
            )

            assert (
                preprocessor.value
                == str(
                    run_dir
                    / "preprocessador"
                )
            )

    asyncio.run(
        scenario()
    )


def test_tui_resume_train_com_evolucao_adversarial(
    tmp_path,
):
    app = ADArenaTUI(
        models_root=tmp_path
    )

    result = SimpleNamespace(
        run_id="run_train",
        run_dir=tmp_path,
        history={
            "taxa_evasao_pre_adaptacao": [
                0.7534,
                0.4422,
                0.3452,
            ],
            "taxa_evasao_pos_adaptacao": [
                0.0020,
                0.0004,
                0.0004,
            ],
        },
        final_metrics={
            "accuracy": 0.991,
            "precision": 0.990,
            "recall": 0.989,
            "f1": 0.9895,
            "fpr": 0.004,
            "fnr": 0.011,
            "roc_auc": 0.998,
        },
    )

    text = app._format_result_summary(
        result,
        detailed=False,
    )

    assert (
        "Evolução adversarial"
        in text
    )

    assert (
        "A1×D0 75.34%"
        in text
    )

    assert (
        "A1×D1 0.20%"
        in text
    )

    assert (
        "A2×D1 44.22%"
        in text
    )

    assert (
        "A2×D2 0.04%"
        in text
    )

    assert (
        "Defender final"
        in text
    )

    assert (
        "ROC-AUC"
        in text
    )


def test_tui_resume_train_tolera_historico_incompleto(
    tmp_path,
):
    app = ADArenaTUI(
        models_root=tmp_path
    )

    result = SimpleNamespace(
        run_id="run_train_partial",
        run_dir=tmp_path,
        history={
            "taxa_evasao_pre_adaptacao": [
                0.75,
                0.40,
            ],
            "taxa_evasao_pos_adaptacao": [
                0.01,
            ],
        },
        final_metrics={},
    )

    text = app._format_result_summary(
        result,
        detailed=False,
    )

    assert (
        "A1×D0 75.00%"
        in text
    )

    assert (
        "A1×D1 1.00%"
        in text
    )

    assert (
        "A2×D1 40.00%"
        in text
    )

    assert (
        "A2×D2 -"
        in text
    )


def test_tui_resume_simulate_mostra_confronto_e_funil(
    tmp_path,
):
    run_dir = (
        tmp_path
        / "experiments"
        / "run_simulate"
    )

    run_dir.mkdir(
        parents=True
    )

    snapshot = {
        "component_config": {
            "attacker": {
                "component_id": (
                    "adarena.perturbation_attacker"
                ),
                "source": (
                    "models/source/checkpoints/"
                    "atacante_rodada_01.pth"
                ),
            },
            "defender": {
                "component_id": (
                    "adarena.binary_mlp_defender"
                ),
                "source": (
                    "models/source/checkpoints/"
                    "defensor_rodada_00.pth"
                ),
            },
        },
    }

    (
        run_dir
        / "config_execucao.json"
    ).write_text(
        json.dumps(
            snapshot
        ),
        encoding="utf-8",
    )

    app = ADArenaTUI(
        models_root=tmp_path
    )

    result = SimpleNamespace(
        run_id="run_simulate",
        run_dir=run_dir,
        history={},
        final_metrics={
            "samples_selected": 200,
            "mathematical_evasions": 150,
            "mathematical_evasion_rate": 0.75,
            "valid_renderings": 21,
            "valid_rendering_rate": 0.105,
            "valid_rendering_rate_given_evasion": 0.14,
            "dry_run_executions": 21,
            "classification_threshold": 0.5,
            "epsilon": 0.3,
            "dry_run": True,
        },
    )

    text = app._format_result_summary(
        result,
        detailed=False,
    )

    assert (
        "Cenário de simulação"
        in text
    )

    assert (
        "A1 × D0"
        in text
    )

    assert (
        "200 amostras selecionadas"
        in text
    )

    assert (
        "75.00%"
        in text
    )

    assert (
        "150 evasões matemáticas"
        in text
    )

    assert (
        "14.00% das evasões"
        in text
    )

    assert (
        "21 traduções válidas"
        in text
    )

    assert (
        "21 execuções dry-run"
        in text
    )
