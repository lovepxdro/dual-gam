from __future__ import annotations

import asyncio
import json

from textual.widgets import (
    DataTable,
    Input,
    Static,
    TabbedContent,
)

from adarena.tui.app import ADArenaTUI


def _make_run(root) -> None:
    run_dir = (
        root
        / "test-output"
        / "experiments"
        / "run_example"
    )

    run_dir.mkdir(parents=True)

    for name in (
        "checkpoints",
        "metrics",
        "plots",
        "logs",
    ):
        (run_dir / name).mkdir()

    (run_dir / "metrics" / "summary.json").write_text(
        "{}",
        encoding="utf-8",
    )

    payload = {
        "run_id": "run_example",
        "created_at": "2026-09-21T17:00:00",
        "mode": "train",
        "seed": 42,
        "metadata": {
            "purpose": "tui-test",
        },
        "components": {
            "defender": "adarena.binary_mlp_defender",
        },
    }

    (run_dir / "config_execucao.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _make_config(root) -> None:
    root.mkdir(parents=True, exist_ok=True)

    (root / "train.example.toml").write_text(
        """
mode = "train"
seed = 42
device = "cpu"
output_dir = "models"

[attack_dataset]
component_id = "adarena.cicids2017"
source = "data/example.parquet"

[attacker]
component_id = "adarena.perturbation_attacker"

[defender]
component_id = "adarena.binary_mlp_defender"

[protocol]
component_id = "adarena.adversarial_training"
""".strip(),
        encoding="utf-8",
    )


def test_tui_monta_dashboard_runs_componentes(tmp_path):
    _make_run(tmp_path)

    async def scenario():
        app = ADArenaTUI(
            models_root=tmp_path,
            configs_root=tmp_path / "configs",
        )

        async with app.run_test():
            assert app.query_one("#tabs", TabbedContent)
            assert app.query_one("#config-input", Input)

            assert (
                app.query_one("#runs-table", DataTable).row_count
                == 1
            )

            assert (
                app.query_one(
                    "#components-table",
                    DataTable,
                ).row_count
                > 0
            )

            overview = app.query_one("#overview-runs", Static)

            assert "Total: 1" in str(overview.render())

    asyncio.run(scenario())


def test_tui_atalhos_trocam_abas(tmp_path):
    async def scenario():
        app = ADArenaTUI(
            models_root=tmp_path,
            configs_root=tmp_path / "configs",
        )

        async with app.run_test() as pilot:
            tabs = app.query_one("#tabs", TabbedContent)

            await pilot.press("3")
            assert tabs.active == "runs"

            await pilot.press("4")
            assert tabs.active == "components"

    asyncio.run(scenario())


def test_tui_lista_configs_e_seleciona(
    tmp_path,
    monkeypatch,
):
    configs_root = tmp_path / "configs"
    _make_config(configs_root)

    class FakeMode:
        value = "train"

    class FakeSelection:
        component_id = "adarena.binary_mlp_defender"

    class FakeProtocol:
        component_id = "adarena.adversarial_training"

    class FakeConfig:
        mode = FakeMode()
        defender = FakeSelection()
        protocol = FakeProtocol()

    monkeypatch.setattr(
        "adarena.tui.app.validate_config_file",
        lambda path: FakeConfig(),
    )

    async def scenario():
        app = ADArenaTUI(
            models_root=tmp_path,
            configs_root=configs_root,
        )

        async with app.run_test():
            table = app.query_one("#configs-table", DataTable)

            assert table.row_count == 1

            app._select_config_row(0)

            input_widget = app.query_one("#config-input", Input)

            assert input_widget.value == str(
                configs_root / "train.example.toml"
            )

            status = app.query_one("#status", Static)

            assert "Configuração válida" in str(
                status.render()
            )

    asyncio.run(scenario())


def test_tui_mostra_detalhes_do_run(tmp_path):
    _make_run(tmp_path)

    async def scenario():
        app = ADArenaTUI(
            models_root=tmp_path,
            configs_root=tmp_path / "configs",
        )

        async with app.run_test():
            app._show_run_details(0)

            details = app.query_one("#run-details", Static)
            rendered = str(details.render())

            assert "run_example" in rendered
            assert "binary_mlp_defender" in rendered
            assert "metrics: 1" in rendered

    asyncio.run(scenario())
