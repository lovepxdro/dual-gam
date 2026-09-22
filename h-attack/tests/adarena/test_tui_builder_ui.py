from __future__ import annotations

from pathlib import Path

import asyncio

from textual.widgets import (
    Input,
    Select,
    Static,
)

from adarena.core.config import (
    ExperimentMode,
)
from adarena.tui.app import (
    ADArenaTUI,
)


def test_tui_builder_monta_train(
    tmp_path,
):
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

            assert (
                str(mode.value)
                == ExperimentMode.TRAIN.value
            )

            config, values = (
                app._build_from_ui()
            )

            assert (
                config.mode
                == ExperimentMode.TRAIN
            )
            assert (
                config.attack_dataset
                is not None
            )
            assert (
                config.attacker
                is not None
            )
            assert config.network is None
            assert values.seed == 42

    asyncio.run(
        scenario()
    )


def test_tui_builder_observe_esconde_attack(
    tmp_path,
):
    async def scenario():
        app = ADArenaTUI(
            models_root=tmp_path,
            configs_root=(
                tmp_path
                / "configs"
            ),
        )

        async with app.run_test():
            app._apply_builder_mode(
                ExperimentMode.OBSERVE
            )

            assert (
                app.query_one(
                    "#builder-attack-group"
                ).display
                is False
            )

            assert (
                app.query_one(
                    "#builder-training-group"
                ).display
                is False
            )

            assert (
                app.query_one(
                    "#builder-network-group"
                ).display
                is True
            )

    asyncio.run(
        scenario()
    )


def test_tui_builder_valida_sem_executar(
    tmp_path,
    monkeypatch,
):
    called = {
        "value": False,
    }

    def fake_validate(
        config,
    ):
        called["value"] = True
        return config

    monkeypatch.setattr(
        "adarena.tui.app."
        "validate_experiment_config",
        fake_validate,
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
            app._validate_builder()

            assert called["value"] is True

            status = app.query_one(
                "#builder-status",
                Static,
            )

            assert (
                "Experimento válido"
                in str(
                    status.render()
                )
            )

    asyncio.run(
        scenario()
    )


def test_tui_builder_detailed_fica_no_metadata(
    tmp_path,
):
    async def scenario():
        app = ADArenaTUI(
            models_root=tmp_path,
            configs_root=(
                tmp_path
                / "configs"
            ),
        )

        async with app.run_test():
            log_select = app.query_one(
                "#builder-log-level",
                Select,
            )

            log_select.value = "detailed"

            config, _ = (
                app._build_from_ui()
            )

            assert (
                config.metadata[
                    "ui_log_level"
                ]
                == "detailed"
            )

    asyncio.run(
        scenario()
    )


def test_tui_builder_numeric_error_e_amigavel(
    tmp_path,
):
    async def scenario():
        app = ADArenaTUI(
            models_root=tmp_path,
            configs_root=(
                tmp_path
                / "configs"
            ),
        )

        async with app.run_test():
            app.query_one(
                "#builder-seed",
                Input,
            ).value = "abc"

            app._validate_builder()

            status = app.query_one(
                "#builder-status",
                Static,
            )

            assert (
                "seed deve ser inteiro"
                in str(
                    status.render()
                )
            )

    asyncio.run(
        scenario()
    )

def test_tui_builder_salva_toml(
    tmp_path,
    monkeypatch,
):
    saved = {
        "config": None,
        "path": None,
    }

    def fake_save(
        config,
        path,
        *,
        overwrite=False,
    ):
        saved["config"] = config
        saved["path"] = path
        return Path(path)

    monkeypatch.setattr(
        "adarena.tui.app.save_experiment_config",
        fake_save,
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
            destination = (
                tmp_path
                / "configs"
                / "saved.local.toml"
            )

            app.query_one(
                "#builder-save-path",
                Input,
            ).value = str(
                destination
            )

            app._save_builder_config()

            assert (
                saved["config"]
                is not None
            )
            assert (
                saved["path"]
                == str(destination)
            )

            status = app.query_one(
                "#builder-status",
                Static,
            )

            assert (
                "Configuração salva"
                in str(
                    status.render()
                )
            )

    asyncio.run(
        scenario()
    )
