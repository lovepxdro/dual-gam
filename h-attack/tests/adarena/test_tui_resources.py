from __future__ import annotations

import json

from adarena.tui.resources import (
    attacker_checkpoint_options,
    dataset_options,
    defender_checkpoint_options,
    inspect_run_resources,
    training_run_options,
)


def _make_train_run(
    root,
):
    run_dir = (
        root
        / "train-output"
        / "experiments"
        / "run_20260922_120000_seed42"
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
            "run_20260922_120000_seed42"
        ),
        "created_at": (
            "2026-09-22T12:00:00"
        ),
        "mode": "train",
        "seed": 42,
        "metadata": {
            "purpose": "teste",
        },
        "components": {},
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
    ):
        (
            checkpoints
            / name
        ).write_bytes(
            b"test"
        )

    return run_dir


def test_dataset_options(
    tmp_path,
):
    data = (
        tmp_path
        / "data"
    )
    data.mkdir()

    (
        data
        / "dataset.parquet"
    ).write_bytes(
        b"test"
    )

    options = dataset_options(
        data
    )

    assert options == [
        (
            "dataset.parquet",
            str(
                data
                / "dataset.parquet"
            ),
        )
    ]


def test_training_run_options(
    tmp_path,
):
    run_dir = _make_train_run(
        tmp_path
    )

    options = training_run_options(
        tmp_path
    )

    assert len(options) == 1

    label, value = options[0]

    assert (
        "run_20260922_120000_seed42"
        in label
    )

    assert value == str(
        run_dir
    )


def test_inspect_run_resources(
    tmp_path,
):
    run_dir = _make_train_run(
        tmp_path
    )

    resources = (
        inspect_run_resources(
            run_dir
        )
    )

    assert (
        resources.preprocessor
        == run_dir
        / "preprocessador"
    )

    attacker = (
        attacker_checkpoint_options(
            resources
        )
    )

    defender = (
        defender_checkpoint_options(
            resources
        )
    )

    assert attacker[0][0].startswith(
        "Final"
    )

    assert attacker[1][0].startswith(
        "A1"
    )

    assert defender[0][0].startswith(
        "Final"
    )

    assert any(
        label.startswith(
            "D0"
        )
        for label, _
        in defender
    )
