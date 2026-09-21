from __future__ import annotations

import json

import pytest

from adarena.runs import (
    AmbiguousRunError,
    RunNotFoundError,
    artifact_counts,
    discover_runs,
    find_run,
)


def _make_run(
    root,
    *,
    output_name,
    run_id,
    mode,
    created_at,
    seed=42,
    purpose=None,
):
    run_dir = (
        root
        / output_name
        / "experiments"
        / run_id
    )

    run_dir.mkdir(
        parents=True
    )

    for name in (
        "checkpoints",
        "metrics",
        "plots",
        "logs",
    ):
        (
            run_dir
            / name
        ).mkdir()

    payload = {
        "run_id": run_id,
        "created_at": created_at,
        "mode": mode,
        "seed": seed,
        "metadata": {},
        "components": {
            "defender": (
                "adarena.binary_mlp_defender"
            ),
        },
    }

    if purpose is not None:
        payload["metadata"][
            "purpose"
        ] = purpose

    (
        run_dir
        / "config_execucao.json"
    ).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    return run_dir


def test_discover_runs_ordena_mais_recente_primeiro(
    tmp_path,
):
    _make_run(
        tmp_path,
        output_name="train-a",
        run_id="run_20260920_120000_seed42",
        mode="train",
        created_at="2026-09-20T12:00:00",
    )

    _make_run(
        tmp_path,
        output_name="observe-a",
        run_id="run_20260921_120000_seed42",
        mode="observe",
        created_at="2026-09-21T12:00:00",
        purpose="passive-observation",
    )

    records = discover_runs(
        tmp_path
    )

    assert [
        record.mode
        for record in records
    ] == [
        "observe",
        "train",
    ]

    assert (
        records[0].purpose
        == "passive-observation"
    )


def test_find_run(
    tmp_path,
):
    expected = _make_run(
        tmp_path,
        output_name="train-a",
        run_id="run_example",
        mode="train",
        created_at="2026-09-21T12:00:00",
    )

    record = find_run(
        "run_example",
        root=tmp_path,
    )

    assert (
        record.run_dir
        == expected
    )


def test_find_run_ausente(
    tmp_path,
):
    with pytest.raises(
        RunNotFoundError
    ):
        find_run(
            "run_missing",
            root=tmp_path,
        )


def test_find_run_ambiguo(
    tmp_path,
):
    for output_name in (
        "a",
        "b",
    ):
        _make_run(
            tmp_path,
            output_name=output_name,
            run_id="run_same",
            mode="train",
            created_at="2026-09-21T12:00:00",
        )

    with pytest.raises(
        AmbiguousRunError
    ):
        find_run(
            "run_same",
            root=tmp_path,
        )


def test_artifact_counts(
    tmp_path,
):
    run_dir = _make_run(
        tmp_path,
        output_name="train-a",
        run_id="run_example",
        mode="train",
        created_at="2026-09-21T12:00:00",
    )

    (
        run_dir
        / "metrics"
        / "summary.json"
    ).write_text(
        "{}",
        encoding="utf-8",
    )

    record = find_run(
        "run_example",
        root=tmp_path,
    )

    counts = artifact_counts(
        record
    )

    assert counts["metrics"] == 1
    assert counts["logs"] == 0
