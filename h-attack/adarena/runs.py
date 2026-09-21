from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    run_dir: Path
    mode: str
    created_at: str
    seed: int | None
    purpose: str | None
    config: dict[str, Any]


class RunNotFoundError(LookupError):
    pass


class AmbiguousRunError(LookupError):
    pass


def discover_runs(
    root: str | Path = "models",
) -> list[RunRecord]:
    """
    Descobre runs da ADArena abaixo de ``root``.

    Um diretório é reconhecido como run quando:
    - o pai se chama ``experiments``;
    - existe ``config_execucao.json``.
    """

    root_path = Path(root)

    if not root_path.exists():
        return []

    records: list[RunRecord] = []

    for candidate in root_path.rglob("run_*"):
        if (
            not candidate.is_dir()
            or candidate.parent.name != "experiments"
        ):
            continue

        config_path = (
            candidate
            / "config_execucao.json"
        )

        if not config_path.is_file():
            continue

        try:
            config = _load_json(
                config_path
            )
        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            # Uma execução parcialmente escrita ou corrompida
            # não derruba a inspeção das demais.
            continue

        metadata = config.get(
            "metadata"
        )

        if not isinstance(
            metadata,
            dict,
        ):
            metadata = {}

        seed_value = config.get(
            "seed"
        )

        seed = (
            seed_value
            if isinstance(seed_value, int)
            else None
        )

        records.append(
            RunRecord(
                run_id=str(
                    config.get(
                        "run_id",
                        candidate.name,
                    )
                ),
                run_dir=candidate,
                mode=str(
                    config.get(
                        "mode",
                        "unknown",
                    )
                ),
                created_at=str(
                    config.get(
                        "created_at",
                        "",
                    )
                ),
                seed=seed,
                purpose=(
                    str(metadata["purpose"])
                    if "purpose" in metadata
                    else None
                ),
                config=config,
            )
        )

    records.sort(
        key=lambda item: (
            item.created_at,
            item.run_id,
        ),
        reverse=True,
    )

    return records


def find_run(
    run_id: str,
    *,
    root: str | Path = "models",
) -> RunRecord:
    matches = [
        record
        for record in discover_runs(
            root
        )
        if record.run_id == run_id
    ]

    if not matches:
        raise RunNotFoundError(
            f"run não encontrado: {run_id}"
        )

    if len(matches) > 1:
        paths = ", ".join(
            str(record.run_dir)
            for record in matches
        )

        raise AmbiguousRunError(
            (
                f"run_id ambíguo: {run_id}. "
                f"Encontrado em: {paths}"
            )
        )

    return matches[0]


def summarize_components(
    record: RunRecord,
) -> dict[str, str | None]:
    components = record.config.get(
        "components"
    )

    if not isinstance(
        components,
        dict,
    ):
        return {}

    result: dict[
        str,
        str | None,
    ] = {}

    for key, value in components.items():
        if value is None:
            result[str(key)] = None
        else:
            result[str(key)] = str(
                value
            )

    return result


def artifact_counts(
    record: RunRecord,
) -> dict[str, int]:
    counts: dict[str, int] = {}

    for name in (
        "checkpoints",
        "metrics",
        "plots",
        "logs",
    ):
        directory = (
            record.run_dir
            / name
        )

        if not directory.is_dir():
            counts[name] = 0
            continue

        counts[name] = sum(
            1
            for path in directory.iterdir()
            if path.is_file()
        )

    return counts


def _load_json(
    path: Path,
) -> dict[str, Any]:
    with open(
        path,
        encoding="utf-8",
    ) as file:
        payload = json.load(
            file
        )

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            f"{path} não contém um objeto JSON"
        )

    return payload
