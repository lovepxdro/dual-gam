from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adarena.runs import (
    RunRecord,
    discover_runs,
)


@dataclass(frozen=True, slots=True)
class RunResources:
    """
    Artefatos reutilizáveis encontrados em um run.

    A TUI usa esta estrutura apenas para apresentar opções.
    A validação real continua pertencendo à camada de aplicação.
    """

    record: RunRecord

    attacker_checkpoints: tuple[Path, ...]
    defender_checkpoints: tuple[Path, ...]

    preprocessor: Path | None


def discover_dataset_files(
    root: str | Path = "data",
) -> list[Path]:
    """
    Descobre arquivos disponíveis para seleção de Dataset.

    Não tenta decidir se o arquivo é semanticamente compatível com
    um Dataset adapter específico; essa responsabilidade continua
    com a validação/preflight da ADArena.
    """

    root_path = Path(root)

    if not root_path.is_dir():
        return []

    return sorted(
        (
            path
            for path in root_path.rglob("*")
            if (
                path.is_file()
                and not path.name.startswith(".")
            )
        ),
        key=lambda path: str(path).lower(),
    )


def dataset_options(
    root: str | Path = "data",
) -> list[tuple[str, str]]:
    """
    Converte datasets descobertos para opções de Select.

    label -> texto mostrado
    value -> caminho persistido em ExperimentConfig
    """

    paths = discover_dataset_files(
        root
    )

    return [
        (
            _relative_label(
                path,
                Path(root),
            ),
            str(path),
        )
        for path in paths
    ]


def discover_training_runs(
    root: str | Path = "models",
) -> list[RunRecord]:
    """
    Retorna apenas runs de TRAIN que possuem algum artefato
    potencialmente reutilizável.
    """

    records = discover_runs(
        root
    )

    result: list[RunRecord] = []

    for record in records:
        if record.mode != "train":
            continue

        checkpoint_dir = (
            record.run_dir
            / "checkpoints"
        )

        preprocessor_dir = (
            record.run_dir
            / "preprocessador"
        )

        if (
            checkpoint_dir.is_dir()
            or preprocessor_dir.is_dir()
        ):
            result.append(
                record
            )

    return result


def training_run_options(
    root: str | Path = "models",
) -> list[tuple[str, str]]:
    """
    Opções para seleção de um run de origem.

    O valor armazenado é o caminho do run, e não somente run_id,
    evitando ambiguidade caso existam IDs iguais abaixo de roots
    diferentes.
    """

    return [
        (
            _run_label(record),
            str(record.run_dir),
        )
        for record in discover_training_runs(
            root
        )
    ]


def inspect_run_resources(
    run_dir: str | Path,
) -> RunResources:
    """
    Descobre checkpoints e preprocessador de um run já selecionado.
    """

    path = Path(run_dir)

    record = _record_for_run_dir(
        path
    )

    checkpoint_dir = (
        path
        / "checkpoints"
    )

    checkpoints = (
        list(
            checkpoint_dir.glob(
                "*.pth"
            )
        )
        if checkpoint_dir.is_dir()
        else []
    )

    attacker = sorted(
        (
            item
            for item in checkpoints
            if _is_attacker_checkpoint(
                item
            )
        ),
        key=_attacker_sort_key,
    )

    defender = sorted(
        (
            item
            for item in checkpoints
            if _is_defender_checkpoint(
                item
            )
        ),
        key=_defender_sort_key,
    )

    preprocessor = (
        path
        / "preprocessador"
    )

    if not preprocessor.is_dir():
        preprocessor = None

    return RunResources(
        record=record,
        attacker_checkpoints=tuple(
            attacker
        ),
        defender_checkpoints=tuple(
            defender
        ),
        preprocessor=preprocessor,
    )


def attacker_checkpoint_options(
    resources: RunResources,
) -> list[tuple[str, str]]:
    return [
        (
            _attacker_label(
                path
            ),
            str(path),
        )
        for path
        in resources.attacker_checkpoints
    ]


def defender_checkpoint_options(
    resources: RunResources,
) -> list[tuple[str, str]]:
    return [
        (
            _defender_label(
                path
            ),
            str(path),
        )
        for path
        in resources.defender_checkpoints
    ]


def _record_for_run_dir(
    run_dir: Path,
) -> RunRecord:
    """
    Localiza o RunRecord correspondente ao diretório selecionado.

    Um run da ADArena possui a estrutura:

        <output_dir>/experiments/run_...

    Portanto, não assumimos que o output_dir se chama "models".
    Isso permite outputs customizados e também funciona nos testes.
    """

    run_dir = Path(
        run_dir
    )

    try:
        resolved = (
            run_dir
            .resolve()
        )
    except OSError as exc:
        raise ValueError(
            "Não foi possível resolver "
            f"o diretório do run: {run_dir}"
        ) from exc

    if (
        run_dir.parent.name
        != "experiments"
    ):
        raise ValueError(
            "O diretório selecionado não possui "
            "a estrutura esperada "
            "<output_dir>/experiments/run_*: "
            f"{run_dir}"
        )

    output_root = (
        run_dir
        .parent
        .parent
    )

    for record in discover_runs(
        output_root
    ):
        try:
            candidate = (
                record
                .run_dir
                .resolve()
            )
        except OSError:
            continue

        if candidate == resolved:
            return record

    raise ValueError(
        "O diretório selecionado não corresponde "
        "a um run reconhecido pela ADArena: "
        f"{run_dir}"
    )


def _is_attacker_checkpoint(
    path: Path,
) -> bool:
    name = path.name.lower()

    return (
        name.startswith(
            "atacante_"
        )
        or name.startswith(
            "attacker_"
        )
    )


def _is_defender_checkpoint(
    path: Path,
) -> bool:
    name = path.name.lower()

    return (
        name.startswith(
            "defensor_"
        )
        or name.startswith(
            "defender_"
        )
    )


def _attacker_sort_key(
    path: Path,
) -> tuple[int, int, str]:
    name = path.stem.lower()

    if name in {
        "atacante_final",
        "attacker_final",
    }:
        return (
            0,
            0,
            name,
        )

    round_number = _round_number(
        name
    )

    return (
        1,
        round_number,
        name,
    )


def _defender_sort_key(
    path: Path,
) -> tuple[int, int, str]:
    name = path.stem.lower()

    if name in {
        "defensor_adaptativo_final",
        "defensor_final",
        "defender_final",
    }:
        return (
            0,
            0,
            name,
        )

    round_number = _round_number(
        name
    )

    return (
        1,
        round_number,
        name,
    )


def _round_number(
    name: str,
) -> int:
    marker = "_rodada_"

    if marker not in name:
        return 10**9

    raw = name.rsplit(
        marker,
        1,
    )[1]

    try:
        return int(raw)
    except ValueError:
        return 10**9


def _attacker_label(
    path: Path,
) -> str:
    name = path.stem

    if name.lower() in {
        "atacante_final",
        "attacker_final",
    }:
        return (
            f"Final — {path.name}"
        )

    round_number = _round_number(
        name.lower()
    )

    if round_number < 10**9:
        return (
            f"A{round_number} — "
            f"{path.name}"
        )

    return path.name


def _defender_label(
    path: Path,
) -> str:
    name = path.stem

    if name.lower() in {
        "defensor_adaptativo_final",
        "defensor_final",
        "defender_final",
    }:
        return (
            f"Final — {path.name}"
        )

    round_number = _round_number(
        name.lower()
    )

    if round_number < 10**9:
        return (
            f"D{round_number} — "
            f"{path.name}"
        )

    return path.name


def _run_label(
    record: RunRecord,
) -> str:
    suffix = ""

    if record.purpose:
        suffix = (
            f" — {record.purpose}"
        )

    return (
        f"{record.run_id}"
        f"{suffix}"
    )


def _relative_label(
    path: Path,
    root: Path,
) -> str:
    try:
        return str(
            path.relative_to(
                root
            )
        )
    except ValueError:
        return path.name
