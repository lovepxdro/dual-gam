from __future__ import annotations

from importlib.metadata import (
    PackageNotFoundError,
    version,
)

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from adarena.application import (
    ModeMismatchError,
    PreflightError,
    execute_config,
)
from adarena.builtin import (
    create_default_registry,
)
from adarena.config_io import (
    ConfigLoadError,
    load_experiment_config,
)
from adarena.core.components import (
    ComponentKind,
)
from adarena.core.config import (
    ExperimentMode,
)
from adarena.logging_config import (
    configure_console_logging,
)
from adarena.runs import (
    AmbiguousRunError,
    RunNotFoundError,
    artifact_counts,
    discover_runs,
    find_run,
    summarize_components,
)


app = typer.Typer(
    name="adarena",
    help=(
        "ADArena — ambiente experimental para "
        "defesa adaptativa."
    ),
    no_args_is_help=True,
    add_completion=False,
)

config_app = typer.Typer(
    help=(
        "Carrega e valida configurações "
        "experimentais TOML."
    ),
    no_args_is_help=True,
)

runs_app = typer.Typer(
    help=(
        "Inspeciona execuções e artefatos "
        "persistidos."
    ),
    invoke_without_command=True,
    no_args_is_help=False,
)

app.add_typer(
    config_app,
    name="config",
)

app.add_typer(
    runs_app,
    name="runs",
)

console = Console()


def _version() -> str:
    try:
        return version("dual-gam")
    except PackageNotFoundError:
        return "2.4.0"


def _parse_component_kind(
    value: str | None,
) -> ComponentKind | None:
    if value is None:
        return None

    normalized = value.strip().lower()

    try:
        return ComponentKind(
            normalized
        )
    except ValueError as exc:
        valid = ", ".join(
            kind.value
            for kind in ComponentKind
        )

        raise typer.BadParameter(
            (
                f"tipo de componente inválido: "
                f"{value}. Valores aceitos: "
                f"{valid}"
            ),
            param_hint="--kind",
        ) from exc


def _format_metric(
    value: Any,
) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"

    return str(value)


def _print_run_summary(
    result,
    *,
    mode: ExperimentMode,
) -> None:
    console.print()
    console.print(
        "[bold green]✓ Experimento concluído[/bold green]"
    )
    console.print(
        f"Mode: {mode.value}"
    )
    console.print(
        f"Run: {result.run_id}"
    )

    metrics = (
        result.final_metrics
        or {}
    )

    if (
        mode
        == ExperimentMode.OBSERVE
    ):
        preferred = (
            ("Pacotes", "captured_packets"),
            ("Fluxos", "reconstructed_flows"),
            ("Benignos", "network_benign_count"),
            ("Ataques", "network_attack_count"),
            ("BLOCK", "block_decisions"),
            ("Regras aplicadas", "rules_applied"),
        )

        table = Table(
            show_header=False,
            box=None,
            pad_edge=False,
        )

        table.add_column(
            "Métrica",
            style="bold",
        )
        table.add_column(
            "Valor",
            justify="right",
        )

        for label, key in preferred:
            if key in metrics:
                table.add_row(
                    label,
                    _format_metric(
                        metrics[key]
                    ),
                )

        console.print(table)

    else:
        scalar_metrics = [
            (key, value)
            for key, value in metrics.items()
            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            )
        ]

        if scalar_metrics:
            table = Table(
                title="Métricas finais",
                show_header=True,
            )

            table.add_column(
                "Métrica"
            )
            table.add_column(
                "Valor",
                justify="right",
            )

            for key, value in scalar_metrics:
                table.add_row(
                    key,
                    _format_metric(value),
                )

            console.print(table)

    console.print(
        f"Artefatos: {result.run_dir}"
    )


def _execute_cli_mode(
    config: Path,
    *,
    mode: ExperimentMode,
) -> None:
    try:
        result = execute_config(
            config,
            expected_mode=mode,
        )

    except (
        ConfigLoadError,
        ModeMismatchError,
        PreflightError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as exc:
        console.print(
            "[bold red]Não foi possível iniciar "
            "o experimento:[/bold red] "
            f"{exc}"
        )
        raise typer.Exit(
            code=2
        ) from exc

    _print_run_summary(
        result,
        mode=mode,
    )


def _print_runs(
    root: Path,
) -> None:
    records = discover_runs(
        root
    )

    if not records:
        console.print(
            "Nenhuma execução encontrada."
        )
        return

    table = Table(
        title="ADArena Runs",
        show_header=True,
    )

    table.add_column(
        "Run",
        no_wrap=True,
    )
    table.add_column(
        "Mode",
        no_wrap=True,
    )
    table.add_column(
        "Created",
        no_wrap=True,
    )
    table.add_column(
        "Seed",
        justify="right",
        no_wrap=True,
    )
    table.add_column(
        "Purpose",
    )
    table.add_column(
        "Path",
    )

    for record in records:
        table.add_row(
            record.run_id,
            record.mode,
            record.created_at or "-",
            (
                str(record.seed)
                if record.seed is not None
                else "-"
            ),
            record.purpose or "-",
            str(record.run_dir),
        )

    console.print(table)


@app.callback()
def root(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help=(
            "Exibe logs técnicos detalhados "
            "no terminal."
        ),
    ),
) -> None:
    configure_console_logging(
        verbose=verbose
    )


@app.command("version")
def version_command() -> None:
    console.print(
        f"ADArena {_version()}"
    )


@app.command()
def components(
    kind: str | None = typer.Option(
        None,
        "--kind",
        "-k",
        help=(
            "Filtra por tipo de componente, "
            "por exemplo: defender, capture "
            "ou experiment_protocol."
        ),
    ),
) -> None:
    registry = create_default_registry()

    component_kind = (
        _parse_component_kind(kind)
    )

    specs = registry.list(
        kind=component_kind
    )

    if not specs:
        console.print(
            "Nenhum componente registrado "
            "para o filtro informado."
        )
        return

    table = Table(
        title="ADArena Components",
        show_header=True,
        header_style="bold",
    )

    table.add_column(
        "Kind",
        no_wrap=True,
    )
    table.add_column(
        "Component ID",
        no_wrap=True,
    )
    table.add_column("Name")
    table.add_column(
        "Version",
        justify="right",
        no_wrap=True,
    )

    for spec in specs:
        table.add_row(
            spec.kind.value,
            spec.component_id,
            spec.name,
            spec.version,
        )

    console.print(table)


@config_app.command("validate")
def validate_config(
    path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Arquivo TOML do experimento.",
    ),
) -> None:
    try:
        config = load_experiment_config(
            path
        )

        registry = (
            create_default_registry()
        )

        config.validate(
            registry
        )

    except (
        ConfigLoadError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        console.print(
            "[bold red]Configuração inválida:[/bold red] "
            f"{exc}"
        )
        raise typer.Exit(
            code=2
        ) from exc

    console.print(
        "[bold green]✓ Configuração válida[/bold green]"
    )
    console.print(
        f"Mode: {config.mode.value}"
    )
    console.print(
        f"Defender: "
        f"{config.defender.component_id}"
    )

    if config.protocol is not None:
        console.print(
            f"Protocol: "
            f"{config.protocol.component_id}"
        )


@runs_app.callback(
    invoke_without_command=True
)
def runs(
    ctx: typer.Context,
    root: Path = typer.Option(
        Path("models"),
        "--root",
        help=(
            "Diretório raiz onde os runs "
            "serão procurados."
        ),
    ),
) -> None:
    """
    Lista execuções conhecidas.
    """

    ctx.ensure_object(
        dict
    )
    ctx.obj["runs_root"] = root

    if (
        ctx.invoked_subcommand
        is None
    ):
        _print_runs(
            root
        )


@runs_app.command("show")
def runs_show(
    ctx: typer.Context,
    run_id: str = typer.Argument(
        ...,
        help="ID da execução.",
    ),
) -> None:
    """
    Mostra detalhes de uma execução.
    """

    root = (
        ctx.obj.get(
            "runs_root",
            Path("models"),
        )
        if isinstance(
            ctx.obj,
            dict,
        )
        else Path("models")
    )

    try:
        record = find_run(
            run_id,
            root=root,
        )

    except (
        RunNotFoundError,
        AmbiguousRunError,
    ) as exc:
        console.print(
            "[bold red]Run inválido:[/bold red] "
            f"{exc}"
        )
        raise typer.Exit(
            code=2
        ) from exc

    console.print(
        f"[bold]{record.run_id}[/bold]"
    )
    console.print(
        f"Mode: {record.mode}"
    )
    console.print(
        f"Created: "
        f"{record.created_at or '-'}"
    )
    console.print(
        f"Seed: "
        f"{record.seed if record.seed is not None else '-'}"
    )
    console.print(
        f"Purpose: "
        f"{record.purpose or '-'}"
    )
    console.print(
        f"Path: {record.run_dir}"
    )

    components = (
        summarize_components(
            record
        )
    )

    if components:
        console.print()
        table = Table(
            title="Components",
            show_header=True,
        )

        table.add_column(
            "Role"
        )
        table.add_column(
            "Component ID"
        )

        for role, component_id in (
            components.items()
        ):
            table.add_row(
                role,
                component_id or "-",
            )

        console.print(table)

    counts = artifact_counts(
        record
    )

    console.print()
    artifact_table = Table(
        title="Artifacts",
        show_header=True,
    )

    artifact_table.add_column(
        "Directory"
    )
    artifact_table.add_column(
        "Files",
        justify="right",
    )

    for name, count in (
        counts.items()
    ):
        artifact_table.add_row(
            name,
            str(count),
        )

    console.print(
        artifact_table
    )


@app.command()
def train(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help=(
            "Arquivo TOML com mode = "
            '"train".'
        ),
    ),
) -> None:
    """
    Executa um experimento de treinamento.
    """

    _execute_cli_mode(
        config,
        mode=ExperimentMode.TRAIN,
    )


@app.command()
def simulate(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help=(
            "Arquivo TOML com mode = "
            '"simulate". A CLI aceita apenas dry-run.'
        ),
    ),
) -> None:
    """
    Executa a simulação experimental exclusivamente em dry-run.
    """

    _execute_cli_mode(
        config,
        mode=ExperimentMode.SIMULATE,
    )


@app.command()
def observe(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help=(
            "Arquivo TOML com mode = "
            '"observe".'
        ),
    ),
) -> None:
    """
    Executa observação passiva da rede.
    """

    _execute_cli_mode(
        config,
        mode=ExperimentMode.OBSERVE,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
