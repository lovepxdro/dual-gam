from __future__ import annotations

import json

from pathlib import Path
from typing import Any

from textual import work
from textual.app import (
    App,
    ComposeResult,
)
from textual.binding import Binding
from textual.containers import (
    Horizontal,
    Vertical,
    VerticalScroll,
)
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from adarena.application import (
    execute_config,
    execute_experiment_config,
    validate_config_file,
    validate_experiment_config,
)
from adarena.builtin import (
    create_default_registry,
)
from adarena.config_export import (
    save_experiment_config,
)
from adarena.core.components import (
    ComponentKind,
)
from adarena.core.config import (
    ExperimentMode,
)
from adarena.runs import (
    RunRecord,
    artifact_counts,
    discover_runs,
    summarize_components,
)
from adarena.tui.builder import (
    BuilderValues,
    build_experiment_config,
    component_options,
    preferred_protocol_id,
)
from adarena.tui.resources import (
    attacker_checkpoint_options,
    dataset_options as dataset_file_options,
    defender_checkpoint_options,
    inspect_run_resources,
    training_run_options,
)


class ADArenaTUI(App):
    """
    Interface terminal da ADArena.

    A TUI é uma camada de apresentação. Configuração, validação
    e execução continuam delegadas para application.py e para o
    ExperimentRunner.
    """

    TITLE = "ADArena"
    SUB_TITLE = "Adaptive Defense Arena"

    CSS = """
    Screen {
        background: $surface;
    }

    #dashboard-grid {
        height: auto;
        margin: 1 2;
    }

    .card {
        border: round $primary;
        padding: 1 2;
        margin: 0 1 1 0;
        min-height: 7;
        width: 1fr;
    }

    #builder-scroll,
    #toml-layout,
    #runs-layout {
        height: 1fr;
        margin: 1 2;
    }

    .builder-section {
        border: round $primary;
        padding: 1 2;
        margin-bottom: 1;
        height: auto;
    }

    .field-label {
        margin-top: 1;
        color: $text-muted;
    }

    Select,
    Input {
        margin-bottom: 1;
    }

    #builder-actions,
    #config-actions {
        height: auto;
        margin: 1 0;
    }

    #builder-actions Button,
    #config-actions Button {
        margin-right: 1;
    }

    #builder-status,
    #status {
        border: round $accent;
        padding: 1 2;
        min-height: 8;
        margin-bottom: 1;
    }

    #config-browser {
        width: 2fr;
        margin-right: 1;
    }

    #config-panel {
        width: 3fr;
    }

    #runs-table {
        width: 3fr;
        margin-right: 1;
    }

    #run-details {
        width: 2fr;
        border: round $primary;
        padding: 1 2;
        height: 1fr;
        overflow-y: auto;
    }

    #components-table {
        margin: 1 2;
        height: 1fr;
    }

    .hint {
        color: $text-muted;
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding(
            "q",
            "quit",
            "Sair",
        ),
        Binding(
            "r",
            "refresh",
            "Atualizar",
        ),
        Binding(
            "1",
            "tab_dashboard",
            "Dashboard",
            show=False,
        ),
        Binding(
            "2",
            "tab_execute",
            "Executar",
            show=False,
        ),
        Binding(
            "3",
            "tab_runs",
            "Runs",
            show=False,
        ),
        Binding(
            "4",
            "tab_components",
            "Componentes",
            show=False,
        ),
    ]

    def __init__(
        self,
        *,
        models_root: str | Path = "models",
        configs_root: str | Path = "configs",
    ) -> None:
        super().__init__()

        self.models_root = Path(
            models_root
        )
        self.configs_root = Path(
            configs_root
        )

        self.registry = (
            create_default_registry()
        )

        self._run_records: list[
            RunRecord
        ] = []

        self._config_paths: list[
            Path
        ] = []

    # ------------------------------------------------------------------
    # Composição
    # ------------------------------------------------------------------

    def compose(
        self,
    ) -> ComposeResult:
        yield Header()

        with TabbedContent(
            initial="dashboard",
            id="tabs",
        ):
            with TabPane(
                "Dashboard",
                id="dashboard",
            ):
                with Horizontal(
                    id="dashboard-grid"
                ):
                    yield Static(
                        id="overview-runs",
                        classes="card",
                    )
                    yield Static(
                        id="overview-components",
                        classes="card",
                    )
                    yield Static(
                        id="overview-safety",
                        classes="card",
                    )

            with TabPane(
                "Executar",
                id="execute",
            ):
                with TabbedContent(
                    initial="builder",
                    id="execute-tabs",
                ):
                    with TabPane(
                        "Novo experimento",
                        id="builder",
                    ):
                        yield from (
                            self
                            ._compose_builder()
                        )

                    with TabPane(
                        "Carregar TOML",
                        id="toml",
                    ):
                        yield from (
                            self
                            ._compose_toml_runner()
                        )

            with TabPane(
                "Runs",
                id="runs",
            ):
                with Horizontal(
                    id="runs-layout"
                ):
                    yield DataTable(
                        id="runs-table",
                        cursor_type="row",
                        zebra_stripes=True,
                    )

                    yield Static(
                        (
                            "[b]Detalhes do run[/b]\n\n"
                            "Selecione uma linha com Enter."
                        ),
                        id="run-details",
                    )

            with TabPane(
                "Componentes",
                id="components",
            ):
                yield DataTable(
                    id="components-table",
                    cursor_type="row",
                    zebra_stripes=True,
                )

        yield Footer()

    def _compose_builder(
        self,
    ) -> ComposeResult:
        dataset_options = (
            component_options(
                self.registry,
                ComponentKind.DATASET,
            )
        )
        attacker_options = (
            component_options(
                self.registry,
                ComponentKind.ATTACKER,
            )
        )
        defender_options = (
            component_options(
                self.registry,
                ComponentKind.DEFENDER,
            )
        )
        protocol_options = (
            component_options(
                self.registry,
                ComponentKind
                .EXPERIMENT_PROTOCOL,
            )
        )
        renderer_options = (
            component_options(
                self.registry,
                ComponentKind.RENDERER,
            )
        )
        backend_options = (
            component_options(
                self.registry,
                ComponentKind
                .NETWORK_BACKEND,
            )
        )
        capture_options = (
            component_options(
                self.registry,
                ComponentKind.CAPTURE,
            )
        )
        extractor_options = (
            component_options(
                self.registry,
                ComponentKind.EXTRACTOR,
            )
        )

        dataset_files = (
            dataset_file_options(
                "data"
            )
        )

        source_runs = (
            training_run_options(
                self.models_root
            )
        )

        with VerticalScroll(
            id="builder-scroll"
        ):
            with Vertical(
                classes="builder-section"
            ):
                yield Static(
                    "[b]1. Experimento[/b]"
                )

                yield Static(
                    "Modo",
                    classes="field-label",
                )
                yield Select(
                    [
                        (
                            "Train",
                            ExperimentMode
                            .TRAIN
                            .value,
                        ),
                        (
                            "Simulate — dry-run",
                            ExperimentMode
                            .SIMULATE
                            .value,
                        ),
                        (
                            "Observe — passivo",
                            ExperimentMode
                            .OBSERVE
                            .value,
                        ),
                    ],
                    value=(
                        ExperimentMode
                        .TRAIN
                        .value
                    ),
                    allow_blank=False,
                    id="builder-mode",
                )

                yield Static(
                    "Defender",
                    classes="field-label",
                )
                yield self._select(
                    defender_options,
                    id="builder-defender",
                )

                yield Static(
                    "Protocol",
                    classes="field-label",
                )
                yield self._select(
                    protocol_options,
                    id="builder-protocol",
                    initial=(
                        preferred_protocol_id(
                            self.registry,
                            ExperimentMode.TRAIN,
                        )
                    ),
                )

            with Vertical(
                id="builder-attack-group",
                classes="builder-section",
            ):
                yield Static(
                    "[b]2. Dados e Attacker[/b]"
                )

                yield Static(
                    "Dataset",
                    classes="field-label",
                )
                yield self._select(
                    dataset_options,
                    id="builder-dataset",
                )

                yield Static(
                    "Arquivo do dataset",
                    classes="field-label",
                )
                yield self._resource_select(
                    dataset_files,
                    id="builder-dataset-source",
                    empty_label=(
                        "Nenhum arquivo encontrado em data/"
                    ),
                )

                yield Static(
                    "Attacker",
                    classes="field-label",
                )
                yield self._select(
                    attacker_options,
                    id="builder-attacker",
                )

            with Vertical(
                classes="builder-section"
            ):
                yield Static(
                    "[b]3. Configuração geral[/b]"
                )

                yield Static(
                    "Seed",
                    classes="field-label",
                )
                yield Input(
                    value="42",
                    id="builder-seed",
                )

                yield Static(
                    "Device",
                    classes="field-label",
                )
                yield Select(
                    [
                        ("CPU", "cpu"),
                        ("CUDA", "cuda"),
                    ],
                    value="cpu",
                    allow_blank=False,
                    id="builder-device",
                )

                yield Static(
                    "Output directory",
                    classes="field-label",
                )
                yield Input(
                    value="models",
                    id="builder-output-dir",
                )

                yield Static(
                    "Classification threshold",
                    classes="field-label",
                )
                yield Input(
                    value="0.5",
                    id="builder-threshold",
                )

                yield Static(
                    "Saída da TUI",
                    classes="field-label",
                )
                yield Select(
                    [
                        (
                            "Normal — resumo",
                            "normal",
                        ),
                        (
                            "Detalhada — mais métricas",
                            "detailed",
                        ),
                    ],
                    value="normal",
                    allow_blank=False,
                    id="builder-log-level",
                )

                yield Static(
                    (
                        "O arquivo de log do run continua "
                        "completo nos dois modos."
                    ),
                    classes="hint",
                )

            with Vertical(
                id="builder-training-group",
                classes="builder-section",
            ):
                yield Static(
                    "[b]4. Treinamento[/b]"
                )

                yield Static(
                    "Test size",
                    classes="field-label",
                )
                yield Input(
                    value="0.2",
                    id="builder-test-size",
                )

                yield Static(
                    "Validation size",
                    classes="field-label",
                )
                yield Input(
                    value="0.1",
                    id="builder-validation-size",
                )

                yield Static(
                    "Noise dim",
                    classes="field-label",
                )
                yield Input(
                    value="32",
                    id="builder-noise-dim",
                )

                yield Static(
                    "Learning rate — Defender",
                    classes="field-label",
                )
                yield Input(
                    value="0.001",
                    id="builder-lr-defender",
                )

                yield Static(
                    "Learning rate — Attacker",
                    classes="field-label",
                )
                yield Input(
                    value="0.0002",
                    id="builder-lr-attacker",
                )

                yield Static(
                    "Epsilon",
                    classes="field-label",
                )
                yield Input(
                    value="0.3",
                    id="builder-epsilon",
                )

                yield Static(
                    "Epochs de pré-treino",
                    classes="field-label",
                )
                yield Input(
                    value="5",
                    id="builder-epochs-pretrain",
                )

                yield Static(
                    "Epochs por rodada",
                    classes="field-label",
                )
                yield Input(
                    value="3",
                    id="builder-epochs-round",
                )

                yield Static(
                    "Número de rodadas",
                    classes="field-label",
                )
                yield Input(
                    value="20",
                    id="builder-rounds",
                )

                yield Static(
                    "Amostras por rodada",
                    classes="field-label",
                )
                yield Input(
                    value="5000",
                    id="builder-samples-round",
                )

                yield Static(
                    "Amostras de avaliação adversarial",
                    classes="field-label",
                )
                yield Input(
                    value="5000",
                    id="builder-eval-samples",
                )

                yield Static(
                    "Batch size",
                    classes="field-label",
                )
                yield Input(
                    value="512",
                    id="builder-batch-size",
                )

            with Vertical(
                id="builder-network-group",
                classes="builder-section",
            ):
                yield Static(
                    "[b]4. Rede e inferência[/b]"
                )

                yield Static(
                    "Run de origem",
                    classes="field-label",
                )
                yield self._resource_select(
                    source_runs,
                    id="builder-source-run",
                    empty_label=(
                        "Nenhum TRAIN encontrado"
                    ),
                )

                yield Static(
                    (
                        "O run fornece checkpoints e "
                        "preprocessador para a execução."
                    ),
                    classes="hint",
                )

                yield Static(
                    "Checkpoint do Defender",
                    classes="field-label",
                )
                yield self._resource_select(
                    [],
                    id="builder-defender-source",
                    empty_label=(
                        "Selecione um run"
                    ),
                )

                yield Static(
                    "Preprocessador persistido",
                    classes="field-label",
                )
                yield Input(
                    value="",
                    placeholder=(
                        "preenchido automaticamente"
                    ),
                    disabled=True,
                    id="builder-preprocessor-source",
                )

                yield Static(
                    "Capture",
                    classes="field-label",
                )
                yield self._select(
                    capture_options,
                    id="builder-capture",
                )

                yield Static(
                    "Extractor",
                    classes="field-label",
                )
                yield self._select(
                    extractor_options,
                    id="builder-extractor",
                )

                yield Static(
                    "Interface de captura (opcional)",
                    classes="field-label",
                )
                yield Input(
                    placeholder="vethXXXXXXXX",
                    id="builder-capture-iface",
                )

                yield Static(
                    "Capture duration (s)",
                    classes="field-label",
                )
                yield Input(
                    value="5.0",
                    id="builder-capture-duration",
                )

                yield Static(
                    "Packet limit (vazio = sem limite explícito)",
                    classes="field-label",
                )
                yield Input(
                    value="",
                    id="builder-packet-limit",
                )

            with Vertical(
                id="builder-simulate-group",
                classes="builder-section",
            ):
                yield Static(
                    "[b]5. Simulação dry-run[/b]"
                )

                yield Static(
                    "Checkpoint do Attacker",
                    classes="field-label",
                )
                yield self._resource_select(
                    [],
                    id="builder-attacker-source",
                    empty_label=(
                        "Selecione um run"
                    ),
                )

                yield Static(
                    "Renderer",
                    classes="field-label",
                )
                yield self._select(
                    renderer_options,
                    id="builder-renderer",
                )

                yield Static(
                    "NetworkBackend",
                    classes="field-label",
                )
                yield self._select(
                    backend_options,
                    id="builder-backend",
                )

                yield Static(
                    "Alvo do laboratório",
                    classes="field-label",
                )
                yield Input(
                    value="172.20.0.10",
                    id="builder-target-ip",
                )

                yield Static(
                    "Porta do alvo",
                    classes="field-label",
                )
                yield Input(
                    value="80",
                    id="builder-target-port",
                )

                yield Static(
                    "Amostras para simulação",
                    classes="field-label",
                )
                yield Input(
                    value="20",
                    id="builder-sample-count",
                )

                yield Static(
                    (
                        "SIMULATE permanece fixo em "
                        "dry-run na interface pública."
                    ),
                    classes="hint",
                )

            with Vertical(
                classes="builder-section"
            ):
                yield Static(
                    "[b]Salvar configuração[/b]"
                )

                yield Static(
                    "Arquivo TOML",
                    classes="field-label",
                )
                yield Input(
                    value=(
                        "configs/"
                        "experiment.local.toml"
                    ),
                    id="builder-save-path",
                )

                yield Static(
                    (
                        "O arquivo salvo pode ser "
                        "reutilizado pela CLI ou pela TUI."
                    ),
                    classes="hint",
                )

            with Horizontal(
                id="builder-actions"
            ):
                yield Button(
                    "Validar",
                    id="builder-validate",
                    variant="primary",
                )
                yield Button(
                    "Salvar TOML",
                    id="builder-save",
                )
                yield Button(
                    "Executar",
                    id="builder-execute",
                    variant="success",
                )

            yield Static(
                (
                    "Monte o experimento pelos campos "
                    "acima e valide antes de executar."
                ),
                id="builder-status",
            )

    def _compose_toml_runner(
        self,
    ) -> ComposeResult:
        with Horizontal(
            id="toml-layout"
        ):
            with Vertical(
                id="config-browser"
            ):
                yield Static(
                    "[b]Configurações encontradas[/b]"
                )

                yield DataTable(
                    id="configs-table",
                    cursor_type="row",
                    zebra_stripes=True,
                )

                yield Static(
                    (
                        "Selecione uma linha com Enter "
                        "para carregar o caminho."
                    ),
                    classes="hint",
                )

            with Vertical(
                id="config-panel"
            ):
                yield Static(
                    "[b]Configuração selecionada[/b]"
                )

                yield Input(
                    placeholder=(
                        "configs/train.example.toml"
                    ),
                    id="config-input",
                )

                with Horizontal(
                    id="config-actions"
                ):
                    yield Button(
                        "Validar",
                        id="validate-config",
                        variant="primary",
                    )
                    yield Button(
                        "Executar",
                        id="execute-config",
                        variant="success",
                    )

                yield Static(
                    (
                        "Selecione ou informe um TOML. "
                        "Este caminho continua disponível "
                        "para reprodutibilidade e automação."
                    ),
                    id="status",
                )

    def _select(
        self,
        options: list[
            tuple[str, str]
        ],
        *,
        id: str,
        initial: str | None = None,
    ) -> Select:
        if not options:
            return Select(
                [("Nenhum componente", "")],
                value="",
                allow_blank=False,
                disabled=True,
                id=id,
            )

        valid_values = {
            value
            for _, value in options
        }

        value = (
            initial
            if initial in valid_values
            else options[0][1]
        )

        return Select(
            options,
            value=value,
            allow_blank=False,
            id=id,
        )

    def _resource_select(
        self,
        options: list[
            tuple[str, str]
        ],
        *,
        id: str,
        empty_label: str,
    ) -> Select:
        """
        Select usado para recursos descobertos no filesystem.

        Diferente de _select(), estes valores não são componentes
        do Registry: são datasets, runs ou checkpoints.
        """

        if not options:
            return Select(
                [
                    (
                        empty_label,
                        "",
                    )
                ],
                value="",
                allow_blank=False,
                disabled=True,
                id=id,
            )

        return Select(
            options,
            value=options[0][1],
            allow_blank=False,
            id=id,
        )

    # ------------------------------------------------------------------
    # Inicialização / refresh
    # ------------------------------------------------------------------

    def on_mount(
        self,
    ) -> None:
        self._prepare_tables()
        self.refresh_data()
        self._apply_builder_mode(
            ExperimentMode.TRAIN
        )

    def _prepare_tables(
        self,
    ) -> None:
        self.query_one(
            "#configs-table",
            DataTable,
        ).add_columns(
            "Arquivo",
            "Mode",
            "Estado",
        )

        self.query_one(
            "#runs-table",
            DataTable,
        ).add_columns(
            "Run",
            "Mode",
            "Created",
            "Seed",
            "Purpose",
        )

        self.query_one(
            "#components-table",
            DataTable,
        ).add_columns(
            "Kind",
            "Component ID",
            "Name",
            "Version",
        )

    def refresh_data(
        self,
    ) -> None:
        self._refresh_configs()
        self._refresh_runs()
        self._refresh_components()
        self._refresh_dashboard()
        self._refresh_builder_resources()

    def _refresh_configs(
        self,
    ) -> None:
        table = self.query_one(
            "#configs-table",
            DataTable,
        )

        table.clear()
        self._config_paths = []

        if not self.configs_root.is_dir():
            return

        for path in sorted(
            self.configs_root.glob(
                "*.toml"
            )
        ):
            self._config_paths.append(
                path
            )

            try:
                config = (
                    validate_config_file(
                        path
                    )
                )
                mode = config.mode.value
                state = "válida"

            except Exception:
                mode = "-"
                state = "inválida"

            table.add_row(
                path.name,
                mode,
                state,
            )

    def _refresh_runs(
        self,
    ) -> None:
        table = self.query_one(
            "#runs-table",
            DataTable,
        )

        table.clear()

        self._run_records = discover_runs(
            self.models_root
        )

        for record in self._run_records:
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
            )

    def _refresh_components(
        self,
    ) -> None:
        table = self.query_one(
            "#components-table",
            DataTable,
        )

        table.clear()

        for spec in self.registry.list():
            table.add_row(
                spec.kind.value,
                spec.component_id,
                spec.name,
                spec.version,
            )

    def _refresh_dashboard(
        self,
    ) -> None:
        components = self.registry.list()

        by_mode: dict[
            str,
            int,
        ] = {}

        for record in self._run_records:
            by_mode[record.mode] = (
                by_mode.get(
                    record.mode,
                    0,
                )
                + 1
            )

        mode_lines = (
            "\n".join(
                f"  {mode}: {count}"
                for mode, count
                in sorted(
                    by_mode.items()
                )
            )
            or "  nenhum run"
        )

        self.query_one(
            "#overview-runs",
            Static,
        ).update(
            (
                "[b]Runs[/b]\n"
                f"Total: {len(self._run_records)}\n"
                f"{mode_lines}"
            )
        )

        self.query_one(
            "#overview-components",
            Static,
        ).update(
            (
                "[b]Componentes[/b]\n"
                f"Registrados: {len(components)}\n"
                f"Configs TOML: "
                f"{len(self._config_paths)}"
            )
        )

        self.query_one(
            "#overview-safety",
            Static,
        ).update(
            (
                "[b]Execução[/b]\n"
                "train: treinamento local\n"
                "simulate: somente dry-run\n"
                "observe: captura passiva"
            )
        )

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    def on_select_changed(
        self,
        event: Select.Changed,
    ) -> None:
        select_id = (
            event.select.id
        )

        if (
            select_id
            == "builder-mode"
        ):
            try:
                mode = ExperimentMode(
                    str(
                        event.value
                    )
                )
            except ValueError:
                return

            self._apply_builder_mode(
                mode
            )

            if mode in {
                ExperimentMode.SIMULATE,
                ExperimentMode.OBSERVE,
            }:
                self._sync_source_run()

            return

        if (
            select_id
            == "builder-source-run"
        ):
            self._sync_source_run()

    def _apply_builder_mode(
        self,
        mode: ExperimentMode,
    ) -> None:
        self.query_one(
            "#builder-attack-group"
        ).display = (
            mode
            != ExperimentMode.OBSERVE
        )

        self.query_one(
            "#builder-training-group"
        ).display = (
            mode
            == ExperimentMode.TRAIN
        )

        self.query_one(
            "#builder-network-group"
        ).display = (
            mode
            in {
                ExperimentMode.SIMULATE,
                ExperimentMode.OBSERVE,
            }
        )

        self.query_one(
            "#builder-simulate-group"
        ).display = (
            mode
            == ExperimentMode.SIMULATE
        )

        preferred = (
            preferred_protocol_id(
                self.registry,
                mode,
            )
        )

        if preferred is not None:
            protocol_select = (
                self.query_one(
                    "#builder-protocol",
                    Select,
                )
            )

            try:
                protocol_select.value = (
                    preferred
                )
            except Exception:
                pass

        mode_notes = {
            ExperimentMode.TRAIN: (
                "TRAIN: dataset → preprocessing → "
                "Attacker/Defender → treinamento."
            ),
            ExperimentMode.SIMULATE: (
                "SIMULATE: usa checkpoints persistidos "
                "e executa somente em dry-run."
            ),
            ExperimentMode.OBSERVE: (
                "OBSERVE: captura passiva → Extractor → "
                "Defender → controle dry-run."
            ),
        }

        self._set_status(
            "#builder-status",
            mode_notes[mode],
        )

    def _sync_source_run(
        self,
    ) -> None:
        """
        Atualiza checkpoints e preprocessador de acordo com
        o TRAIN selecionado pelo usuário.
        """

        run_dir = (
            self._select_optional(
                "#builder-source-run"
            )
        )

        if not run_dir:
            self._replace_resource_options(
                "#builder-attacker-source",
                [],
                empty_label=(
                    "Selecione um run"
                ),
            )

            self._replace_resource_options(
                "#builder-defender-source",
                [],
                empty_label=(
                    "Selecione um run"
                ),
            )

            self.query_one(
                "#builder-preprocessor-source",
                Input,
            ).value = ""

            return

        try:
            resources = (
                inspect_run_resources(
                    run_dir
                )
            )
        except Exception as exc:
            self._set_status(
                "#builder-status",
                (
                    "[b]Run não pôde ser carregado[/b]\n"
                    f"{exc}"
                ),
                error=True,
            )
            return

        attacker_options = (
            attacker_checkpoint_options(
                resources
            )
        )

        defender_options = (
            defender_checkpoint_options(
                resources
            )
        )

        self._replace_resource_options(
            "#builder-attacker-source",
            attacker_options,
            empty_label=(
                "Nenhum checkpoint de Attacker"
            ),
        )

        self._replace_resource_options(
            "#builder-defender-source",
            defender_options,
            empty_label=(
                "Nenhum checkpoint de Defender"
            ),
        )

        preprocessor_input = (
            self.query_one(
                "#builder-preprocessor-source",
                Input,
            )
        )

        if (
            resources.preprocessor
            is not None
        ):
            preprocessor_input.value = str(
                resources.preprocessor
            )
        else:
            preprocessor_input.value = ""

        self._set_status(
            "#builder-status",
            (
                "[b]Run carregado[/b]\n"
                f"{resources.record.run_id}\n"
                f"Attacker checkpoints: "
                f"{len(attacker_options)}\n"
                f"Defender checkpoints: "
                f"{len(defender_options)}\n"
                f"Preprocessador: "
                f"{'OK' if resources.preprocessor else 'ausente'}"
            ),
        )

    def _replace_resource_options(
        self,
        selector: str,
        options: list[
            tuple[str, str]
        ],
        *,
        empty_label: str,
    ) -> None:
        """
        Atualiza dinamicamente um Select preservando a seleção
        atual quando ela ainda existir.
        """

        select = self.query_one(
            selector,
            Select,
        )

        current = (
            str(select.value)
            if select.value is not None
            else None
        )

        if not options:
            select.set_options(
                [
                    (
                        empty_label,
                        "",
                    )
                ]
            )
            select.value = ""
            select.disabled = True
            return

        select.disabled = False

        select.set_options(
            options
        )

        valid_values = {
            value
            for _, value
            in options
        }

        if (
            current
            in valid_values
        ):
            select.value = current
        else:
            select.value = (
                options[0][1]
            )

    def _refresh_builder_resources(
        self,
    ) -> None:
        """
        Atualiza datasets e runs encontrados.

        Isso permite que um TRAIN recém-concluído apareça
        imediatamente como origem de SIMULATE/OBSERVE.
        """

        self._replace_resource_options(
            "#builder-dataset-source",
            dataset_file_options(
                "data"
            ),
            empty_label=(
                "Nenhum arquivo encontrado em data/"
            ),
        )

        self._replace_resource_options(
            "#builder-source-run",
            training_run_options(
                self.models_root
            ),
            empty_label=(
                "Nenhum TRAIN encontrado"
            ),
        )

        self._sync_source_run()

    def _builder_values(
        self,
    ) -> BuilderValues:
        mode = ExperimentMode(
            self._select_value(
                "#builder-mode"
            )
        )

        return BuilderValues(
            mode=mode,

            dataset_id=(
                self._select_optional(
                    "#builder-dataset"
                )
                if (
                    mode
                    != ExperimentMode
                    .OBSERVE
                )
                else None
            ),
            dataset_source=(
                self._select_optional(
                    "#builder-dataset-source"
                )
                if (
                    mode
                    != ExperimentMode
                    .OBSERVE
                )
                else None
            ),

            attacker_id=(
                self._select_optional(
                    "#builder-attacker"
                )
                if (
                    mode
                    != ExperimentMode
                    .OBSERVE
                )
                else None
            ),
            attacker_source=(
                self._select_optional(
                    "#builder-attacker-source"
                )
                if (
                    mode
                    == ExperimentMode
                    .SIMULATE
                )
                else None
            ),

            defender_id=(
                self._select_value(
                    "#builder-defender"
                )
            ),
            defender_source=(
                self._select_optional(
                    "#builder-defender-source"
                )
                if (
                    mode
                    in {
                        ExperimentMode
                        .SIMULATE,
                        ExperimentMode
                        .OBSERVE,
                    }
                )
                else None
            ),

            protocol_id=(
                self._select_value(
                    "#builder-protocol"
                )
            ),

            renderer_id=(
                self._select_optional(
                    "#builder-renderer"
                )
                if (
                    mode
                    == ExperimentMode
                    .SIMULATE
                )
                else None
            ),
            network_backend_id=(
                self._select_optional(
                    "#builder-backend"
                )
                if (
                    mode
                    == ExperimentMode
                    .SIMULATE
                )
                else None
            ),
            capture_id=(
                self._select_optional(
                    "#builder-capture"
                )
                if (
                    mode
                    in {
                        ExperimentMode
                        .SIMULATE,
                        ExperimentMode
                        .OBSERVE,
                    }
                )
                else None
            ),
            extractor_id=(
                self._select_optional(
                    "#builder-extractor"
                )
                if (
                    mode
                    in {
                        ExperimentMode
                        .SIMULATE,
                        ExperimentMode
                        .OBSERVE,
                    }
                )
                else None
            ),

            seed=self._int_input(
                "#builder-seed",
                "seed",
            ),
            device=self._select_value(
                "#builder-device"
            ),
            output_dir=self._input_value(
                "#builder-output-dir"
            ),

            test_size=self._float_input(
                "#builder-test-size",
                "test_size",
            ),
            validation_size=(
                self._float_input(
                    "#builder-validation-size",
                    "validation_size",
                )
            ),

            noise_dim=self._int_input(
                "#builder-noise-dim",
                "noise_dim",
            ),
            lr_defensor=self._float_input(
                "#builder-lr-defender",
                "lr_defensor",
            ),
            lr_atacante=self._float_input(
                "#builder-lr-attacker",
                "lr_atacante",
            ),
            epsilon=self._float_input(
                "#builder-epsilon",
                "epsilon",
            ),
            classification_threshold=(
                self._float_input(
                    "#builder-threshold",
                    "classification_threshold",
                )
            ),

            epochs_pretrain=self._int_input(
                "#builder-epochs-pretrain",
                "epochs_pretrain",
            ),
            epochs_por_rodada=(
                self._int_input(
                    "#builder-epochs-round",
                    "epochs_por_rodada",
                )
            ),
            n_rodadas=self._int_input(
                "#builder-rounds",
                "n_rodadas",
            ),
            amostras_por_rodada=(
                self._int_input(
                    "#builder-samples-round",
                    "amostras_por_rodada",
                )
            ),
            amostras_avaliacao_adversarial=(
                self._int_input(
                    "#builder-eval-samples",
                    (
                        "amostras_avaliacao_"
                        "adversarial"
                    ),
                )
            ),
            batch_size=self._int_input(
                "#builder-batch-size",
                "batch_size",
            ),

            preprocessor_source=(
                self._input_value(
                    "#builder-preprocessor-source"
                )
                if (
                    mode
                    in {
                        ExperimentMode
                        .SIMULATE,
                        ExperimentMode
                        .OBSERVE,
                    }
                )
                else None
            ),
            sample_count=self._int_input(
                "#builder-sample-count",
                "sample_count",
            ),
            packet_limit=(
                self._optional_int_input(
                    "#builder-packet-limit",
                    "packet_limit",
                )
            ),
            capture_duration=(
                self._float_input(
                    "#builder-capture-duration",
                    "capture_duration",
                )
            ),

            capture_iface=(
                self._input_value(
                    "#builder-capture-iface"
                )
                if (
                    mode
                    in {
                        ExperimentMode
                        .SIMULATE,
                        ExperimentMode
                        .OBSERVE,
                    }
                )
                else None
            ),
            target_ip=(
                self._input_value(
                    "#builder-target-ip"
                )
                if (
                    mode
                    == ExperimentMode
                    .SIMULATE
                )
                else None
            ),
            target_port=self._int_input(
                "#builder-target-port",
                "target_port",
            ),

            log_level=self._select_value(
                "#builder-log-level"
            ),
        )

    def _build_from_ui(
        self,
    ):
        values = self._builder_values()

        config = build_experiment_config(
            values
        )

        return (
            config,
            values,
        )

    def _validate_builder(
        self,
    ) -> None:
        try:
            config, values = (
                self._build_from_ui()
            )

            validate_experiment_config(
                config
            )

        except Exception as exc:
            self._set_status(
                "#builder-status",
                (
                    "[b]Experimento inválido[/b]\n"
                    f"{exc}"
                ),
                error=True,
            )
            return

        components = [
            (
                "Defender",
                config.defender.component_id,
            ),
            (
                "Protocol",
                (
                    config.protocol.component_id
                    if config.protocol
                    else "-"
                ),
            ),
        ]

        if config.attack_dataset:
            components.append(
                (
                    "Dataset",
                    config
                    .attack_dataset
                    .component_id,
                )
            )

        if config.attacker:
            components.append(
                (
                    "Attacker",
                    config
                    .attacker
                    .component_id,
                )
            )

        component_text = "\n".join(
            f"{label}: {value}"
            for label, value
            in components
        )

        self._set_status(
            "#builder-status",
            (
                "[b]Experimento válido[/b]\n"
                f"Mode: {config.mode.value}\n"
                f"Seed: {config.seed}\n"
                f"Threshold: "
                f"{values.classification_threshold}\n"
                f"{component_text}"
            ),
        )

    def _save_builder_config(
        self,
    ) -> None:
        try:
            config, _ = (
                self._build_from_ui()
            )

            validate_experiment_config(
                config
            )

            raw_path = self._input_value(
                "#builder-save-path"
            )

            if not raw_path:
                raise ValueError(
                    "Informe o caminho do arquivo TOML."
                )

            path = save_experiment_config(
                config,
                raw_path,
            )

        except Exception as exc:
            self._set_status(
                "#builder-status",
                (
                    "[b]Não foi possível salvar[/b]\n"
                    f"{exc}"
                ),
                error=True,
            )
            return

        self._set_status(
            "#builder-status",
            (
                "[b]Configuração salva[/b]\n"
                f"{path}\n\n"
                "O arquivo pode ser executado "
                "posteriormente pela CLI ou pela TUI."
            ),
        )

        self.refresh_data()

    def _start_builder(
        self,
    ) -> None:
        try:
            config, values = (
                self._build_from_ui()
            )

            validate_experiment_config(
                config
            )

        except Exception as exc:
            self._set_status(
                "#builder-status",
                (
                    "[b]Experimento inválido[/b]\n"
                    f"{exc}"
                ),
                error=True,
            )
            return

        button = self.query_one(
            "#builder-execute",
            Button,
        )
        button.disabled = True

        self._set_status(
            "#builder-status",
            (
                "[b]Executando experimento...[/b]\n"
                f"Mode: {config.mode.value}\n"
                f"Output: {config.output_dir}\n\n"
                "A interface continuará responsiva."
            ),
        )

        self._execute_builder_worker(
            config,
            values.log_level,
        )

    @work(
        thread=True,
        exclusive=True,
        group="experiment",
        exit_on_error=False,
    )
    def _execute_builder_worker(
        self,
        config,
        log_level: str,
    ) -> None:
        try:
            result = (
                execute_experiment_config(
                    config
                )
            )
        except Exception as exc:
            self.call_from_thread(
                self._execution_failed,
                "#builder-execute",
                "#builder-status",
                str(exc),
            )
            return

        self.call_from_thread(
            self._execution_finished,
            result,
            "#builder-execute",
            "#builder-status",
            (
                log_level
                == "detailed"
            ),
        )

    # ------------------------------------------------------------------
    # TOML runner
    # ------------------------------------------------------------------

    def _selected_config_path(
        self,
    ) -> Path | None:
        value = self.query_one(
            "#config-input",
            Input,
        ).value.strip()

        if not value:
            self._set_status(
                "#status",
                (
                    "Informe o caminho de "
                    "um arquivo TOML."
                ),
                error=True,
            )
            return None

        return Path(value)

    def _validate_selected_config(
        self,
    ) -> None:
        path = self._selected_config_path()

        if path is None:
            return

        try:
            config = validate_config_file(
                path
            )
        except Exception as exc:
            self._set_status(
                "#status",
                (
                    "[b]Configuração inválida[/b]\n"
                    f"{exc}"
                ),
                error=True,
            )
            return

        protocol = (
            config.protocol.component_id
            if config.protocol is not None
            else "-"
        )

        self._set_status(
            "#status",
            (
                "[b]Configuração válida[/b]\n"
                f"Arquivo: {path}\n"
                f"Mode: {config.mode.value}\n"
                f"Defender: "
                f"{config.defender.component_id}\n"
                f"Protocol: {protocol}"
            ),
        )

    def _start_selected_config(
        self,
    ) -> None:
        path = self._selected_config_path()

        if path is None:
            return

        button = self.query_one(
            "#execute-config",
            Button,
        )
        button.disabled = True

        self._set_status(
            "#status",
            (
                "[b]Executando experimento...[/b]\n"
                f"{path}\n\n"
                "A interface continuará responsiva."
            ),
        )

        self._execute_toml_worker(
            path
        )

    @work(
        thread=True,
        exclusive=True,
        group="experiment",
        exit_on_error=False,
    )
    def _execute_toml_worker(
        self,
        path: Path,
    ) -> None:
        try:
            result = execute_config(
                path
            )
        except Exception as exc:
            self.call_from_thread(
                self._execution_failed,
                "#execute-config",
                "#status",
                str(exc),
            )
            return

        self.call_from_thread(
            self._execution_finished,
            result,
            "#execute-config",
            "#status",
            False,
        )

    # ------------------------------------------------------------------
    # Eventos
    # ------------------------------------------------------------------

    def on_button_pressed(
        self,
        event: Button.Pressed,
    ) -> None:
        button_id = event.button.id

        if button_id == "builder-validate":
            self._validate_builder()

        elif button_id == "builder-save":
            self._save_builder_config()

        elif button_id == "builder-execute":
            self._start_builder()

        elif button_id == "validate-config":
            self._validate_selected_config()

        elif button_id == "execute-config":
            self._start_selected_config()

    def on_data_table_row_selected(
        self,
        event: DataTable.RowSelected,
    ) -> None:
        table_id = event.data_table.id

        if table_id == "configs-table":
            self._select_config_row(
                event.cursor_row
            )

        elif table_id == "runs-table":
            self._show_run_details(
                event.cursor_row
            )

    def _select_config_row(
        self,
        row_index: int,
    ) -> None:
        if not (
            0
            <= row_index
            < len(
                self._config_paths
            )
        ):
            return

        path = self._config_paths[
            row_index
        ]

        self.query_one(
            "#config-input",
            Input,
        ).value = str(path)

        self._validate_selected_config()

    def _show_run_details(
        self,
        row_index: int,
    ) -> None:
        if not (
            0
            <= row_index
            < len(
                self._run_records
            )
        ):
            return

        record = self._run_records[
            row_index
        ]

        components = (
            summarize_components(
                record
            )
        )

        counts = artifact_counts(
            record
        )

        component_lines = (
            "\n".join(
                (
                    f"  {role}: "
                    f"{component_id or '-'}"
                )
                for role, component_id
                in components.items()
            )
            or "  -"
        )

        artifact_lines = (
            "\n".join(
                f"  {name}: {count}"
                for name, count
                in counts.items()
            )
            or "  -"
        )

        important_artifacts = (
            self._important_artifacts(
                record.run_dir
            )
        )

        important_text = (
            "\n".join(
                f"  {label}: {path}"
                for label, path
                in important_artifacts
            )
            or "  -"
        )

        self.query_one(
            "#run-details",
            Static,
        ).update(
            (
                f"[b]{record.run_id}[/b]\n\n"
                f"Mode: {record.mode}\n"
                f"Created: "
                f"{record.created_at or '-'}\n"
                f"Seed: "
                f"{record.seed if record.seed is not None else '-'}\n"
                f"Purpose: "
                f"{record.purpose or '-'}\n"
                f"Path: {record.run_dir}\n\n"
                "[b]Components[/b]\n"
                f"{component_lines}\n\n"
                "[b]Artifact counts[/b]\n"
                f"{artifact_lines}\n\n"
                "[b]Resultados importantes[/b]\n"
                f"{important_text}"
            )
        )

    @staticmethod
    def _important_artifacts(
        run_dir: Path,
    ) -> list[tuple[str, str]]:
        """
        Lista artefatos úteis para inspeção humana do run.

        A TUI apenas aponta para os arquivos persistidos; ela não
        interpreta nem altera os resultados científicos.
        """

        candidates = (
            (
                "Config",
                run_dir
                / "config_execucao.json",
            ),
            (
                "Summary",
                run_dir
                / "summary.json",
            ),
            (
                "Histórico",
                run_dir
                / "historico_treino.json",
            ),
            (
                "Matriz A×D",
                run_dir
                / "matriz_checkpoints.csv",
            ),
            (
                "Log",
                run_dir
                / "logs"
                / "run.log",
            ),
            (
                "Observação",
                run_dir
                / "metrics"
                / "network_observation.json",
            ),
        )

        result: list[
            tuple[str, str]
        ] = []

        for label, path in candidates:
            if path.is_file():
                result.append(
                    (
                        label,
                        str(path),
                    )
                )

        plots_dir = (
            run_dir
            / "plots"
        )

        if plots_dir.is_dir():
            plot_count = sum(
                1
                for path
                in plots_dir.iterdir()
                if path.is_file()
            )

            if plot_count:
                result.append(
                    (
                        f"Plots ({plot_count})",
                        str(plots_dir),
                    )
                )

        return result

    # ------------------------------------------------------------------
    # Resultado / parsing
    # ------------------------------------------------------------------

    def _execution_failed(
        self,
        button_id: str,
        status_id: str,
        message: str,
    ) -> None:
        self.query_one(
            button_id,
            Button,
        ).disabled = False

        self._set_status(
            status_id,
            (
                "[b]Execução não concluída[/b]\n"
                f"{message}"
            ),
            error=True,
        )

    def _execution_finished(
        self,
        result,
        button_id: str,
        status_id: str,
        detailed: bool,
    ) -> None:
        self.query_one(
            button_id,
            Button,
        ).disabled = False

        summary = (
            self._format_result_summary(
                result,
                detailed=detailed,
            )
        )

        self._set_status(
            status_id,
            (
                "[b]Experimento concluído[/b]\n"
                f"Run: {result.run_id}\n"
                f"Artefatos: {result.run_dir}\n\n"
                f"{summary}"
            ),
        )

        self.refresh_data()

    def _format_result_summary(
        self,
        result,
        *,
        detailed: bool,
    ) -> str:
        """
        Apresenta o resultado de acordo com a semântica do modo.

        A intenção é evitar reduzir TRAIN, SIMULATE e OBSERVE a uma
        lista genérica de métricas escalares.
        """

        history = (
            result.history
            or {}
        )

        metrics = (
            result.final_metrics
            or {}
        )

        if (
            history.get(
                "taxa_evasao_pre_adaptacao"
            )
            or history.get(
                "taxa_evasao_pos_adaptacao"
            )
        ):
            return (
                self._format_train_summary(
                    result,
                    detailed=detailed,
                )
            )

        if (
            "samples_selected"
            in metrics
            and "mathematical_evasions"
            in metrics
        ):
            return (
                self._format_simulate_summary(
                    result,
                    detailed=detailed,
                )
            )

        if (
            "captured_packets"
            in metrics
            and "reconstructed_flows"
            in metrics
        ):
            return (
                self._format_observe_summary(
                    result,
                    detailed=detailed,
                )
            )

        return self._format_generic_metrics(
            metrics,
            detailed=detailed,
        )

    def _format_train_summary(
        self,
        result,
        *,
        detailed: bool,
    ) -> str:
        history = (
            result.history
            or {}
        )

        metrics = (
            result.final_metrics
            or {}
        )

        pre = list(
            history.get(
                "taxa_evasao_pre_adaptacao",
                [],
            )
            or []
        )

        post = list(
            history.get(
                "taxa_evasao_pos_adaptacao",
                [],
            )
            or []
        )

        round_count = max(
            len(pre),
            len(post),
        )

        lines = [
            "[b]Evolução adversarial[/b]",
        ]

        selected_rounds = (
            self._round_indexes(
                round_count,
                detailed=detailed,
            )
        )

        previous = None

        for index in selected_rounds:
            if (
                previous is not None
                and index
                > previous + 1
            ):
                lines.append(
                    "  ..."
                )

            round_number = (
                index + 1
            )

            pre_value = (
                pre[index]
                if index < len(pre)
                else None
            )

            post_value = (
                post[index]
                if index < len(post)
                else None
            )

            lines.append(
                (
                    f"  R{round_number:02d}: "
                    f"A{round_number}×D{round_number - 1} "
                    f"{self._percent(pre_value)} "
                    "→ "
                    f"A{round_number}×D{round_number} "
                    f"{self._percent(post_value)}"
                )
            )

            previous = index

        if round_count == 0:
            lines.append(
                "  histórico por rodada indisponível"
            )

        lines.extend(
            [
                "",
                "[b]Defender final — teste reservado[/b]",
                (
                    "  Acc: "
                    f"{self._metric_percent(metrics, 'accuracy')} | "
                    "Precision: "
                    f"{self._metric_percent(metrics, 'precision')} | "
                    "Recall: "
                    f"{self._metric_percent(metrics, 'recall')}"
                ),
                (
                    "  F1: "
                    f"{self._metric_percent(metrics, 'f1')} | "
                    "FPR: "
                    f"{self._metric_percent(metrics, 'fpr')} | "
                    "FNR: "
                    f"{self._metric_percent(metrics, 'fnr')}"
                ),
                (
                    "  ROC-AUC: "
                    f"{self._metric_number(metrics, 'roc_auc')}"
                ),
            ]
        )

        artifacts = (
            self._important_artifacts(
                Path(result.run_dir)
            )
        )

        if artifacts:
            lines.extend(
                [
                    "",
                    "[b]Resultados persistidos[/b]",
                ]
            )

            for label, path in artifacts:
                if label.startswith(
                    (
                        "Histórico",
                        "Matriz",
                        "Plots",
                        "Summary",
                    )
                ):
                    lines.append(
                        f"  {label}: {path}"
                    )

        return "\n".join(
            lines
        )

    def _format_simulate_summary(
        self,
        result,
        *,
        detailed: bool,
    ) -> str:
        metrics = (
            result.final_metrics
            or {}
        )

        snapshot = (
            self._load_run_snapshot(
                Path(result.run_dir)
            )
        )

        confrontation = (
            self._confrontation_label(
                snapshot
            )
        )

        samples = int(
            metrics.get(
                "samples_selected",
                0,
            )
            or 0
        )

        evasions = int(
            metrics.get(
                "mathematical_evasions",
                0,
            )
            or 0
        )

        valid = int(
            metrics.get(
                "valid_renderings",
                0,
            )
            or 0
        )

        executions = int(
            metrics.get(
                "dry_run_executions",
                0,
            )
            or 0
        )

        evasion_rate = float(
            metrics.get(
                "mathematical_evasion_rate",
                0.0,
            )
            or 0.0
        )

        valid_given_evasion = float(
            metrics.get(
                "valid_rendering_rate_given_evasion",
                0.0,
            )
            or 0.0
        )

        lines = [
            "[b]Cenário de simulação[/b]",
            f"  Confronto: {confrontation}",
            "",
            "[b]Funil[/b]",
            f"  {samples} amostras selecionadas",
            (
                "       ↓ "
                f"{evasion_rate * 100:.2f}%"
            ),
            f"  {evasions} evasões matemáticas",
            (
                "       ↓ "
                f"{valid_given_evasion * 100:.2f}% das evasões"
            ),
            f"  {valid} traduções válidas",
            "       ↓",
            f"  {executions} execuções dry-run",
        ]

        if detailed:
            lines.extend(
                [
                    "",
                    "[b]Parâmetros[/b]",
                    (
                        "  Threshold: "
                        f"{self._metric_number(metrics, 'classification_threshold')}"
                    ),
                    (
                        "  Epsilon: "
                        f"{self._metric_number(metrics, 'epsilon')}"
                    ),
                    (
                        "  Taxa de traduções no total: "
                        f"{self._metric_percent(metrics, 'valid_rendering_rate')}"
                    ),
                    (
                        "  Dry-run: "
                        f"{metrics.get('dry_run', True)}"
                    ),
                ]
            )

        return "\n".join(
            lines
        )

    def _format_observe_summary(
        self,
        result,
        *,
        detailed: bool,
    ) -> str:
        metrics = (
            result.final_metrics
            or {}
        )

        packets = metrics.get(
            "captured_packets",
            0,
        )
        flows = metrics.get(
            "reconstructed_flows",
            0,
        )
        benign = metrics.get(
            "network_benign_count",
            0,
        )
        attacks = metrics.get(
            "network_attack_count",
            0,
        )
        blocks = metrics.get(
            "block_decisions",
            0,
        )
        applied = metrics.get(
            "rules_applied",
            0,
        )

        lines = [
            "[b]Observação passiva[/b]",
            f"  Pacotes capturados: {packets}",
            f"  Fluxos reconstruídos: {flows}",
            "",
            "[b]Classificação[/b]",
            f"  Benignos: {benign}",
            f"  Ataques: {attacks}",
            (
                "  Taxa de ataque: "
                f"{self._metric_percent(metrics, 'network_attack_rate')}"
            ),
            "",
            "[b]Controle[/b]",
            f"  Decisões BLOCK: {blocks}",
            f"  Regras aplicadas: {applied}",
            "  Enforcement: dry-run",
        ]

        if detailed:
            lines.extend(
                [
                    "",
                    "[b]Captura[/b]",
                    (
                        "  Duração: "
                        f"{self._metric_number(metrics, 'capture_duration')} s"
                    ),
                    (
                        "  Packet limit: "
                        f"{metrics.get('packet_limit', '-')}"
                    ),
                    (
                        "  Threshold: "
                        f"{self._metric_number(metrics, 'classification_threshold')}"
                    ),
                ]
            )

        return "\n".join(
            lines
        )

    @staticmethod
    def _round_indexes(
        count: int,
        *,
        detailed: bool,
    ) -> list[int]:
        if count <= 0:
            return []

        if detailed or count <= 7:
            return list(
                range(count)
            )

        indexes = [
            0,
            1,
            2,
            count - 2,
            count - 1,
        ]

        return sorted(
            set(indexes)
        )

    @staticmethod
    def _percent(
        value: Any,
    ) -> str:
        if not isinstance(
            value,
            (int, float),
        ):
            return "-"

        return f"{float(value) * 100:.2f}%"

    def _metric_percent(
        self,
        metrics: dict,
        key: str,
    ) -> str:
        return self._percent(
            metrics.get(key)
        )

    @staticmethod
    def _metric_number(
        metrics: dict,
        key: str,
    ) -> str:
        value = metrics.get(
            key
        )

        if isinstance(
            value,
            float,
        ):
            return f"{value:.6g}"

        if value is None:
            return "-"

        return str(value)

    @staticmethod
    def _format_generic_metrics(
        metrics: dict,
        *,
        detailed: bool,
    ) -> str:
        metric_lines = []

        for key, value in metrics.items():
            if isinstance(
                value,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            ):
                metric_lines.append(
                    f"{key}: {value}"
                )

        limit = (
            20
            if detailed
            else 8
        )

        return (
            "\n".join(
                metric_lines[:limit]
            )
            or "(sem métricas escalares)"
        )

    @staticmethod
    def _load_run_snapshot(
        run_dir: Path,
    ) -> dict[str, Any]:
        path = (
            run_dir
            / "config_execucao.json"
        )

        if not path.is_file():
            return {}

        try:
            with open(
                path,
                encoding="utf-8",
            ) as file:
                payload = json.load(
                    file
                )
        except (
            OSError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ):
            return {}

        return (
            payload
            if isinstance(
                payload,
                dict,
            )
            else {}
        )

    def _confrontation_label(
        self,
        snapshot: dict[str, Any],
    ) -> str:
        configs = snapshot.get(
            "component_config",
            {}
        )

        if not isinstance(
            configs,
            dict,
        ):
            return "checkpoint selecionado"

        attacker = configs.get(
            "attacker"
        )
        defender = configs.get(
            "defender"
        )

        attacker_source = (
            attacker.get("source")
            if isinstance(
                attacker,
                dict,
            )
            else None
        )

        defender_source = (
            defender.get("source")
            if isinstance(
                defender,
                dict,
            )
            else None
        )

        attacker_label = (
            self._checkpoint_label(
                attacker_source,
                role="attacker",
            )
        )

        defender_label = (
            self._checkpoint_label(
                defender_source,
                role="defender",
            )
        )

        if (
            attacker_label
            and defender_label
        ):
            return (
                f"{attacker_label} × "
                f"{defender_label}"
            )

        return "checkpoint selecionado"

    @staticmethod
    def _checkpoint_label(
        source: Any,
        *,
        role: str,
    ) -> str | None:
        if not isinstance(
            source,
            str,
        ):
            return None

        stem = (
            Path(source)
            .stem
            .lower()
        )

        if role == "attacker":
            if stem in {
                "atacante_final",
                "attacker_final",
            }:
                return "Afinal"

            marker = "_rodada_"

            if marker in stem:
                raw = stem.rsplit(
                    marker,
                    1,
                )[1]

                try:
                    return f"A{int(raw)}"
                except ValueError:
                    pass

        if role == "defender":
            if stem in {
                "defensor_adaptativo_final",
                "defensor_final",
                "defender_final",
            }:
                return "Dfinal"

            marker = "_rodada_"

            if marker in stem:
                raw = stem.rsplit(
                    marker,
                    1,
                )[1]

                try:
                    return f"D{int(raw)}"
                except ValueError:
                    pass

        return Path(source).name

    def _set_status(
        self,
        selector: str,
        text: str,
        *,
        error: bool = False,
    ) -> None:
        status = self.query_one(
            selector,
            Static,
        )

        status.update(
            text
        )

        status.styles.border = (
            (
                "round",
                "red",
            )
            if error
            else (
                "round",
                "green",
            )
        )

    def _select_value(
        self,
        selector: str,
    ) -> str:
        value = self.query_one(
            selector,
            Select,
        ).value

        if value is None:
            raise ValueError(
                f"{selector}: valor não selecionado"
            )

        text = str(value)

        if not text:
            raise ValueError(
                f"{selector}: componente não disponível"
            )

        return text

    def _select_optional(
        self,
        selector: str,
    ) -> str | None:
        try:
            return self._select_value(
                selector
            )
        except ValueError:
            return None

    def _input_value(
        self,
        selector: str,
    ) -> str:
        return self.query_one(
            selector,
            Input,
        ).value.strip()

    def _int_input(
        self,
        selector: str,
        name: str,
    ) -> int:
        raw = self._input_value(
            selector
        )

        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{name} deve ser inteiro"
            ) from exc

    def _optional_int_input(
        self,
        selector: str,
        name: str,
    ) -> int | None:
        raw = self._input_value(
            selector
        )

        if not raw:
            return None

        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{name} deve ser inteiro"
            ) from exc

    def _float_input(
        self,
        selector: str,
        name: str,
    ) -> float:
        raw = self._input_value(
            selector
        )

        try:
            return float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{name} deve ser numérico"
            ) from exc

    # ------------------------------------------------------------------
    # Atalhos
    # ------------------------------------------------------------------

    def action_refresh(
        self,
    ) -> None:
        self.refresh_data()
        self.notify(
            "Dados atualizados."
        )

    def _activate_tab(
        self,
        tab_id: str,
    ) -> None:
        self.query_one(
            "#tabs",
            TabbedContent,
        ).active = tab_id

    def action_tab_dashboard(
        self,
    ) -> None:
        self._activate_tab(
            "dashboard"
        )

    def action_tab_execute(
        self,
    ) -> None:
        self._activate_tab(
            "execute"
        )

    def action_tab_runs(
        self,
    ) -> None:
        self._activate_tab(
            "runs"
        )

    def action_tab_components(
        self,
    ) -> None:
        self._activate_tab(
            "components"
        )


def run_tui() -> None:
    ADArenaTUI().run()
