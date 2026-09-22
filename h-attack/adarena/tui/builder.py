from __future__ import annotations

from dataclasses import dataclass

from adarena.core.components import (
    ComponentKind,
)
from adarena.core.config import (
    ComponentSelection,
    DataSplitConfig,
    ExperimentConfig,
    ExperimentMode,
    NetworkSettings,
    TrainingSettings,
)
from adarena.core.registry import (
    ComponentRegistry,
)


@dataclass(slots=True)
class BuilderValues:
    """
    Valores coletados pela TUI para construir um ExperimentConfig.

    A classe não conhece widgets. Ela funciona como uma fronteira
    simples entre a interface e o Core.
    """

    mode: ExperimentMode

    dataset_id: str | None
    dataset_source: str | None

    attacker_id: str | None
    attacker_source: str | None

    defender_id: str
    defender_source: str | None

    protocol_id: str

    renderer_id: str | None = None
    network_backend_id: str | None = None
    capture_id: str | None = None
    extractor_id: str | None = None

    seed: int = 42
    device: str = "cpu"
    output_dir: str = "models"

    test_size: float = 0.2
    validation_size: float = 0.1

    noise_dim: int = 32
    lr_defensor: float = 1e-3
    lr_atacante: float = 2e-4
    epsilon: float = 0.3
    classification_threshold: float = 0.5

    epochs_pretrain: int = 5
    epochs_por_rodada: int = 3
    n_rodadas: int = 20

    amostras_por_rodada: int = 5000
    amostras_avaliacao_adversarial: int = 5000
    batch_size: int = 512

    preprocessor_source: str | None = None
    sample_count: int = 20
    packet_limit: int | None = None
    capture_duration: float = 5.0

    capture_iface: str | None = None
    target_ip: str | None = None
    target_port: int = 80

    log_level: str = "normal"


def component_options(
    registry: ComponentRegistry,
    kind: ComponentKind,
) -> list[tuple[str, str]]:
    """
    Converte componentes do Registry em opções de interface.

    O valor persistido continua sendo o component_id.
    """

    return [
        (
            f"{spec.name} — {spec.component_id}",
            spec.component_id,
        )
        for spec in registry.list(
            kind=kind
        )
    ]


def first_component_id(
    registry: ComponentRegistry,
    kind: ComponentKind,
) -> str | None:
    options = component_options(
        registry,
        kind,
    )

    if not options:
        return None

    return options[0][1]


def preferred_protocol_id(
    registry: ComponentRegistry,
    mode: ExperimentMode,
) -> str | None:
    """
    Escolhe um protocolo inicial sem acoplar a TUI a IDs concretos.

    Primeiro usa tags/nome/ID. Se nenhum protocolo declarar uma pista
    compatível, retorna o primeiro protocolo registrado.
    """

    specs = registry.list(
        kind=(
            ComponentKind
            .EXPERIMENT_PROTOCOL
        )
    )

    if not specs:
        return None

    tokens = {
        ExperimentMode.TRAIN: (
            "training",
            "train",
            "adversarial",
        ),
        ExperimentMode.SIMULATE: (
            "simulation",
            "simulate",
        ),
        ExperimentMode.OBSERVE: (
            "observation",
            "observe",
        ),
    }.get(
        mode,
        (),
    )

    for spec in specs:
        haystack = " ".join(
            (
                spec.component_id,
                spec.name,
                *spec.tags,
            )
        ).lower()

        if any(
            token in haystack
            for token in tokens
        ):
            return spec.component_id

    return specs[0].component_id


def build_experiment_config(
    values: BuilderValues,
) -> ExperimentConfig:
    """
    Constrói a configuração declarativa usada pelo ExperimentRunner.

    A TUI não cria modelos, datasets ou protocolos diretamente.
    Ela apenas descreve o experimento.
    """

    _validate_builder_values(
        values
    )

    training = TrainingSettings(
        noise_dim=values.noise_dim,
        lr_defensor=values.lr_defensor,
        lr_atacante=values.lr_atacante,
        epsilon=values.epsilon,
        classification_threshold=(
            values.classification_threshold
        ),
        epochs_pretrain=(
            values.epochs_pretrain
        ),
        epochs_por_rodada=(
            values.epochs_por_rodada
        ),
        n_rodadas=values.n_rodadas,
        amostras_por_rodada=(
            values.amostras_por_rodada
        ),
        amostras_avaliacao_adversarial=(
            values
            .amostras_avaliacao_adversarial
        ),
        batch_size=values.batch_size,
    )

    split = DataSplitConfig(
        test_size=values.test_size,
        validation_size=(
            values.validation_size
        ),
    )

    defender = ComponentSelection(
        component_id=values.defender_id,
        source=_clean(
            values.defender_source
        ),
    )

    protocol = ComponentSelection(
        component_id=values.protocol_id,
    )

    attacker = None
    attack_dataset = None
    network = None

    if values.mode in {
        ExperimentMode.TRAIN,
        ExperimentMode.SIMULATE,
    }:
        assert values.attacker_id is not None
        assert values.dataset_id is not None

        attacker = ComponentSelection(
            component_id=(
                values.attacker_id
            ),
            source=_clean(
                values.attacker_source
            ),
        )

        attack_dataset = (
            ComponentSelection(
                component_id=(
                    values.dataset_id
                ),
                source=_clean(
                    values.dataset_source
                ),
            )
        )

    if values.mode == ExperimentMode.SIMULATE:
        assert values.renderer_id is not None
        assert (
            values.network_backend_id
            is not None
        )
        assert values.capture_id is not None
        assert values.extractor_id is not None

        renderer_params = {}

        if _clean(values.target_ip):
            renderer_params[
                "target_ip"
            ] = _clean(
                values.target_ip
            )

        renderer_params[
            "target_port"
        ] = values.target_port

        capture_params = {}

        if _clean(values.capture_iface):
            capture_params[
                "iface"
            ] = _clean(
                values.capture_iface
            )

        network = NetworkSettings(
            renderer=ComponentSelection(
                component_id=(
                    values.renderer_id
                ),
                params=renderer_params,
            ),
            network_backend=(
                ComponentSelection(
                    component_id=(
                        values
                        .network_backend_id
                    ),
                    params={
                        "dry_run": True,
                    },
                )
            ),
            capture=ComponentSelection(
                component_id=(
                    values.capture_id
                ),
                params=capture_params,
            ),
            extractor=ComponentSelection(
                component_id=(
                    values.extractor_id
                ),
            ),
            preprocessor_source=_clean(
                values.preprocessor_source
            ),
            sample_count=(
                values.sample_count
            ),
            packet_limit=(
                values.packet_limit
            ),
            capture_duration=(
                values.capture_duration
            ),
            classification_threshold=(
                values
                .classification_threshold
            ),
            dry_run=True,
            observe=False,
        )

    elif values.mode == ExperimentMode.OBSERVE:
        assert values.capture_id is not None
        assert values.extractor_id is not None

        capture_params = {}

        if _clean(values.capture_iface):
            capture_params[
                "iface"
            ] = _clean(
                values.capture_iface
            )

        network = NetworkSettings(
            capture=ComponentSelection(
                component_id=(
                    values.capture_id
                ),
                params=capture_params,
            ),
            extractor=ComponentSelection(
                component_id=(
                    values.extractor_id
                ),
            ),
            preprocessor_source=_clean(
                values.preprocessor_source
            ),
            packet_limit=(
                values.packet_limit
            ),
            capture_duration=(
                values.capture_duration
            ),
            classification_threshold=(
                values
                .classification_threshold
            ),
            dry_run=True,
            observe=True,
        )

    return ExperimentConfig(
        attacker=attacker,
        defender=defender,
        attack_dataset=attack_dataset,
        benign_dataset=None,
        protocol=protocol,
        mode=values.mode,
        seed=values.seed,
        device=values.device,
        output_dir=values.output_dir,
        split=split,
        training=training,
        network=network,
        metadata={
            "configured_via": "tui-builder",
            "ui_log_level": values.log_level,
        },
    )


def _validate_builder_values(
    values: BuilderValues,
) -> None:
    if not values.defender_id:
        raise ValueError(
            "Selecione um Defender."
        )

    if not values.protocol_id:
        raise ValueError(
            "Selecione um protocolo."
        )

    if values.mode in {
        ExperimentMode.TRAIN,
        ExperimentMode.SIMULATE,
    }:
        if not values.dataset_id:
            raise ValueError(
                "Selecione um Dataset."
            )

        if not _clean(
            values.dataset_source
        ):
            raise ValueError(
                "Informe o arquivo do Dataset."
            )

        if not values.attacker_id:
            raise ValueError(
                "Selecione um Attacker."
            )

    if values.mode == ExperimentMode.SIMULATE:
        if not _clean(
            values.attacker_source
        ):
            raise ValueError(
                "SIMULATE exige checkpoint "
                "do Attacker."
            )

        if not _clean(
            values.defender_source
        ):
            raise ValueError(
                "SIMULATE exige checkpoint "
                "do Defender."
            )

        if not _clean(
            values.preprocessor_source
        ):
            raise ValueError(
                "SIMULATE exige preprocessador "
                "persistido."
            )

        if not _clean(
            values.target_ip
        ):
            raise ValueError(
                "Informe o alvo do laboratório "
                "para o Renderer."
            )

        _require_network_components(
            values
        )

    if values.mode == ExperimentMode.OBSERVE:
        if not _clean(
            values.defender_source
        ):
            raise ValueError(
                "OBSERVE exige checkpoint "
                "do Defender."
            )

        if not _clean(
            values.preprocessor_source
        ):
            raise ValueError(
                "OBSERVE exige preprocessador "
                "persistido."
            )

        if not values.capture_id:
            raise ValueError(
                "OBSERVE exige Capture."
            )

        if not values.extractor_id:
            raise ValueError(
                "OBSERVE exige Extractor."
            )


def _require_network_components(
    values: BuilderValues,
) -> None:
    missing = []

    if not values.renderer_id:
        missing.append("Renderer")

    if not values.network_backend_id:
        missing.append(
            "NetworkBackend"
        )

    if not values.capture_id:
        missing.append("Capture")

    if not values.extractor_id:
        missing.append("Extractor")

    if missing:
        raise ValueError(
            "Componentes de rede ausentes: "
            + ", ".join(missing)
        )


def _clean(
    value: str | None,
) -> str | None:
    if value is None:
        return None

    stripped = value.strip()

    return stripped or None
