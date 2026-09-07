from __future__ import annotations

from .core.components import (
    ComponentKind,
    ComponentSpec,
    DataRepresentation,
)

from .core.registry import (
    ComponentRegistry,
)

from .datasets import (
    CICIDS2017Dataset,
)


PERTURBATION_ATTACKER_ID = (
    "adarena.perturbation_attacker"
)

BINARY_MLP_DEFENDER_ID = (
    "adarena.binary_mlp_defender"
)

CICIDS2017_DATASET_ID = (
    "adarena.cicids2017"
)

SCAPY_NETWORK_BACKEND_ID = (
    "adarena.scapy_backend"
)

ADVERSARIAL_PROTOCOL_ID = (
    "adarena.adversarial_training"
)

CICIDS2017_RENDERER_ID = (
    "adarena.cicids2017_renderer"
)

SCAPY_NETWORK_BACKEND_ID = (
    "adarena.scapy_backend"
)

SCAPY_CAPTURE_ID = (
    "adarena.scapy_capture"
)

BASIC_FLOW_EXTRACTOR_ID = (
    "adarena.basic_flow_extractor"
)


def _create_perturbation_attacker(
    noise_dim: int = 32,
    output_dim: int = 77,
    **_,
):
    from gan.models import Atacante

    return Atacante(
        noise_dim=noise_dim,
        output_dim=output_dim,
    )


def _create_binary_mlp_defender(
    input_dim: int = 77,
    **_,
):
    from gan.models import Defensor

    return Defensor(
        input_dim=input_dim,
    )


def _create_cicids2017_dataset(
    label_column: str = "Label",
    benign_labels=(
        "BENIGN",
    ),
    attack_labels=None,
    **_,
):
    return CICIDS2017Dataset(
        label_column=label_column,

        benign_labels=tuple(
            benign_labels
        ),

        attack_labels=(
            attack_labels
        ),
    )

def _create_cicids2017_renderer(
    preprocessor,
    target_ip: str,
    target_port: int = 80,
    consistency_tolerance: float = 0.75,
    **_,
):
    from .network.adapters import (
        TranslatorRenderer,
    )

    return TranslatorRenderer(
        preprocessor=preprocessor,
        target_ip=target_ip,
        target_port=target_port,
        consistency_tolerance=(
            consistency_tolerance
        ),
    )


def _create_scapy_network_backend(
    iface=None,
    dry_run: bool = False,
    require_private_target: bool = True,
    **_,
):
    from .network.adapters import (
        SenderNetworkBackend,
    )

    return SenderNetworkBackend(
        iface=iface,
        dry_run=dry_run,
        require_private_target=(
            require_private_target
        ),
    )


def _create_adversarial_protocol(
    **_,
):
    # Import tardio para evitar ciclo:
    #
    # builtin
    #   → protocols.adversarial
    #   → gan.trainer
    #   → builtin
    from .protocols.adversarial import (
        AdversarialTrainingProtocol,
    )

    return (
        AdversarialTrainingProtocol()
    )


def register_builtin_components(
    registry: ComponentRegistry,
) -> ComponentRegistry:

    registry.register(
        ComponentSpec(
            component_id=(
                PERTURBATION_ATTACKER_ID
            ),

            kind=(
                ComponentKind.ATTACKER
            ),

            name=(
                "Perturbation Attacker"
            ),

            version="1.7",

            description=(
                "Atacante da linha 1.x "
                "baseado em perturbação "
                "adversarial."
            ),

            input_representation=(
                DataRepresentation
                .FLOW_FEATURES
            ),

            output_representation=(
                DataRepresentation
                .FLOW_FEATURES
            ),

            tags=(
                "builtin",
                "pytorch",
                "adversarial",
                "legacy-v1",
            ),
        ),

        _create_perturbation_attacker,
    )

    registry.register(
        ComponentSpec(
            component_id=(
                BINARY_MLP_DEFENDER_ID
            ),

            kind=(
                ComponentKind.DEFENDER
            ),

            name=(
                "Binary MLP Defender"
            ),

            version="1.7",

            description=(
                "Classificador binário "
                "de referência da ADArena."
            ),

            input_representation=(
                DataRepresentation
                .FLOW_FEATURES
            ),

            output_representation=(
                DataRepresentation
                .BINARY_CLASSIFICATION
            ),

            tags=(
                "builtin",
                "pytorch",
                "binary-classifier",
                "legacy-v1",
            ),
        ),

        _create_binary_mlp_defender,
    )

    registry.register(
        ComponentSpec(
            component_id=(
                CICIDS2017_DATASET_ID
            ),

            kind=(
                ComponentKind.DATASET
            ),

            name="CIC-IDS2017",

            version="2.0",

            description=(
                "Adapter CIC-IDS2017 "
                "para flow features."
            ),

            output_representation=(
                DataRepresentation
                .FLOW_FEATURES
            ),

            tags=(
                "builtin",
                "dataset",
                "flow-features",
                "cicids2017",
            ),

            metadata={
                "label_column": "Label",
            },
        ),

        _create_cicids2017_dataset,
    )

    registry.register(
        ComponentSpec(
            component_id=(
                ADVERSARIAL_PROTOCOL_ID
            ),

            kind=(
                ComponentKind
                .EXPERIMENT_PROTOCOL
            ),

            name=(
                "Adversarial Training Protocol"
            ),

            version="1.0",

            description=(
                "Protocolo de treinamento "
                "adaptativo utilizado pela "
                "linha 1.x."
            ),

            tags=(
                "builtin",
                "training",
                "adversarial",
            ),
        ),

        _create_adversarial_protocol,
    )

    registry.register(
        ComponentSpec(
            component_id=(
                CICIDS2017_RENDERER_ID
            ),

            kind=(
                ComponentKind.RENDERER
            ),

            name=(
                "CIC-IDS2017 Flow Renderer"
            ),

            version="2.0",

            description=(
                "Converte flow features produzidas "
                "pelos modelos atuais em parâmetros "
                "materializáveis pelo backend de rede."
            ),

            input_representation=(
                DataRepresentation
                .FLOW_FEATURES
            ),

            output_representation=(
                DataRepresentation
                .ATTACK_PARAMS
            ),

            tags=(
                "builtin",
                "renderer",
                "flow-features",
                "cicids2017",
                "legacy-translator",
            ),
        ),

        _create_cicids2017_renderer,
    )


    registry.register(
        ComponentSpec(
            component_id=(
                SCAPY_NETWORK_BACKEND_ID
            ),

            kind=(
                ComponentKind
                .NETWORK_BACKEND
            ),

            name=(
                "Scapy Network Backend"
            ),

            version="2.0",

            description=(
                "Backend de execução utilizado "
                "no ambiente experimental isolado."
            ),

            input_representation=(
                DataRepresentation
                .ATTACK_PARAMS
            ),

            output_representation=(
                DataRepresentation
                .NETWORK_RESULT
            ),

            tags=(
                "builtin",
                "network",
                "scapy",
                "laboratory",
            ),
        ),

        _create_scapy_network_backend,
    )

    registry.register(
        ComponentSpec(
            component_id=(
                SCAPY_CAPTURE_ID
            ),

            kind=(
                ComponentKind.CAPTURE
            ),

            name=(
                "Scapy Packet Capture"
            ),

            version="2.0",

            description=(
                "Captura passiva de pacotes "
                "no ambiente experimental."
            ),

            output_representation=(
                DataRepresentation
                .PACKET_RECORDS
            ),

            tags=(
                "builtin",
                "capture",
                "scapy",
                "laboratory",
            ),
        ),

        _create_scapy_capture,
    )


    registry.register(
        ComponentSpec(
            component_id=(
                BASIC_FLOW_EXTRACTOR_ID
            ),

            kind=(
                ComponentKind.EXTRACTOR
            ),

            name=(
                "Basic Flow Extractor"
            ),

            version="2.0",

            description=(
                "Reconstrói estatísticas "
                "auditáveis de fluxo a partir "
                "de pacotes capturados."
            ),

            input_representation=(
                DataRepresentation
                .PACKET_RECORDS
            ),

            output_representation=(
                DataRepresentation
                .FLOW_FEATURES
            ),

            tags=(
                "builtin",
                "extractor",
                "flow-features",
            ),
        ),

        _create_basic_flow_extractor,
    )

    return registry


def _create_scapy_capture(
    iface=None,
    bpf_filter=None,
    **_,
):
    from .network.capture import (
        ScapyPacketCapture,
    )

    return ScapyPacketCapture(
        iface=iface,
        bpf_filter=bpf_filter,
    )


def _create_basic_flow_extractor(
    **_,
):
    from .network.extractor import (
        BasicFlowExtractor,
    )

    return BasicFlowExtractor()


def create_default_registry() -> (
    ComponentRegistry
):
    return (
        register_builtin_components(
            ComponentRegistry()
        )
    )
