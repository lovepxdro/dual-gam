import json

import pytest

from adarena.builtin import (
    BASIC_FLOW_EXTRACTOR_ID,
    BINARY_MLP_DEFENDER_ID,
    NETWORK_OBSERVATION_PROTOCOL_ID,
    SCAPY_CAPTURE_ID,
    create_default_registry,
)

from adarena.core.config import (
    ComponentSelection,
    ExperimentConfig,
    ExperimentMode,
    NetworkSettings,
)

from adarena.core.experiment import (
    ExperimentRunner,
)

from adarena.protocols.base import (
    ProtocolResult,
)

from adarena.protocols.network_observation import (
    NetworkObservationProtocol,
)

from gan.preprocessing import (
    Preprocessador,
)


class FakePersistedPreprocessor:

    def __init__(self):
        self.feature_names = [
            "f1",
            "f2",
            "f3",
        ]

        self.saved_to = None

    def salvar(
        self,
        path,
    ) -> None:

        self.saved_to = path

        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def test_experiment_runner_executa_observe_sem_dataset(
    tmp_path,
    monkeypatch,
):

    registry = (
        create_default_registry()
    )

    models_root = (
        tmp_path
        / "models"
    )

    preprocessor_source = (
        tmp_path
        / "persisted-preprocessor"
    )

    defender_source = (
        tmp_path
        / "defender.pth"
    )

    config = ExperimentConfig(
        attacker=None,

        defender=ComponentSelection(
            BINARY_MLP_DEFENDER_ID,
            source=str(
                defender_source
            ),
        ),

        attack_dataset=None,

        protocol=ComponentSelection(
            NETWORK_OBSERVATION_PROTOCOL_ID
        ),

        mode=ExperimentMode.OBSERVE,

        device="cpu",

        output_dir=str(
            models_root
        ),

        network=NetworkSettings(
            capture=ComponentSelection(
                SCAPY_CAPTURE_ID,
                {
                    "iface": "test0",
                },
            ),

            extractor=ComponentSelection(
                BASIC_FLOW_EXTRACTOR_ID
            ),

            preprocessor_source=str(
                preprocessor_source
            ),

            capture_duration=2.0,

            packet_limit=25,

            classification_threshold=0.5,

            dry_run=True,
        ),
    )

    fake_preprocessor = (
        FakePersistedPreprocessor()
    )

    loaded_from = []

    def fake_carregar(
        cls,
        path,
    ):
        loaded_from.append(
            str(path)
        )

        return fake_preprocessor

    monkeypatch.setattr(
        Preprocessador,
        "carregar",
        classmethod(
            fake_carregar
        ),
    )

    observed = {}

    def fake_protocol_run(
        self,
        context,
    ):

        observed[
            "context"
        ] = context

        return ProtocolResult(
            final_metrics={
                "runner_observe": True,
            },

            artifacts={
                "observation": (
                    str(
                        context.run_dir
                        / "metrics"
                        / "observation.json"
                    )
                ),
            },
        )

    monkeypatch.setattr(
        NetworkObservationProtocol,
        "run",
        fake_protocol_run,
    )

    def fail_preprocessor_factory():
        pytest.fail(
            "OBSERVE não deve criar "
            "um preprocessador novo"
        )

    runner = ExperimentRunner(
        config=config,
        registry=registry,
        preprocessor_factory=(
            fail_preprocessor_factory
        ),
    )

    def fail_load_dataset():
        pytest.fail(
            "OBSERVE não deve "
            "carregar dataset"
        )

    monkeypatch.setattr(
        runner,
        "_load_dataset",
        fail_load_dataset,
    )

    result = runner.run()

    assert loaded_from == [
        str(
            preprocessor_source
        )
    ]

    assert (
        result.final_metrics[
            "runner_observe"
        ]
        is True
    )

    context = observed[
        "context"
    ]

    assert (
        context.dataset_data
        is None
    )

    assert (
        context.X_train
        is None
    )

    assert (
        context.X_val
        is None
    )

    assert (
        context.X_test
        is None
    )

    assert (
        context.y_train
        is None
    )

    assert (
        context.y_val
        is None
    )

    assert (
        context.y_test
        is None
    )

    assert (
        context.preprocessor
        is fake_preprocessor
    )

    assert (
        context.config.mode
        == ExperimentMode.OBSERVE
    )

    assert (
        fake_preprocessor
        .saved_to
        == (
            result.run_dir
            / "preprocessador"
        )
    )

    config_path = (
        result.run_dir
        / "config_execucao.json"
    )

    assert config_path.exists()

    with open(
        config_path,
        encoding="utf-8",
    ) as file:
        snapshot = json.load(
            file
        )

    assert (
        snapshot["mode"]
        == "observe"
    )

    assert (
        snapshot["dataset"]
        is None
    )

    assert (
        snapshot[
            "components"
        ][
            "attacker"
        ]
        is None
    )

    assert (
        snapshot[
            "components"
        ][
            "protocol"
        ]
        == NETWORK_OBSERVATION_PROTOCOL_ID
    )

    assert (
        snapshot[
            "dados"
        ][
            "input_dim"
        ]
        == 3
    )

    assert (
        snapshot[
            "dados"
        ][
            "validation_size"
        ]
        is None
    )

    assert (
        snapshot[
            "dados"
        ][
            "test_size"
        ]
        is None
    )

    assert (
        snapshot[
            "network"
        ][
            "capture_duration"
        ]
        == 2.0
    )

    latest = (
        models_root
        / "latest"
    )

    assert latest.is_symlink()

    assert (
        latest.resolve()
        == result.run_dir.resolve()
    )
