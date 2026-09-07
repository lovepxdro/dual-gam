import numpy as np

from adarena.core import (
    ComponentKind,
    ComponentSelection,
    ComponentSpec,
    DataRepresentation,
    ExperimentConfig,
    ExperimentRunner,
)

from adarena.core.registry import (
    ComponentRegistry,
)

from adarena.datasets.base import (
    DatasetData,
    DatasetSchema,
)

from adarena.protocols.base import (
    ExperimentProtocol,
    ProtocolResult,
)


class FakeDataset:

    def load(
        self,
        source,
    ) -> DatasetData:

        X = np.arange(
            80,
            dtype=np.float32,
        ).reshape(
            40,
            2,
        )

        y = np.asarray(
            [
                i % 2
                for i
                in range(40)
            ],
            dtype=np.float32,
        )

        return DatasetData(
            X=X,
            y=y,

            schema=DatasetSchema(
                feature_names=(
                    "f0",
                    "f1",
                ),

                representation=(
                    DataRepresentation
                    .FLOW_FEATURES
                ),
            ),

            metadata={
                "fake": True,
            },
        )


class FakeProtocol(
    ExperimentProtocol
):

    def run(
        self,
        context,
    ) -> ProtocolResult:

        assert (
            len(context.X_train)
            > 0
        )

        assert (
            len(context.X_val)
            > 0
        )

        assert (
            len(context.X_test)
            > 0
        )

        return ProtocolResult(
            history={
                "executed": True,
            },

            final_metrics={
                "accuracy": 1.0,
            },

            artifacts={},
        )


def _registry() -> ComponentRegistry:

    registry = ComponentRegistry()

    registry.register(
        ComponentSpec(
            component_id=(
                "test.attacker"
            ),

            kind=(
                ComponentKind.ATTACKER
            ),

            name="Fake attacker",

            input_representation=(
                DataRepresentation
                .FLOW_FEATURES
            ),

            output_representation=(
                DataRepresentation
                .FLOW_FEATURES
            ),
        ),

        object,
    )

    registry.register(
        ComponentSpec(
            component_id=(
                "test.defender"
            ),

            kind=(
                ComponentKind.DEFENDER
            ),

            name="Fake defender",

            input_representation=(
                DataRepresentation
                .FLOW_FEATURES
            ),

            output_representation=(
                DataRepresentation
                .BINARY_CLASSIFICATION
            ),
        ),

        object,
    )

    registry.register(
        ComponentSpec(
            component_id=(
                "test.dataset"
            ),

            kind=(
                ComponentKind.DATASET
            ),

            name="Fake dataset",

            output_representation=(
                DataRepresentation
                .FLOW_FEATURES
            ),
        ),

        lambda **_: FakeDataset(),
    )

    registry.register(
        ComponentSpec(
            component_id=(
                "test.protocol"
            ),

            kind=(
                ComponentKind
                .EXPERIMENT_PROTOCOL
            ),

            name="Fake protocol",
        ),

        lambda **_: FakeProtocol(),
    )

    return registry


def _config(
    tmp_path,
) -> ExperimentConfig:

    return ExperimentConfig(
        attacker=ComponentSelection(
            "test.attacker"
        ),

        defender=ComponentSelection(
            "test.defender"
        ),

        attack_dataset=(
            ComponentSelection(
                "test.dataset",

                source=(
                    "memory://dataset"
                ),
            )
        ),

        protocol=ComponentSelection(
            "test.protocol"
        ),

        seed=42,

        output_dir=str(
            tmp_path
        ),
    )


def test_runner_executa_protocolo(
    tmp_path,
):
    registry = _registry()

    runner = ExperimentRunner(
        config=_config(
            tmp_path
        ),
        registry=registry,
    )

    result = runner.run()

    assert (
        result
        .history[
            "executed"
        ]
        is True
    )

    assert (
        result
        .final_metrics[
            "accuracy"
        ]
        == 1.0
    )

    assert (
        result.run_dir.exists()
    )


def test_runner_persiste_preprocessador(
    tmp_path,
):
    runner = ExperimentRunner(
        config=_config(
            tmp_path
        ),
        registry=_registry(),
    )

    result = runner.run()

    prep_dir = (
        result.run_dir
        / "preprocessador"
    )

    assert (
        prep_dir
        / "scaler.pkl"
    ).exists()

    assert (
        prep_dir
        / "feature_names.pkl"
    ).exists()


def test_runner_atualiza_latest(
    tmp_path,
):
    runner = ExperimentRunner(
        config=_config(
            tmp_path
        ),
        registry=_registry(),
    )

    result = runner.run()

    latest = (
        tmp_path
        / "latest"
    )

    assert latest.is_symlink()

    assert (
        latest.resolve()
        == result.run_dir.resolve()
    )
