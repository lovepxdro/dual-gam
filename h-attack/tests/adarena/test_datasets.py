import numpy as np
import pandas as pd
import pytest

from adarena.core.components import (
    DataRepresentation,
)

from adarena.datasets.cicids2017 import (
    CICIDS2017Dataset,
)

from gan.preprocessing import (
    Preprocessador,
)


def _dataframe():
    return pd.DataFrame(
        {
            "f0": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],

            "f1": [
                10.0,
                np.inf,
                30.0,
                40.0,
            ],

            "Label": [
                "BENIGN",
                "DDoS",
                "PortScan",
                "DDoS",
            ],
        }
    )


def _mock_parquet(
    monkeypatch,
):
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda _: _dataframe().copy(),
    )


def test_cicids_adapter_normaliza_labels_binarios(
    monkeypatch,
):
    _mock_parquet(
        monkeypatch
    )

    dataset = (
        CICIDS2017Dataset()
        .load(
            "dataset.parquet"
        )
    )

    assert (
        dataset.schema.representation
        == DataRepresentation.FLOW_FEATURES
    )

    assert (
        dataset.schema.feature_names
        == (
            "f0",
            "f1",
        )
    )

    np.testing.assert_array_equal(
        dataset.y,
        [
            0.0,
            1.0,
            1.0,
            1.0,
        ],
    )

    assert np.isfinite(
        dataset.X
    ).all()

    assert (
        dataset.metadata[
            "filtered_samples"
        ]
        == 0
    )


def test_cicids_adapter_pode_selecionar_um_ataque(
    monkeypatch,
):
    _mock_parquet(
        monkeypatch
    )

    dataset = CICIDS2017Dataset(
        attack_labels=(
            "DDoS",
        )
    ).load(
        "dataset.parquet"
    )

    np.testing.assert_array_equal(
        dataset.y,
        [
            0.0,
            1.0,
            1.0,
        ],
    )

    assert (
        dataset.metadata[
            "filtered_samples"
        ]
        == 1
    )

    assert (
        dataset.metadata[
            "attack_labels"
        ]
        == [
            "DDoS"
        ]
    )


def test_cicids_adapter_rejeita_classe_inexistente(
    monkeypatch,
):
    _mock_parquet(
        monkeypatch
    )

    with pytest.raises(
        ValueError,
        match="não encontradas",
    ):
        CICIDS2017Dataset(
            attack_labels=(
                "BruteForce",
            )
        ).load(
            "dataset.parquet"
        )


def test_preprocessador_adota_schema_do_dataset(
    monkeypatch,
):
    _mock_parquet(
        monkeypatch
    )

    dataset = CICIDS2017Dataset(
        attack_labels=(
            "DDoS",
        )
    ).load(
        "dataset.parquet"
    )

    prep = Preprocessador()

    prep.configurar_dataset(
        dataset
    )

    assert prep.feature_names == [
        "f0",
        "f1",
    ]

    assert (
        prep
        .label_encoder
        .classes_
        .tolist()
        == [
            "benign",
            "attack",
        ]
    )
