import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from gan.preprocessing import Preprocessador


def _dataset(n_per_class: int = 50):
    # Valores únicos e determinísticos, com duas classes balanceadas.
    n = n_per_class * 2
    X = np.column_stack([
        np.arange(n, dtype=np.float32),
        np.arange(n, dtype=np.float32) * 10.0 + 3.0,
        np.linspace(-5.0, 5.0, n, dtype=np.float32),
    ])
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=np.float32)
    return X, y


def test_remove_duplicatas_antes_do_split():
    X, y = _dataset()
    X = np.vstack([X, X[0], X[-1]])
    y = np.concatenate([y, [y[0], y[-1]]]).astype(np.float32)

    prep = Preprocessador()
    prep.feature_names = ["f0", "f1", "f2"]

    result = prep.split_e_normalizar(X, y, random_state=42)
    X_train, X_val, X_test, *_ = result

    assert prep.audit_info["duplicatas_detectadas"] == 2
    assert prep.audit_info["duplicatas_removidas"] == 2
    assert prep.audit_info["amostras_pos_deduplicacao"] == 100
    assert len(X_train) + len(X_val) + len(X_test) == 100


def test_scaler_e_ajustado_exclusivamente_no_treino():
    X, y = _dataset()
    prep = Preprocessador()
    prep.feature_names = ["f0", "f1", "f2"]

    prep.split_e_normalizar(X, y, random_state=42)

    # Reproduz exatamente o split bruto para descobrir quais amostras
    # deveriam participar do fit do scaler.
    X_temp, _, y_temp, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train_raw, _, _, _ = train_test_split(
        X_temp,
        y_temp,
        test_size=0.1 / 0.8,
        random_state=42,
        stratify=y_temp,
    )

    np.testing.assert_allclose(
        prep.scaler.mean_,
        X_train_raw.mean(axis=0, dtype=np.float64),
        rtol=0,
        atol=1e-7,
    )


def test_split_e_reprodutivel_com_mesmo_random_state():
    X, y = _dataset()

    prep_a = Preprocessador()
    prep_a.feature_names = ["f0", "f1", "f2"]
    result_a = prep_a.split_e_normalizar(X, y, random_state=42)

    prep_b = Preprocessador()
    prep_b.feature_names = ["f0", "f1", "f2"]
    result_b = prep_b.split_e_normalizar(X, y, random_state=42)

    for array_a, array_b in zip(result_a, result_b):
        np.testing.assert_array_equal(array_a, array_b)


def test_auditoria_interrompe_quando_ha_intersecao():
    prep = Preprocessador()
    prep.feature_names = ["f0", "f1"]

    X_train = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    y_train = np.array([0.0, 1.0], dtype=np.float32)
    X_val = np.array([[5.0, 6.0]], dtype=np.float32)
    y_val = np.array([0.0], dtype=np.float32)
    X_test = np.array([[1.0, 2.0]], dtype=np.float32)
    y_test = np.array([0.0], dtype=np.float32)

    with pytest.raises(RuntimeError, match="Data leakage detectado"):
        prep.auditar_isolamento_triplo(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
        )


def test_salvar_carregar_preserva_estado(tmp_path):
    X, y = _dataset()
    prep = Preprocessador()
    prep.feature_names = ["f0", "f1", "f2"]
    prep.label_encoder.fit(["Benign", "DDoS"])
    prep.split_e_normalizar(X, y, random_state=42)

    out = tmp_path / "preprocessador"
    prep.salvar(out)
    restored = Preprocessador.carregar(out)

    assert restored.feature_names == prep.feature_names
    assert restored.audit_info == prep.audit_info
    np.testing.assert_allclose(restored.scaler.mean_, prep.scaler.mean_)
    np.testing.assert_allclose(restored.scaler.scale_, prep.scaler.scale_)
    np.testing.assert_array_equal(restored.label_encoder.classes_, prep.label_encoder.classes_)


def test_desnormalizar_rejeita_dimensao_incompativel():
    X, y = _dataset()
    prep = Preprocessador()
    prep.feature_names = ["f0", "f1", "f2"]
    prep.split_e_normalizar(X, y)

    with pytest.raises(ValueError, match="Número de features incompatível"):
        prep.desnormalizar(np.zeros((1, 2), dtype=np.float32))
