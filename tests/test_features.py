import numpy as np
import pennylane as qml

from qfm.features import get_quantum_features


def test_output_shape_matches_input_rows_and_n_qubits():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(6, 3))
    params = rng.normal(size=(2, 3, 3))

    feats = get_quantum_features(X, params)

    assert feats.shape == (6, 3)


def test_expectation_values_are_bounded():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 4))
    params = rng.normal(size=(2, 4, 3))

    feats = get_quantum_features(X, params)

    assert np.all(feats >= -1.0 - 1e-9) and np.all(feats <= 1.0 + 1e-9)


def test_infers_n_qubits_from_params_when_not_given():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, 2))
    params = rng.normal(size=(1, 2, 3))

    feats = get_quantum_features(X, params)

    assert feats.shape == (4, 2)


def test_custom_device_is_used():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3, 2))
    params = rng.normal(size=(1, 2, 3))
    device = qml.device("default.qubit", wires=2)

    feats = get_quantum_features(X, params, device=device, n_qubits=2)

    assert feats.shape == (3, 2)
