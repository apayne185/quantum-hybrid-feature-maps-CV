import numpy as np
import pennylane as qml
import pytest

from qfm.feature_maps import (
    zz_feature_map,
    basis_encoding,
    angle_encoding,
    amplitude_encoding,
)


@pytest.fixture
def device():
    return qml.device("default.qubit", wires=4)


def test_zz_feature_map_produces_normalized_state(device):
    x = np.array([0.1, 0.2, 0.3, 0.4])

    @qml.qnode(device)
    def circuit(x):
        zz_feature_map(x, wires=range(4), reps=1)
        return qml.state()

    state = circuit(x)
    assert np.isclose(np.sum(np.abs(state) ** 2), 1.0)


def test_zz_feature_map_deterministic(device):
    x = np.array([0.5, -0.3, 0.1, 0.9])

    @qml.qnode(device)
    def circuit(x):
        zz_feature_map(x, wires=range(4), reps=1)
        return qml.state()

    assert np.allclose(circuit(x), circuit(x))


def test_zz_feature_map_zero_input_is_uniform_superposition(device):
    x = np.zeros(4)

    @qml.qnode(device)
    def circuit(x):
        zz_feature_map(x, wires=range(4), reps=1)
        return qml.probs(wires=range(4))

    probs = circuit(x)
    assert np.allclose(probs, np.full(16, 1 / 16))


def test_zz_feature_map_more_reps_changes_state(device):
    x = np.array([0.7, 0.2, -0.4, 0.1])

    def circuit_factory(reps):
        @qml.qnode(device)
        def circuit(x):
            zz_feature_map(x, wires=range(4), reps=reps)
            return qml.state()

        return circuit

    state_1 = circuit_factory(1)(x)
    state_2 = circuit_factory(2)(x)
    assert not np.allclose(state_1, state_2)


def test_basis_encoding_flips_qubits_matching_input(device):
    x = [1, 0, 1, 0]

    @qml.qnode(device)
    def circuit(x):
        basis_encoding(x, wires=range(4))
        return qml.probs(wires=range(4))

    probs = circuit(x)
    expected_index = int("1010", 2)
    assert np.isclose(probs[expected_index], 1.0)


def test_angle_encoding_normalized(device):
    x = np.array([0.1, 0.2, 0.3, 0.4])

    @qml.qnode(device)
    def circuit(x):
        angle_encoding(x, wires=range(4), rotation="Z")
        return qml.state()

    state = circuit(x)
    assert np.isclose(np.sum(np.abs(state) ** 2), 1.0)


def test_amplitude_encoding_matches_normalized_input():
    dev = qml.device("default.qubit", wires=2)
    x = np.array([1.0, 1.0, 1.0, 1.0])

    @qml.qnode(dev)
    def circuit(x):
        amplitude_encoding(x, wires=range(2), normalize=True)
        return qml.state()

    state = circuit(x)
    assert np.allclose(np.abs(state), 0.5)
