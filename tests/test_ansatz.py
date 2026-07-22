import numpy as np
import pennylane as qml
import pytest

from qfm.ansatz import var_ansatz


@pytest.fixture
def device():
    return qml.device("default.qubit", wires=3)


def test_var_ansatz_produces_normalized_state(device):
    params = np.random.default_rng(0).normal(size=(2, 3, 3))

    @qml.qnode(device)
    def circuit(params):
        var_ansatz(params)
        return qml.state()

    state = circuit(params)
    assert np.isclose(np.sum(np.abs(state) ** 2), 1.0)


def test_var_ansatz_zero_params_is_identity(device):
    params = np.zeros((2, 3, 3))

    @qml.qnode(device)
    def circuit(params):
        var_ansatz(params)
        return qml.state()

    state = circuit(params)
    expected = np.zeros(8)
    expected[0] = 1.0
    assert np.allclose(state, expected)


def test_var_ansatz_more_layers_changes_state(device):
    rng = np.random.default_rng(1)
    params_1_layer = rng.normal(size=(1, 3, 3))
    params_2_layers = np.concatenate([params_1_layer, rng.normal(size=(1, 3, 3))], axis=0)

    @qml.qnode(device)
    def circuit(params):
        var_ansatz(params)
        return qml.state()

    assert not np.allclose(circuit(params_1_layer), circuit(params_2_layers))
