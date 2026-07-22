"""Turn a quantum feature map + variational ansatz into classical feature vectors."""

import numpy as np
import pennylane as qml

from qfm.feature_maps import zz_feature_map
from qfm.ansatz import var_ansatz


def make_qnode(device, n_qubits, reps=1, shots=None):
    """Build the QNode: ZZ feature map encoding -> variational ansatz -> per-qubit <Z>.

    Shots are applied via the `qml.set_shots` transform rather than at device
    construction time, since setting shots on the device directly is deprecated.
    """

    @qml.qnode(device)
    def circuit(x, params):
        zz_feature_map(x, wires=range(n_qubits), reps=reps)
        var_ansatz(params)
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return qml.set_shots(circuit, shots=shots)


def get_quantum_features(X, params, device=None, n_qubits=None, reps=1, shots=None):
    """Map each row of X through the feature-map + ansatz circuit to <Z> expectation values.

    Args:
        X: array of shape (N, n_qubits) — classical feature vectors (e.g. PCA output).
        params: ansatz parameters, shape (layers, n_qubits, 3).
        device: a PennyLane device; defaults to a `default.qubit` simulator sized to n_qubits.
        n_qubits: number of qubits/wires to use; inferred from `params` if not given.
        reps: repetitions of the ZZ feature map encoding layer.
        shots: None for analytic expectation values, or a shot count to sample instead.

    Returns:
        np.ndarray of shape (N, n_qubits) of per-qubit PauliZ expectation values.
    """
    if n_qubits is None:
        n_qubits = params.shape[1]
    if device is None:
        device = qml.device("default.qubit", wires=n_qubits)

    circuit = make_qnode(device, n_qubits, reps=reps, shots=shots)
    return np.array([circuit(x, params) for x in X])
