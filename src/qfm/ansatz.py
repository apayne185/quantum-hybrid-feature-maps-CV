"""Variational ansatz: the trainable quantum transformation applied after encoding."""

import pennylane as qml


def var_ansatz(params):
    """Layered single-qubit rotations + nearest-neighbor CNOT entanglers.

    Args:
        params: array of shape (layers, n_qubits, 3) — 3 Euler angles per qubit per layer.
    """
    layers, n_qubits, _ = params.shape

    for l in range(layers):
        for i in range(n_qubits):
            qml.Rot(params[l, i, 0], params[l, i, 1], params[l, i, 2], wires=i)

        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
