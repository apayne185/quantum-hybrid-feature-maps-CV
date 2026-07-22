"""Quantum feature maps: encode classical feature vectors into quantum states.

Shared by notebooks/quantum_pipeline.ipynb and notebooks/quantum_feature_maps_tests.ipynb
so the circuit definitions live in one tested place instead of being copy-pasted.
"""

import pennylane as qml


def zz_feature_map(x, wires, reps=1):
    """ZZ feature map: Hadamard layer + RZ single-qubit encoding + pairwise ZZ entanglers.

    Not built into PennyLane, so implemented manually with Hadamard/RZ/CNOT gates.

    Args:
        x: classical feature vector, one value per wire.
        wires: qubit indices to encode onto.
        reps: number of repetitions of the encoding layer.
    """
    n_qubits = len(wires)

    for _ in range(reps):
        for i in wires:
            qml.Hadamard(wires=i)

        for i, wire in enumerate(wires):
            qml.RZ(2 * x[i], wires=wire)

        for i in range(n_qubits - 1):
            qml.CNOT(wires=[wires[i], wires[i + 1]])
            qml.RZ(2 * x[i] * x[i + 1], wires=wires[i + 1])
            qml.CNOT(wires=[wires[i], wires[i + 1]])


def basis_encoding(x, wires):
    """Basis encoding: binarized features flip qubits directly (computational basis)."""
    qml.BasisEmbedding(features=x, wires=wires)


def angle_encoding(x, wires, rotation="Z"):
    """Angle encoding: each feature becomes a single-qubit rotation angle."""
    qml.AngleEmbedding(features=x, wires=wires, rotation=rotation)


def amplitude_encoding(x, wires, normalize=True):
    """Amplitude encoding: features become the amplitudes of the quantum state."""
    qml.AmplitudeEmbedding(features=x, wires=wires, normalize=normalize)
