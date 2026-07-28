"""Generate a dataset whose labels come from a fixed, hidden quantum circuit.

Follows the construction from Havlicek et al. (2019), "Supervised learning with
quantum-enhanced feature spaces" (Nature 567, 209-212): labels are defined by
evaluating a *fixed, random* circuit built from the same feature-map + ansatz
family this repo's quantum classifier learns from, on random input points.

Why this is a meaningful test and not just "hard-looking data": a quantum model
drawn from that same circuit family can, in principle, represent the exact
function that generated the labels. A classical model only ever sees the raw
input vector x - it has no access to the interference structure the labeling
circuit computed internally, so it has to approximate a decision boundary shaped
by entanglement using surface statistics of x alone. That asymmetry is the actual
mechanism behind a quantum-kernel advantage, not an assumption that quantum
"looks hard so it must be strong" - the MNIST case study elsewhere in this repo
is the demonstration that hard-looking is not sufficient by itself.

Points too close to the labeling circuit's decision boundary are dropped (a
margin), matching the original construction, so the two classes are cleanly
separable in the quantum feature space by design - though not necessarily easy
for a *trained* quantum model to recover, since training still has to find
those parameters via gradient descent from a random start.
"""

import numpy as np
import pennylane as qml

from qfm.feature_maps import zz_feature_map
from qfm.ansatz import var_ansatz


def generate_quantum_engineered_dataset(
    n_samples, n_qubits, reps=2, layers=3, margin=0.3, seed=0, max_attempts_multiplier=50
):
    """Sample points and label them via a fixed random quantum circuit.

    Args:
        n_samples: number of samples to return (after margin filtering).
        n_qubits: number of qubits / input dimensions.
        reps: ZZ feature map repetitions used by the (hidden) labeling circuit -
            should match what the quantum learner will be given, so the same
            circuit family can represent the labeling function.
        layers: variational ansatz layers for the (hidden) labeling circuit.
        margin: minimum |<Z_0>| required to keep a candidate point (drops
            near-boundary points so the classes are cleanly separable in the
            quantum feature space). Larger margin = cleaner separation but more
            candidates rejected.
        seed: random seed for both the fixed labeling circuit and the sampled points.
        max_attempts_multiplier: safety cap on candidate points drawn while
            filtering for the margin, relative to n_samples.

    Returns:
        (X, y, label_params): X of shape (n_samples, n_qubits) in [0, 2*pi], y of
        shape (n_samples,) in {0, 1}, and the fixed label_params used to generate
        them (returned for inspection/reproducibility only - a real learner never
        sees these; it only ever sees X and y).

    Raises:
        RuntimeError: if fewer than n_samples candidates meet the margin within
            the attempt budget - try a smaller margin or more attempts.
    """
    rng = np.random.default_rng(seed)
    label_params = rng.standard_normal((layers, n_qubits, 3))

    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device)
    def labeling_circuit(x):
        zz_feature_map(x, wires=range(n_qubits), reps=reps)
        var_ansatz(label_params)
        return qml.expval(qml.PauliZ(0))

    max_attempts = n_samples * max_attempts_multiplier
    X_kept, y_kept = [], []
    n_attempts = 0
    batch_size = 256

    while len(X_kept) < n_samples and n_attempts < max_attempts:
        remaining_budget = max_attempts - n_attempts
        batch = rng.uniform(0, 2 * np.pi, size=(min(batch_size, remaining_budget), n_qubits))
        for x in batch:
            n_attempts += 1
            z = float(labeling_circuit(x))
            if abs(z) >= margin:
                X_kept.append(x)
                y_kept.append(1 if z > 0 else 0)
            if len(X_kept) >= n_samples:
                break

    if len(X_kept) < n_samples:
        raise RuntimeError(
            f"only found {len(X_kept)}/{n_samples} points meeting margin={margin} "
            f"after {n_attempts} attempts; try a smaller margin, more attempts, or fewer n_samples"
        )

    return np.array(X_kept), np.array(y_kept), label_params
