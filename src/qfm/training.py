"""Train variational ansatz parameters on an arbitrary binary-labeled dataset.

Generalizes the training loop in notebooks/quantum_pipeline.ipynb (which trains
against the fixed MNIST 0-vs-1 PCA features) so the same feature map + ansatz
can be fit to any dataset, not just the one this repo started with.
"""

import pennylane as qml
import pennylane.numpy as pnp
import numpy as np

from qfm.features import make_qnode


def train_ansatz(X, y, n_qubits=None, layers=2, reps=1, epochs=20, stepsize=0.1, batch_size=32, seed=0):
    """Fit ansatz parameters so the qubit-0 <Z> expectation value tracks the label.

    Args:
        X: array of shape (N, n_qubits) - classical feature vectors, expected to
            already be scaled (e.g. to [0, pi]) as the ZZ feature map expects.
        y: array of shape (N,) of binary labels - exactly 2 distinct values, numeric
            or not (e.g. "yes"/"no"). Raises ValueError otherwise. Mapped to the
            [-1, 1] range of a Pauli-Z expectation value via np.unique ordering:
            the second (sorted) class becomes +1, the first becomes -1.
        n_qubits: number of wires; inferred from X's second dimension if not given.
        layers: number of variational ansatz layers.
        reps: repetitions of the ZZ feature map encoding layer.
        epochs: number of gradient descent steps.
        stepsize: gradient descent step size.
        batch_size: examples sampled per epoch (default 32). Bounds training cost
            independent of dataset size; pass None to use the full dataset each epoch.
        seed: random seed for the initial parameters and batch sampling.

    Returns:
        (params, loss_history): trained params of shape (layers, n_qubits, 3),
        and the per-epoch MSE loss.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if n_qubits is None:
        n_qubits = X.shape[1]

    # np.unique + equality (not max/min + arithmetic) so this works for both
    # non-numeric labels (e.g. "yes"/"no") and validates binary-ness: the old
    # (y.max() + y.min()) / 2 threshold raised on string labels, and silently
    # mismapped >2 classes instead of erroring (e.g. {0, 1, 2} with midpoint 1.0
    # put both classes 0 and 1 on the same side of the threshold).
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(
            f"train_ansatz requires a binary target (exactly 2 classes), got "
            f"{len(classes)}: {classes.tolist()!r}"
        )

    rng = np.random.default_rng(seed)
    y_pm1 = np.where(y == classes[1], 1.0, -1.0)

    device = qml.device("default.qubit", wires=n_qubits)
    qnode = make_qnode(device, n_qubits, reps=reps)

    params = pnp.array(rng.standard_normal((layers, n_qubits, 3)), requires_grad=True)
    opt = qml.GradientDescentOptimizer(stepsize=stepsize)

    n = len(X)
    batch_size = min(batch_size, n) if batch_size else n

    def cost(p, X_batch, y_batch):
        preds = pnp.array([qnode(x, p)[0] for x in X_batch])
        return pnp.mean((preds - y_batch) ** 2)

    loss_history = []
    for _ in range(epochs):
        idx = rng.permutation(n)[:batch_size]
        X_batch, y_batch = X[idx], y_pm1[idx]
        params = opt.step(lambda p: cost(p, X_batch, y_batch), params)
        loss_history.append(float(cost(params, X_batch, y_batch)))

    return np.array(params), loss_history
