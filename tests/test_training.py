import numpy as np

from qfm.training import train_ansatz


def _toy_dataset(n=16, n_qubits=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, np.pi, size=(n, n_qubits))
    y = (X[:, 0] > np.pi / 2).astype(float)
    return X, y


def test_train_ansatz_returns_expected_shapes():
    X, y = _toy_dataset()
    params, loss_history = train_ansatz(X, y, n_qubits=2, layers=2, epochs=3, batch_size=8)

    assert params.shape == (2, 2, 3)
    assert len(loss_history) == 3


def test_train_ansatz_infers_n_qubits_from_X():
    X, y = _toy_dataset(n_qubits=3)
    params, _ = train_ansatz(X, y, layers=1, epochs=2, batch_size=8)

    assert params.shape == (1, 3, 3)


def test_train_ansatz_loss_is_finite_and_non_negative():
    X, y = _toy_dataset()
    _, loss_history = train_ansatz(X, y, n_qubits=2, layers=2, epochs=4, batch_size=8)

    assert all(np.isfinite(loss_history))
    assert all(l >= 0 for l in loss_history)


def test_train_ansatz_is_deterministic_given_seed():
    X, y = _toy_dataset()
    params_a, _ = train_ansatz(X, y, n_qubits=2, layers=1, epochs=3, batch_size=8, seed=42)
    params_b, _ = train_ansatz(X, y, n_qubits=2, layers=1, epochs=3, batch_size=8, seed=42)

    assert np.allclose(params_a, params_b)


def test_train_ansatz_accepts_full_batch_when_batch_size_none():
    X, y = _toy_dataset(n=10)
    params, loss_history = train_ansatz(X, y, n_qubits=2, layers=1, epochs=2, batch_size=None)

    assert params.shape == (1, 2, 3)
    assert len(loss_history) == 2
