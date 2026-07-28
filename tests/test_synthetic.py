import numpy as np
import pytest

from qfm.synthetic import generate_quantum_engineered_dataset


def test_returns_expected_shapes():
    X, y, label_params = generate_quantum_engineered_dataset(
        n_samples=20, n_qubits=2, reps=1, layers=2, margin=0.1, seed=0
    )
    assert X.shape == (20, 2)
    assert y.shape == (20,)
    assert label_params.shape == (2, 2, 3)


def test_labels_are_binary():
    _, y, _ = generate_quantum_engineered_dataset(n_samples=20, n_qubits=2, reps=1, layers=2, margin=0.1, seed=0)
    assert set(np.unique(y)) <= {0, 1}


def test_points_are_within_domain():
    X, _, _ = generate_quantum_engineered_dataset(n_samples=20, n_qubits=2, reps=1, layers=2, margin=0.1, seed=0)
    assert np.all(X >= 0) and np.all(X <= 2 * np.pi)


def test_is_deterministic_given_seed():
    X_a, y_a, params_a = generate_quantum_engineered_dataset(
        n_samples=15, n_qubits=2, reps=1, layers=2, margin=0.1, seed=42
    )
    X_b, y_b, params_b = generate_quantum_engineered_dataset(
        n_samples=15, n_qubits=2, reps=1, layers=2, margin=0.1, seed=42
    )
    assert np.array_equal(X_a, X_b)
    assert np.array_equal(y_a, y_b)
    assert np.array_equal(params_a, params_b)


def test_different_seeds_give_different_labeling_circuits():
    _, _, params_a = generate_quantum_engineered_dataset(n_samples=10, n_qubits=2, layers=2, margin=0.1, seed=1)
    _, _, params_b = generate_quantum_engineered_dataset(n_samples=10, n_qubits=2, layers=2, margin=0.1, seed=2)
    assert not np.array_equal(params_a, params_b)


def test_raises_when_margin_unreachable_within_attempt_budget():
    with pytest.raises(RuntimeError, match="only found"):
        generate_quantum_engineered_dataset(
            n_samples=1000, n_qubits=2, margin=0.999, seed=0, max_attempts_multiplier=2
        )
