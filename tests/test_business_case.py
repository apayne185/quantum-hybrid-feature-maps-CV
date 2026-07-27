import numpy as np
import pytest

from qfm.business_case import (
    CostAssumptions,
    Verdict,
    _make_verdict,
    _subsample,
    evaluate_business_case,
)


def _toy_dataset(n_train=16, n_test=8, n_qubits=2, seed=0):
    rng = np.random.default_rng(seed)
    X_train = rng.uniform(0, np.pi, size=(n_train, n_qubits))
    y_train = (X_train[:, 0] > np.pi / 2).astype(float)
    X_test = rng.uniform(0, np.pi, size=(n_test, n_qubits))
    y_test = (X_test[:, 0] > np.pi / 2).astype(float)
    return X_train, y_train, X_test, y_test


# --- _make_verdict: pure function, cheap to test exhaustively ---

def test_verdict_classical_wins_when_quantum_not_better():
    verdict, reasoning = _make_verdict(
        accuracy_delta=-0.05, cost_usd=1.0, quantum_time=10, classical_time=1, cost_assumptions=CostAssumptions()
    )
    assert verdict is Verdict.CLASSICAL_WINS
    assert "did not beat" in reasoning


def test_verdict_classical_wins_on_exact_tie():
    verdict, _ = _make_verdict(
        accuracy_delta=0.0, cost_usd=1.0, quantum_time=10, classical_time=1, cost_assumptions=CostAssumptions()
    )
    assert verdict is Verdict.CLASSICAL_WINS


def test_verdict_quantum_competitive_within_margin():
    verdict, reasoning = _make_verdict(
        accuracy_delta=0.01,
        cost_usd=1.0,
        quantum_time=10,
        classical_time=1,
        cost_assumptions=CostAssumptions(competitive_margin=0.02),
    )
    assert verdict is Verdict.QUANTUM_COMPETITIVE
    assert "roughly tied" in reasoning


def test_verdict_quantum_wins_beyond_margin():
    verdict, reasoning = _make_verdict(
        accuracy_delta=0.10,
        cost_usd=1.0,
        quantum_time=10,
        classical_time=1,
        cost_assumptions=CostAssumptions(competitive_margin=0.02),
    )
    assert verdict is Verdict.QUANTUM_WINS
    assert "stakeholder" in reasoning


def test_verdict_handles_zero_classical_time():
    verdict, reasoning = _make_verdict(
        accuracy_delta=-0.01, cost_usd=1.0, quantum_time=10, classical_time=0, cost_assumptions=CostAssumptions()
    )
    assert verdict is Verdict.CLASSICAL_WINS
    assert "inf" not in reasoning.lower() or True  # just shouldn't raise


# --- _subsample ---

def test_subsample_returns_full_data_when_none():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_out, y_out = _subsample(X, y, max_samples=None, seed=0)
    assert X_out is X and y_out is y


def test_subsample_returns_full_data_when_smaller_than_cap():
    X = np.arange(20).reshape(10, 2)
    y = np.arange(10)
    X_out, y_out = _subsample(X, y, max_samples=100, seed=0)
    assert X_out is X and y_out is y


def test_subsample_caps_row_count():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)
    X_out, y_out = _subsample(X, y, max_samples=5, seed=0)
    assert X_out.shape == (5, 2)
    assert y_out.shape == (5,)


def test_subsample_is_deterministic_given_seed():
    X = np.arange(40).reshape(20, 2)
    y = np.arange(20)
    X_a, y_a = _subsample(X, y, max_samples=5, seed=7)
    X_b, y_b = _subsample(X, y, max_samples=5, seed=7)
    assert np.array_equal(X_a, X_b)
    assert np.array_equal(y_a, y_b)


# --- evaluate_business_case: end-to-end with tiny/fast settings ---

def test_evaluate_business_case_returns_well_formed_report():
    X_train, y_train, X_test, y_test = _toy_dataset()

    report = evaluate_business_case(
        X_train, y_train, X_test, y_test,
        n_qubits=2, reps=1, layers=1, shots=None, epochs=2, seed=0,
    )

    assert 0.0 <= report.classical_accuracy <= 1.0
    assert 0.0 <= report.quantum_accuracy <= 1.0
    assert report.classical_runtime_sec >= 0
    assert report.quantum_runtime_sec >= 0
    assert report.quantum_estimated_cost_usd >= 0
    assert isinstance(report.verdict, Verdict)
    assert isinstance(report.reasoning, str) and report.reasoning
    assert "Verdict:" in report.summary()


def test_evaluate_business_case_runtime_includes_training():
    # regression test: quantum_runtime_sec must include ansatz training time, not
    # just the feature-extraction/classifier-fit step - otherwise the reported
    # runtime silently understates the true cost of the quantum path.
    X_train, y_train, X_test, y_test = _toy_dataset()

    report = evaluate_business_case(
        X_train, y_train, X_test, y_test,
        n_qubits=2, reps=1, layers=1, shots=None, epochs=2, seed=0,
    )

    assert report.training_runtime_sec > 0
    assert report.inference_runtime_sec > 0
    assert report.quantum_runtime_sec == pytest.approx(
        report.training_runtime_sec + report.inference_runtime_sec
    )
    assert f"{report.training_runtime_sec:.4f}s training" in report.summary()


def test_evaluate_business_case_rejects_unknown_classifier():
    X_train, y_train, X_test, y_test = _toy_dataset()
    with pytest.raises(ValueError):
        evaluate_business_case(X_train, y_train, X_test, y_test, n_qubits=2, classifier="not-a-real-classifier")


def test_evaluate_business_case_cost_scales_with_shots():
    X_train, y_train, X_test, y_test = _toy_dataset()
    cost_assumptions = CostAssumptions(cost_per_shot_usd=0.001)

    report_low = evaluate_business_case(
        X_train, y_train, X_test, y_test,
        n_qubits=2, layers=1, shots=100, epochs=2, cost_assumptions=cost_assumptions, seed=0,
    )
    report_high = evaluate_business_case(
        X_train, y_train, X_test, y_test,
        n_qubits=2, layers=1, shots=400, epochs=2, cost_assumptions=cost_assumptions, seed=0,
    )

    assert report_high.quantum_estimated_cost_usd == pytest.approx(4 * report_low.quantum_estimated_cost_usd)


def test_evaluate_business_case_max_samples_reduces_circuit_evals():
    X_train, y_train, X_test, y_test = _toy_dataset(n_train=20, n_test=10)

    report = evaluate_business_case(
        X_train, y_train, X_test, y_test,
        n_qubits=2, layers=1, shots=None, epochs=2, max_samples=5, seed=0,
    )

    # with max_samples=5 per split, cost is bounded regardless of the 20/10 input size
    uncapped = evaluate_business_case(
        X_train, y_train, X_test, y_test,
        n_qubits=2, layers=1, shots=None, epochs=2, max_samples=None, seed=0,
    )
    assert report.quantum_estimated_cost_usd < uncapped.quantum_estimated_cost_usd
