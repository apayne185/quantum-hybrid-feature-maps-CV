"""Regression test for the quantum-engineered dataset actually showing a quantum
advantage through the real evaluate_business_case pipeline - not just that the
generator produces data, but that the effect the construction predicts (Havlicek
et al. 2019) reproduces empirically in this codebase.

Kept small (2 qubits, 10 epochs, 30 samples) to stay fast; see
examples/quantum_advantage_case_study.py for the fuller, more dramatic version
this is a cheap regression check for.
"""

from qfm.business_case import Verdict, evaluate_business_case
from qfm.synthetic import generate_quantum_engineered_dataset


def test_quantum_wins_on_quantum_engineered_dataset():
    X, y, _ = generate_quantum_engineered_dataset(n_samples=30, n_qubits=2, reps=1, layers=2, margin=0.2, seed=0)
    X_train, y_train = X[:20], y[:20]
    X_test, y_test = X[20:], y[20:]

    report = evaluate_business_case(
        X_train, y_train, X_test, y_test,
        n_qubits=2, reps=1, layers=2, shots=None, epochs=10, seed=0,
    )

    # a linear classical model has no access to the interference structure that
    # generated the labels, so it should sit near chance; the matching quantum
    # circuit family should recover the labeling function far better
    assert report.classical_accuracy <= 0.6
    assert report.quantum_accuracy >= 0.8
    assert report.verdict is Verdict.QUANTUM_WINS
