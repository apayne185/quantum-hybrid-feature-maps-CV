"""Case study: a dataset where quantum wins, by construction.

Everywhere else in this repo (MNIST, sklearn's breast cancer dataset), classical
baselines beat the quantum-feature pipeline - see README.md's "When Would a
Business Actually Use This?" section. That's an honest result, but it leaves an
open question: does the quantum pipeline in this repo have *any* real advantage
under *any* circumstances, or does it just never win?

This script answers that with a dataset constructed per Havlicek et al. (2019),
"Supervised learning with quantum-enhanced feature spaces" (Nature 567,
209-212): labels come from a fixed, random circuit built from the same
feature-map + ansatz family this repo's quantum classifier learns from (see
qfm.synthetic for the construction and why it isn't just "hard-looking data").
A linear classical model has no access to the interference structure that
produced the label; a quantum model from the same circuit family can, in
principle, represent it exactly.

This is a deliberately constructed proof-of-concept, not a claim about real
business data - it demonstrates the mechanism behind a quantum-kernel advantage
exists and reproduces in this codebase, which is precisely why it looks nothing
like the honest negative results found elsewhere in this repo on real data.

Run: python examples/quantum_advantage_case_study.py
Takes ~2-3 minutes (each row trains a fresh ansatz from scratch).
"""

import time

from qfm.business_case import evaluate_business_case
from qfm.synthetic import generate_quantum_engineered_dataset

CONFIGS = [
    dict(name="2 qubits", n_qubits=2, reps=1, label_layers=2, train_layers=2, margin=0.20, epochs=30, n_samples=120),
    dict(name="3 qubits", n_qubits=3, reps=1, label_layers=2, train_layers=2, margin=0.20, epochs=30, n_samples=120),
    dict(name="4 qubits", n_qubits=4, reps=1, label_layers=2, train_layers=2, margin=0.15, epochs=30, n_samples=120),
]


def run(cfg):
    X, y, _ = generate_quantum_engineered_dataset(
        n_samples=cfg["n_samples"],
        n_qubits=cfg["n_qubits"],
        reps=cfg["reps"],
        layers=cfg["label_layers"],
        margin=cfg["margin"],
        seed=0,
    )
    split = int(0.7 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    return evaluate_business_case(
        X_train,
        y_train,
        X_test,
        y_test,
        n_qubits=cfg["n_qubits"],
        reps=cfg["reps"],
        layers=cfg["train_layers"],
        shots=None,
        epochs=cfg["epochs"],
        seed=0,
    )


def main():
    print(__doc__)
    rows = []
    for cfg in CONFIGS:
        t0 = time.time()
        report = run(cfg)
        elapsed = time.time() - t0
        rows.append((cfg["name"], report.classical_accuracy, report.quantum_accuracy, report.verdict.value))
        print(
            f"{cfg['name']}: classical={report.classical_accuracy:.1%}  "
            f"quantum={report.quantum_accuracy:.1%}  verdict={report.verdict.value}  ({elapsed:.0f}s)"
        )

    print("\n| Config | Classical accuracy | Quantum accuracy | Verdict |")
    print("|---|---|---|---|")
    for name, classical_acc, quantum_acc, verdict in rows:
        print(f"| {name} | {classical_acc:.1%} | {quantum_acc:.1%} | {verdict} |")


if __name__ == "__main__":
    main()
