"""Turn a classical-vs-quantum comparison into a go/no-go recommendation.

This is the actual "bridge a business problem to quantum hardware" piece: instead
of just reporting accuracy numbers, it weighs them against runtime/compute cost
and returns a structured verdict, so the comparison this repo runs on MNIST can
be pointed at any dataset to make the same call.
"""

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pennylane as qml
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from qfm.features import get_quantum_features
from qfm.training import train_ansatz

_CLASSIFIERS = {
    "logistic_regression": lambda: LogisticRegression(),
    "svm": lambda: SVC(kernel="linear", C=1.0, random_state=42),
}


def _subsample(X, y, max_samples, seed):
    if max_samples is None or len(X) <= max_samples:
        return X, y
    idx = np.random.default_rng(seed).choice(len(X), size=max_samples, replace=False)
    return X[idx], y[idx]


class Verdict(str, Enum):
    CLASSICAL_WINS = "classical_wins"
    QUANTUM_COMPETITIVE = "quantum_competitive"
    QUANTUM_WINS = "quantum_wins"


@dataclass
class CostAssumptions:
    """Illustrative, user-adjustable cost inputs - not a live pricing quote.

    QPU access is typically billed per second of runtime, not per shot; this
    collapses that into a flat per-shot rate as a simple, adjustable proxy so a
    client's actual quoted rate can be substituted via `cost_per_shot_usd`.
    """

    cost_per_shot_usd: float = 0.00003
    competitive_margin: float = 0.02  # accuracy gap (as a fraction) treated as "roughly tied"


@dataclass
class BusinessCaseReport:
    classical_accuracy: float
    classical_runtime_sec: float
    quantum_accuracy: float
    quantum_runtime_sec: float
    quantum_estimated_cost_usd: float
    accuracy_delta: float
    verdict: Verdict
    reasoning: str
    loss_history: list = field(default_factory=list, repr=False)

    def summary(self) -> str:
        lines = [
            f"Classical accuracy:  {self.classical_accuracy:.1%}  ({self.classical_runtime_sec:.4f}s)",
            f"Quantum accuracy:    {self.quantum_accuracy:.1%}  ({self.quantum_runtime_sec:.4f}s, "
            f"~${self.quantum_estimated_cost_usd:.4f} estimated)",
            f"Accuracy delta:      {self.accuracy_delta:+.1%}",
            f"Verdict:             {self.verdict.value}",
            "",
            self.reasoning,
        ]
        return "\n".join(lines)


def evaluate_business_case(
    X_train,
    y_train,
    X_test,
    y_test,
    n_qubits=None,
    reps=1,
    layers=2,
    shots=1024,
    epochs=20,
    classifier="logistic_regression",
    cost_assumptions=None,
    max_samples=None,
    seed=0,
):
    """Compare a classical baseline against a quantum-feature pipeline on the same data.

    Trains the ansatz on X_train/y_train (see qfm.training.train_ansatz), so this
    works on any binary-labeled tabular dataset - it does not depend on the
    pretrained MNIST parameters used elsewhere in this repo.

    Args:
        X_train, y_train, X_test, y_test: arrays as would be passed to sklearn;
            X should already be scaled to roughly [0, pi] for the ZZ feature map.
        n_qubits: number of qubits/features to use; inferred from X_train if not given.
        reps: ZZ feature map encoding repetitions.
        layers: variational ansatz layers.
        shots: measurement shots for the quantum feature extraction (None = analytic).
        epochs: training epochs for the ansatz.
        classifier: "logistic_regression" or "svm", applied identically to both
            the classical baseline and the quantum-feature classifier for a fair comparison.
        cost_assumptions: a CostAssumptions instance; defaults are illustrative.
        max_samples: if given, caps how many train/test rows go through the quantum
            feature-extraction circuit after ansatz training (each row is one unbatched
            circuit execution, so cost is roughly linear in row count). Does not affect
            ansatz training, which always uses the full X_train (see train_ansatz's own
            batch_size for that). The classical baseline still uses the full dataset, so
            this trades a strictly apples-to-apples comparison for a bounded, interactive
            runtime; pass None (default) for a full-data comparison.
        seed: random seed for ansatz training and subsampling.

    Returns:
        A BusinessCaseReport.
    """
    if classifier not in _CLASSIFIERS:
        raise ValueError(f"classifier must be one of {sorted(_CLASSIFIERS)}, got {classifier!r}")

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    cost_assumptions = cost_assumptions or CostAssumptions()
    n_qubits = n_qubits or X_train.shape[1]

    t0 = time.time()
    classical_clf = _CLASSIFIERS[classifier]()
    classical_clf.fit(X_train, y_train)
    classical_accuracy = classical_clf.score(X_test, y_test)
    classical_runtime_sec = time.time() - t0

    # Trained on the full X_train, not a max_samples subsample: train_ansatz already
    # bounds its own per-epoch cost via batch_size regardless of dataset size, so
    # subsampling here first would only shrink the pool of rows the ansatz ever sees
    # across all epochs, for no speed benefit - artificially handicapping the quantum
    # path's accuracy. max_samples is applied below, only to the feature-extraction
    # step, which has no such internal bounding mechanism of its own.
    params, loss_history = train_ansatz(
        X_train, y_train, n_qubits=n_qubits, layers=layers, reps=reps, epochs=epochs, seed=seed
    )

    Xq_train_in, yq_train = _subsample(X_train, y_train, max_samples, seed)
    Xq_test_in, yq_test = _subsample(X_test, y_test, max_samples, seed)

    # lightning.qubit (compiled C++ backend) for feature extraction: this is shot-based
    # inference with no gradients, where it's faster than default.qubit. Training uses
    # default.qubit instead (see qfm.training) since backprop autodiff there beats
    # lightning's parameter-shift gradients for this circuit size.
    device = qml.device("lightning.qubit", wires=n_qubits)
    t0 = time.time()
    Xq_train = get_quantum_features(Xq_train_in, params, device=device, n_qubits=n_qubits, reps=reps, shots=shots)
    Xq_test = get_quantum_features(Xq_test_in, params, device=device, n_qubits=n_qubits, reps=reps, shots=shots)
    quantum_clf = _CLASSIFIERS[classifier]()
    quantum_clf.fit(Xq_train, yq_train)
    quantum_accuracy = quantum_clf.score(Xq_test, yq_test)
    quantum_runtime_sec = time.time() - t0

    n_circuit_evals = len(Xq_train_in) + len(Xq_test_in)
    billed_shots_per_eval = shots or 1  # analytic mode still stands in for ~1 shot-equivalent of cost
    quantum_estimated_cost_usd = n_circuit_evals * billed_shots_per_eval * cost_assumptions.cost_per_shot_usd

    accuracy_delta = quantum_accuracy - classical_accuracy
    verdict, reasoning = _make_verdict(
        accuracy_delta, quantum_estimated_cost_usd, quantum_runtime_sec, classical_runtime_sec, cost_assumptions
    )

    return BusinessCaseReport(
        classical_accuracy=classical_accuracy,
        classical_runtime_sec=classical_runtime_sec,
        quantum_accuracy=quantum_accuracy,
        quantum_runtime_sec=quantum_runtime_sec,
        quantum_estimated_cost_usd=quantum_estimated_cost_usd,
        accuracy_delta=accuracy_delta,
        verdict=verdict,
        reasoning=reasoning,
        loss_history=loss_history,
    )


def _make_verdict(accuracy_delta, cost_usd, quantum_time, classical_time, cost_assumptions):
    slowdown = quantum_time / classical_time if classical_time > 0 else float("inf")

    if accuracy_delta <= 0:
        verdict = Verdict.CLASSICAL_WINS
        reasoning = (
            f"Quantum features did not beat the classical baseline ({accuracy_delta:+.1%}) "
            f"while costing an estimated ${cost_usd:.4f} and running ~{slowdown:.0f}x slower. "
            "Recommend the classical model."
        )
    elif accuracy_delta < cost_assumptions.competitive_margin:
        verdict = Verdict.QUANTUM_COMPETITIVE
        reasoning = (
            f"Quantum features edged out classical by {accuracy_delta:+.1%}, within the "
            f"{cost_assumptions.competitive_margin:.0%} margin treated as roughly tied here. "
            f"At an estimated ${cost_usd:.4f} and ~{slowdown:.0f}x the runtime, this alone "
            "doesn't justify quantum compute - worth a second look only if the dataset or "
            "ansatz changes meaningfully, or if classical approaches have already plateaued."
        )
    else:
        verdict = Verdict.QUANTUM_WINS
        reasoning = (
            f"Quantum features beat classical by {accuracy_delta:+.1%}, clearing the "
            f"{cost_assumptions.competitive_margin:.0%} competitive margin. Whether that's worth "
            f"the estimated ${cost_usd:.4f} and ~{slowdown:.0f}x runtime depends on what each "
            "accuracy point is worth in this business context - worth taking to a stakeholder."
        )

    return verdict, reasoning
