"""Interactive demo: quantum feature map + classical classifier vs. classical baseline.

Run with: streamlit run app.py
Uses the pretrained ansatz parameters in notebooks/params/ (training takes hours,
so this app does inference only) and a random subsample of the PCA-reduced MNIST
data for interactive response times. Full-dataset sweep results, produced offline,
live in results/metrics/ and are summarized in README.md.
"""

import time

import numpy as np
import pandas as pd
import pennylane as qml
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from qfm.features import make_qnode, get_quantum_features

DATA_PATH = "notebooks/data/mnist01_pca4.npz"
PARAMS_PATH_TMPL = "notebooks/params/trained_params_{}.npy"
LR_BASELINE_CSV = "results/metrics/lr_results.csv"
SVM_BASELINE_CSV = "results/metrics/svm_results.csv"

N_TRAIN_SUBSAMPLE = 300
N_TEST_SUBSAMPLE = 150
SUBSAMPLE_SEED = 0


@st.cache_data
def load_data():
    data = np.load(DATA_PATH)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    # scale to [0, pi] so the ZZ feature map gets meaningful rotation angles,
    # matching the preprocessing done in notebooks/quantum_pipeline.ipynb
    X_max = np.max(np.abs(X_train))
    X_train = (np.pi * X_train) / X_max
    X_test = (np.pi * X_test) / X_max

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    train_idx = rng.choice(len(X_train), size=N_TRAIN_SUBSAMPLE, replace=False)
    test_idx = rng.choice(len(X_test), size=N_TEST_SUBSAMPLE, replace=False)
    return (
        X_train[train_idx],
        y_train[train_idx],
        X_test[test_idx],
        y_test[test_idx],
    )


@st.cache_data
def classical_baseline():
    lr = pd.read_csv(LR_BASELINE_CSV).iloc[0]
    svm = pd.read_csv(SVM_BASELINE_CSV).iloc[0]
    return lr, svm


@st.cache_data
def load_trained_params(n_qubits):
    return np.load(PARAMS_PATH_TMPL.format(n_qubits))


@st.cache_data(show_spinner=False)
def run_quantum_pipeline(n_qubits, reps, shots):
    X_train, y_train, X_test, y_test = load_data()
    X_train = X_train[:, :n_qubits]
    X_test = X_test[:, :n_qubits]
    params = load_trained_params(n_qubits)

    device = qml.device("default.qubit", wires=n_qubits)

    t0 = time.time()
    Xq_train = get_quantum_features(X_train, params, device=device, n_qubits=n_qubits, reps=reps, shots=shots)
    Xq_test = get_quantum_features(X_test, params, device=device, n_qubits=n_qubits, reps=reps, shots=shots)
    feature_time = time.time() - t0

    results = {}
    for name, clf in [
        ("Logistic Regression", LogisticRegression()),
        ("SVM", SVC(kernel="linear", C=1.0, random_state=42)),
    ]:
        start = time.time()
        clf.fit(Xq_train, y_train)
        acc = clf.score(Xq_test, y_test)
        results[name] = {"accuracy": acc, "runtime_sec": time.time() - start}

    return results, feature_time


st.set_page_config(page_title="Quantum Feature Maps Demo", page_icon="⚛️", layout="wide")

st.title("Quantum Feature Maps: Interactive Demo")
st.caption(
    "Encode PCA-reduced MNIST (0 vs. 1) through a ZZ feature map + trained variational "
    "ansatz, extract quantum features, and compare classical models trained on them "
    "against the classical baseline. See README.md for the full write-up and when this "
    "is (and isn't) worth the added cost/latency."
)

with st.sidebar:
    st.header("Circuit configuration")
    n_qubits = st.select_slider("Number of qubits", options=[2, 3, 4], value=3)
    reps = st.select_slider("ZZ feature map depth (reps)", options=[1, 2], value=1)
    shots_choice = st.radio("Measurement shots", ["Analytic (exact)", "1024 shots"], index=0)
    shots = None if shots_choice.startswith("Analytic") else 1024
    st.caption(
        f"Using a random {N_TRAIN_SUBSAMPLE}/{N_TEST_SUBSAMPLE} train/test subsample "
        "so the circuit re-runs interactively. The full-dataset sweep (12,665/2,115 "
        "samples) that this demo is based on is logged in `results/metrics/`."
    )

lr_baseline, svm_baseline = classical_baseline()

with st.spinner("Running the quantum feature map circuit and training classical models..."):
    results, feature_time = run_quantum_pipeline(n_qubits, reps, shots)

st.subheader("Accuracy: quantum features vs. classical baseline")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Classical LR", f"{lr_baseline['accuracy'] * 100:.1f}%")
col2.metric(
    "Quantum LR",
    f"{results['Logistic Regression']['accuracy'] * 100:.1f}%",
    delta=f"{(results['Logistic Regression']['accuracy'] - lr_baseline['accuracy']) * 100:+.1f} pts",
)
col3.metric("Classical SVM", f"{svm_baseline['accuracy'] * 100:.1f}%")
col4.metric(
    "Quantum SVM",
    f"{results['SVM']['accuracy'] * 100:.1f}%",
    delta=f"{(results['SVM']['accuracy'] - svm_baseline['accuracy']) * 100:+.1f} pts",
)

st.subheader("Cost: runtime per configuration")
runtime_df = pd.DataFrame(
    {
        "Stage": ["Quantum feature extraction (train+test)", "Classical LR fit", "Classical SVM fit"],
        "Runtime (s)": [
            feature_time,
            results["Logistic Regression"]["runtime_sec"],
            results["SVM"]["runtime_sec"],
        ],
    }
)
st.dataframe(runtime_df, hide_index=True, width="stretch")
st.caption(
    f"For reference, the classical baseline (full dataset) trains in "
    f"{lr_baseline['runtime_sec']:.3f}s (LR) / {svm_baseline['runtime_sec']:.3f}s (SVM) total."
)

st.subheader("Circuit diagram")
device = qml.device("default.qubit", wires=n_qubits)
qnode = make_qnode(device, n_qubits, reps=reps, shots=shots)
params = load_trained_params(n_qubits)
sample_x = load_data()[0][0]
st.code(qml.draw(qnode)(sample_x, params), language="text")
