# Quantum Feature Maps (Hybrid Preprocessing for Computer Vision)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.37+-purple.svg)](https://pennylane.ai/)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.41+-violet.svg)](https://qiskit.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![NumPy](https://img.shields.io/badge/numpy-1.24+-lightblue.svg)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/pandas-2.0+-teal.svg)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e.svg)](https://scikit-learn.org/stable/)
[![Matplotlib](https://img.shields.io/badge/matplotlib-3.7+-yellow.svg)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/seaborn-0.12+-9cf.svg)](https://seaborn.pydata.org/)
[![JupyterLab](https://img.shields.io/badge/JupyterLab-4.0+-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Project Overview

Quantum feature maps (QFMs) are hybrid ML techniques that explore how quantum circuits can enhance classical preprocessing pipelines, specifically here for computer vision datasets such as MNIST. This project investigates the use of QFMs as nonlinear transformations that embed classical data into a high-dimensional Hilbert space, allowing classical models (Logistic Regression, SVM) to potentially capture richer relationships than standard classical embeddings.

QFM integrates classical dimensionality reduction with parameterized quantum circuits (PQCs) implemented in PennyLane (and tested in Qiskit), benchmarking their performance against classical baselines. The goal is to understand:

* How the choice of quantum feature map, ansatz, and backend impacts performance.
* When/if quantum-enhanced preprocessing can outperform classical methods on small-scale classification tasks.

This repo provides reproducible experiments for classical and hybrid quantum-classical models, along with visualizations, parameter sweeps, and IBM Quantum integration.

## Results at a Glance

Task: binary classification of MNIST digits 0 vs. 1, reduced to ≤4 features via PCA.

| Model | Setup | Accuracy | Runtime (s) |
|---|---|---|---|
| Logistic Regression (classical) | PCA(4) features | **99.6%** | 0.008 |
| SVM (classical) | PCA(4) features | **99.6%** | 0.06 |
| Logistic Regression (quantum features) | 3 qubits, depth 1, analytic | 91.5% | 0.042 |
| SVM (quantum features) | 3 qubits, depth 1, analytic | 91.7% | 1.63 |

Full sweep across qubit count (2-4), circuit depth (1-2), and shot count (analytic vs. 1024 shots) is logged in [`results/metrics/`](results/metrics/) and produced by [`quantum_pipeline.ipynb`](notebooks/quantum_pipeline.ipynb) / [`final_analysis.ipynb`](notebooks/final_analysis.ipynb).

**Takeaway:** on this task the classes are close to linearly separable, so the classical baselines are hard to beat — the quantum-feature models top out around 91-92% (3 qubits, depth 1) and *degrade* as qubits/depth increase (down to ~70% at 4 qubits, depth 2). That's consistent with added circuit expressivity increasing optimization difficulty (more local minima / flatter loss landscape) faster than it adds useful nonlinearity, given no noise mitigation or trainable-embedding tricks are used here. This is treated as a real, reported negative result rather than something to talk around — see [Research Motivation](#research-motivation) below.

## Configure Environment

**Create Conda Environment**
```bash
conda env create --file environment.yml
conda activate qfm-env
```

## Repository Organization
```
quantum-hybrid-feature-maps-CV/
├── environment.yml       # Conda env configuration for qfm-env
├── README.md             # This file
├── LICENSE
├── notebooks/
│   ├── data/
│   │   └── mnist01_pca4.npz     # PCA-reduced features (raw MNIST is gitignored, auto-downloaded)
│   ├── imgs/                    # Images used for documentation
│   ├── params/                  # Saved trained ansatz parameter files
│   ├── data_prep.ipynb          # Data preprocessing, dimensionality reduction
│   ├── classical_baseline.ipynb # Classical ML baselines
│   ├── quantum_pipeline.ipynb   # Full quantum feature map + variational circuit pipeline
│   ├── quantum_feature_maps_tests.ipynb  # Tests for PennyLane/Qiskit feature maps
│   └── final_analysis.ipynb     # Post-training metrics, plots, performance summary
├── results/
│   ├── metrics/           # Accuracy, loss, and runtime data (CSV)
│   │   └── final/         # Final summary tables from final_analysis.ipynb
│   └── models/            # Pickled trained classical models
└── .env                   # Stores IBM Quantum API key (gitignored, not committed)
```

## Notebooks

### `data_prep.ipynb`
**Purpose:** Prepare and visualize the input data.
**Main steps:**
- Load a small dataset (`MNIST 0 vs 1` subset), auto-downloaded via `torchvision`.
- Normalize and scale features.
- Apply **Principal Component Analysis (PCA)** to reduce dimensionality (≤ 4 features).
- Visualize the reduced features for interpretability.

**Output:** `notebooks/data/mnist01_pca4.npz` (used by later notebooks)

---

### `classical_baseline.ipynb`
**Purpose:** Establish classical machine learning baselines.
**Main steps:**
- Train classifiers (Logistic Regression, SVM) on PCA-reduced features.
- Perform cross-validation to assess robustness.
- Plot decision boundaries for the 2D case.
- Save accuracy and runtime metrics.

**Output:** `results/metrics/lr_results.csv`, `results/metrics/svm_results.csv`

---

### `quantum_pipeline.ipynb`
**Purpose:** Implement the **quantum feature map + variational ansatz** pipeline using **PennyLane** and **Qiskit**.
**Main steps:**
- Define a **ZZFeatureMap** for encoding classical data into quantum states.
- Implement a **variational ansatz** (1-2 layers of parameterized rotations + entanglers).
- Measure expectation values of Pauli operators → quantum feature vectors.
- Train Logistic Regression and SVM on these quantum features.
- Sweep over:
  - Number of qubits (2-4)
  - Circuit depth (repetitions)
  - Number of shots (analytic vs. 1024)
  - Noise models (optional)
- Log results to CSV.
- Connects to **IBM Quantum devices** using `qiskit-ibm-runtime`.

**Output:**
- Trained parameters → `notebooks/params/trained_params*.npy`
- Logged metrics → `results/metrics/q_lr_sweep_results.csv`, `results/metrics/q_svm_sweep_results.csv`

---

### `quantum_feature_maps_tests.ipynb`
**Purpose:** Explore and visualize quantum encoding schemes side by side.
**Main steps:**
- Test **Basis**, **Angle**, and **Amplitude** encoding using both **Qiskit** and **PennyLane**.
- Test **ZZFeatureMap** and raw feature vector encoding of classical data into quantum states.
- Visualize each method's transformations and circuits (see `notebooks/imgs/`).

---

### `final_analysis.ipynb`
**Purpose:** Aggregate and analyze all results.
**Main steps:**
- Load metrics from both classical and quantum experiments.
- Generate comparison plots: accuracy vs. shots, accuracy vs. circuit depth, runtime vs. qubit count.
- Compute accuracy deltas and resource overhead vs. classical baselines.
- Summarize when (if ever) the quantum pipeline matches or beats classical baselines.

**Output:** `results/metrics/final/q_lr_summary.csv`, `results/metrics/final/q_svm_summary.csv`

## Key Features

### Hybrid Classical-Quantum Pipeline
* Combines PCA-based feature reduction with quantum embeddings using a PennyLane ZZ feature map.
* Integrates a variational ansatz as a learnable quantum transformation for enhanced feature expressivity.

### Configurable Experiment Parameters
* Varied number of qubits, circuit depth, number of measurement shots, device backend (simulator vs. IBM backend), and noise models.

### Classical Baseline Comparison
* Uses Logistic Regression and SVM trained on the same PCA-reduced data.
* Enables direct accuracy/runtime comparison between classical and quantum-enhanced models.

### Quantum Circuit Visualization
* Visualizes feature maps and ansatz circuits for inspection using PennyLane and Qiskit.

### IBM Quantum Integration
* Access to real IBM Quantum devices via `qiskit-ibm-runtime`, using least-busy backend selection for queue management.

### Logging Results and Analysis
* Automatic metric tracking (accuracy, runtime, number of parameters).
* Generates CSV logs and summary visualizations for classical/quantum experiments.

## Key Concepts

### Quantum Feature Maps
Transform classical data into quantum states through unitary operations, allowing models to implicitly compute inner products in a high-dimensional Hilbert space (similar to kernel methods, but with potentially richer representational capacity).

Maps used in `quantum_feature_maps_tests.ipynb`:
* Basis Encoding
* Angle Encoding (encodes data into rotation angles)
  * ZZ feature map (adds entangling gates to capture data correlations)
* Amplitude Encoding (uses vector amplitudes of quantum states)

### Variational Ansatz
A parameterized quantum circuit designed to learn transformations that best separate classes. These parameters are optimized classically, bridging classical optimization and quantum hardware.

### Hybrid Training Workflow
* Preprocessing: Classical PCA
* Encoding: Angle encoding QFM
* Measurement: Expectation values produce quantum features
* Classification: Classical models trained on quantum features

### Quantum Hardware and Simulators
Supports both **local simulators** (`default.qubit`, `aer_simulator`) and **IBM Quantum devices** (`qiskit-ibm-runtime`), so experiments can scale from local to realistic noisy devices.

## Research Motivation

Quantum computing is still in an early, noise-limited stage, but hybrid quantum-classical learning is one of the more promising paths toward a practical quantum advantage in ML.

Classical models often struggle to represent highly entangled or nonlinear data relationships. Quantum circuits can encode data into exponentially larger Hilbert spaces using relatively few qubits, which in principle could reveal correlations that are classically hard to capture.

This project explores QFMs as a trainable preprocessing layer, bridging classical feature extraction (PCA) with quantum-enhanced computation. The results above are an honest negative result on an easy dataset: quantum features did not beat the classical baseline here, and performance degraded with added depth/qubits. That's a useful data point in itself, and it directly motivates the next steps below.

## Future Work
* Integration with classical deep learning pipelines (hybrid QNN-CNN architectures).
* Noise-resilient training: evaluate how noise impacts accuracy/stability, and explore noise-adaptive ansatz designs.
* Trainable QFMs (learned embeddings instead of fixed ZZ feature maps).
* Re-run the comparison on a harder, non-linearly-separable task (e.g. MNIST digit pairs that are not trivially separable, or a synthetic dataset with known nonlinear structure) where a quantum kernel is more likely to show an advantage.
