"""qfm-evaluate: run the classical-vs-quantum business-case comparison on any CSV.

    qfm-evaluate --data mydata.csv --target label_col --qubits 3

Preprocessing (scaling, PCA down to --qubits dimensions, [0, pi] rescaling for the
ZZ feature map) mirrors notebooks/data_prep.ipynb so a new dataset is evaluated
the same way the MNIST case study in this repo was.
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from qfm.business_case import CostAssumptions, evaluate_business_case


def _prepare_data(df, target, n_qubits, test_size, seed):
    y = df[target].to_numpy()
    X_df = df.drop(columns=[target])

    non_numeric = X_df.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise ValueError(
            f"non-numeric feature column(s) {non_numeric}: qfm-evaluate expects numeric "
            "features - encode categorical columns (e.g. one-hot or label encoding) before passing them in"
        )
    X = X_df.to_numpy()

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
    except ValueError as e:
        raise ValueError(
            f"{e}\nThis usually means a class in the target column has too few rows for "
            f"the requested --test-size={test_size} split. Try a smaller --test-size, or "
            "check the target column has enough examples of each class."
        ) from e

    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    n_components = min(n_qubits, X_train.shape[1])
    pca = PCA(n_components=n_components).fit(X_train)
    X_train, X_test = pca.transform(X_train), pca.transform(X_test)

    x_max = np.max(np.abs(X_train))
    if x_max > 0:
        X_train, X_test = (np.pi * X_train) / x_max, (np.pi * X_test) / x_max

    return X_train, y_train, X_test, y_test, n_components


def build_parser():
    parser = argparse.ArgumentParser(
        prog="qfm-evaluate",
        description="Compare a classical baseline against a quantum-feature-map "
        "pipeline on a CSV dataset, and get a cost-aware go/no-go recommendation.",
    )
    parser.add_argument("--data", required=True, help="Path to a CSV file")
    parser.add_argument("--target", required=True, help="Name of the binary target column")
    parser.add_argument("--qubits", type=int, default=3, help="Number of qubits/PCA components (default: 3)")
    parser.add_argument("--reps", type=int, default=1, help="ZZ feature map repetitions (default: 1)")
    parser.add_argument("--layers", type=int, default=2, help="Variational ansatz layers (default: 2)")
    parser.add_argument("--shots", type=int, default=512, help="Measurement shots, or 0 for analytic (default: 512)")
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Ansatz training epochs (default: 10, tuned for a fast first look; raise "
        "this for a more rigorously converged comparison)",
    )
    parser.add_argument(
        "--classifier", choices=["logistic_regression", "svm"], default="logistic_regression"
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default: 0.2)")
    parser.add_argument(
        "--cost-per-shot",
        type=float,
        default=0.00003,
        help="Illustrative USD cost per shot, used to estimate quantum compute cost (default: 0.00003)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=150,
        help="Cap rows sent through the quantum circuit for a fast, interactive run "
        "(the classical baseline still uses the full dataset). Pass 0 to disable "
        "the cap and use the full dataset on both sides (default: 150)",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    df = pd.read_csv(args.data)
    if args.target not in df.columns:
        print(f"error: target column {args.target!r} not found in {args.data}", file=sys.stderr)
        return 1

    try:
        X_train, y_train, X_test, y_test, n_qubits = _prepare_data(
            df, args.target, args.qubits, args.test_size, args.seed
        )
    except ValueError as e:
        # _prepare_data raises a specific, already-complete message per failure
        # cause (non-numeric features vs. a target class too small to split) -
        # relayed as-is rather than appending a one-size-fits-all explanation
        # that could misattribute the actual cause.
        print(f"error: could not prepare {args.data!r}: {e}", file=sys.stderr)
        return 1
    if n_qubits < args.qubits:
        print(f"note: using {n_qubits} qubits (dataset only has {n_qubits} usable features)", file=sys.stderr)

    max_samples = args.max_samples or None
    if max_samples and (len(X_train) > max_samples or len(X_test) > max_samples):
        print(
            f"note: capping the quantum path at {max_samples} rows per split for a fast run "
            "(classical baseline still uses the full dataset) - pass --max-samples 0 for a "
            "full-data, apples-to-apples comparison",
            file=sys.stderr,
        )

    report = evaluate_business_case(
        X_train,
        y_train,
        X_test,
        y_test,
        n_qubits=n_qubits,
        reps=args.reps,
        layers=args.layers,
        shots=args.shots or None,
        epochs=args.epochs,
        classifier=args.classifier,
        cost_assumptions=CostAssumptions(cost_per_shot_usd=args.cost_per_shot),
        max_samples=max_samples,
        seed=args.seed,
    )
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
