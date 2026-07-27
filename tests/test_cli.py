import numpy as np
import pandas as pd
import pytest

from qfm.cli import _prepare_data, build_parser, main


def _toy_csv(tmp_path, n=30, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = (X[:, 0] > 0).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["label"] = y
    path = tmp_path / "toy.csv"
    df.to_csv(path, index=False)
    return path


def test_build_parser_defaults():
    args = build_parser().parse_args(["--data", "x.csv", "--target", "y"])
    assert args.qubits == 3
    assert args.reps == 1
    assert args.layers == 2
    assert args.shots == 512
    assert args.epochs == 10
    assert args.classifier == "logistic_regression"
    assert args.max_samples == 150


def test_prepare_data_scales_and_reduces_dimensions(tmp_path):
    path = _toy_csv(tmp_path, n_features=5)
    df = pd.read_csv(path)

    X_train, y_train, X_test, y_test, n_qubits = _prepare_data(df, "label", n_qubits=3, test_size=0.2, seed=0)

    assert n_qubits == 3
    assert X_train.shape[1] == 3
    assert X_test.shape[1] == 3
    assert set(np.unique(y_train)) <= {0, 1}
    assert np.max(np.abs(X_train)) <= np.pi + 1e-9


def test_prepare_data_caps_qubits_to_available_features(tmp_path):
    path = _toy_csv(tmp_path, n_features=2)
    df = pd.read_csv(path)

    _, _, _, _, n_qubits = _prepare_data(df, "label", n_qubits=5, test_size=0.2, seed=0)

    assert n_qubits == 2


def test_main_end_to_end_with_tiny_csv(tmp_path, capsys):
    path = _toy_csv(tmp_path, n=30, n_features=3)

    exit_code = main([
        "--data", str(path),
        "--target", "label",
        "--qubits", "2",
        "--epochs", "2",
        "--shots", "0",
        "--max-samples", "10",
    ])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Verdict:" in captured.out
    assert "Classical accuracy:" in captured.out
    assert "Quantum accuracy:" in captured.out


def test_main_reports_error_for_missing_target_column(tmp_path, capsys):
    path = _toy_csv(tmp_path)

    exit_code = main(["--data", str(path), "--target", "not_a_column"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "not_a_column" in captured.err


def test_main_reports_clean_error_for_class_too_small_to_stratify(tmp_path, capsys):
    # regression test: a minority class too small for a stratified split used to
    # crash with a raw sklearn traceback instead of a handled, exit-code-1 error.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    y = np.array([0] * 29 + [1])  # 1 minority-class example - too few to split
    df = pd.DataFrame(X, columns=["a", "b", "c"])
    df["label"] = y
    path = tmp_path / "imbalanced.csv"
    df.to_csv(path, index=False)

    exit_code = main(["--data", str(path), "--target", "label", "--epochs", "2"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "error:" in captured.err
