import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.preprocessing import (
    COLUMN_NAMES, SELECTED_FEATURES, REMOVED_FEATURES,
    MAX_RUL, load_data, compute_rul, fit_scaler, normalize, make_windows
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _make_fake_df(n_engines=3, cycles_per_engine=50):
    """Build a minimal synthetic dataframe matching C-MAPSS column layout."""
    rows = []
    for eid in range(1, n_engines + 1):
        for cyc in range(1, cycles_per_engine + 1):
            row = [eid, cyc] + [0.0] * 3 + [float(i) for i in range(1, 22)]
            rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMN_NAMES)
    return df


def test_column_names_length():
    assert len(COLUMN_NAMES) == 26  # engine_id, cycle, 3 op_settings, 21 sensors


def test_selected_features_count():
    assert len(SELECTED_FEATURES) == 15


def test_removed_features_count():
    assert len(REMOVED_FEATURES) == 9


def test_compute_rul_caps_at_125():
    df = _make_fake_df(n_engines=2, cycles_per_engine=200)
    df = compute_rul(df)
    assert df["RUL"].max() == MAX_RUL
    assert df["RUL"].min() == 0


def test_compute_rul_decreasing_per_engine():
    df = _make_fake_df(n_engines=1, cycles_per_engine=50)
    df = compute_rul(df)
    rul_vals = df[df["engine_id"] == 1].sort_values("cycle")["RUL"].values
    assert (np.diff(rul_vals) <= 0).all()


def test_fit_and_normalize():
    df = _make_fake_df(n_engines=3, cycles_per_engine=50)
    df = compute_rul(df)
    scaler = fit_scaler(df)
    normed = normalize(df.copy(), scaler)
    feature_means = normed[SELECTED_FEATURES].mean()
    assert (feature_means.abs() < 1e-6).all()  # mean ~0 after StandardScaler


def test_make_windows_shape_lstm():
    df = _make_fake_df(n_engines=3, cycles_per_engine=50)
    df = compute_rul(df)
    scaler = fit_scaler(df)
    df = normalize(df, scaler)
    X, y, eids = make_windows(df, window_size=10, flatten=False)
    # Each engine contributes (50 - 10 + 1) = 41 windows -> 3 * 41 = 123
    assert X.shape == (123, 10, 15)
    assert y.shape == (123,)
    assert eids.shape == (123,)


def test_make_windows_shape_flat():
    df = _make_fake_df(n_engines=2, cycles_per_engine=40)
    df = compute_rul(df)
    scaler = fit_scaler(df)
    df = normalize(df, scaler)
    X, y, eids = make_windows(df, window_size=10, flatten=True)
    # Each engine: (40 - 10 + 1) = 31 windows -> 2 * 31 = 62
    assert X.shape == (62, 10 * 15)
    assert y.shape == (62,)


def test_load_data_returns_correct_shapes():
    if not os.path.exists(os.path.join(DATA_DIR, "train_FD001.txt")):
        pytest.skip("Dataset files not present")
    train_df, test_df, rul_series = load_data(DATA_DIR)
    assert set(train_df.columns) == set(COLUMN_NAMES + ["RUL"])
    assert len(rul_series) == test_df["engine_id"].nunique()
    assert train_df["RUL"].max() == MAX_RUL
