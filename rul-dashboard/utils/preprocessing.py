import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES = [
    "engine_id", "cycle",
    "operational_setting_1", "operational_setting_2", "operational_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5", "sensor_6",
    "sensor_7", "sensor_8", "sensor_9", "sensor_10", "sensor_11", "sensor_12",
    "sensor_13", "sensor_14", "sensor_15", "sensor_16", "sensor_17", "sensor_18",
    "sensor_19", "sensor_20", "sensor_21",
]

REMOVED_FEATURES = [
    "operational_setting_2", "operational_setting_3",
    "sensor_1", "sensor_5", "sensor_6", "sensor_10",
    "sensor_16", "sensor_18", "sensor_19",
]

SELECTED_FEATURES = [
    c for c in COLUMN_NAMES
    if c not in ("engine_id", "cycle") and c not in REMOVED_FEATURES
]

MAX_RUL = 125
WINDOW_SIZE_LSTM   = 75   # best LSTM
WINDOW_SIZE_RF_MLP = 30   # RF and MLP (30 x 15 = 450-dim flat)


def load_data(data_dir: str):
    """Load C-MAPSS FD001 train/test/RUL files, compute and cap RUL labels.

    Returns:
        train_df   : DataFrame with COLUMN_NAMES + 'RUL', capped at MAX_RUL
        test_df    : DataFrame with COLUMN_NAMES + 'RUL' (computed from RUL_FD001)
        rul_series : Series of ground-truth final RUL values, indexed 1..N_test_engines
    """
    train_df = pd.read_csv(
        os.path.join(data_dir, "train_FD001.txt"), sep=r"\s+", header=None,
        names=COLUMN_NAMES,
    )
    test_df = pd.read_csv(
        os.path.join(data_dir, "test_FD001.txt"), sep=r"\s+", header=None,
        names=COLUMN_NAMES,
    )
    rul_raw = pd.read_csv(
        os.path.join(data_dir, "RUL_FD001.txt"), header=None, names=["RUL_end"],
    )

    # Build RUL for training set: max_cycle - cycle, capped at MAX_RUL
    train_df = compute_rul(train_df)

    # Build RUL for test set: RUL_end + (max_test_cycle - cycle)
    max_test_cycle = test_df.groupby("engine_id")["cycle"].max().reset_index()
    max_test_cycle.columns = ["engine_id", "max_test_cycle"]
    rul_raw["engine_id"] = rul_raw.index + 1
    max_test_cycle = max_test_cycle.merge(rul_raw, on="engine_id", how="left")
    test_df = test_df.merge(
        max_test_cycle[["engine_id", "max_test_cycle", "RUL_end"]], on="engine_id", how="left"
    )
    test_df["RUL"] = (test_df["RUL_end"] + (test_df["max_test_cycle"] - test_df["cycle"])).clip(upper=MAX_RUL)
    test_df.drop(columns=["max_test_cycle", "RUL_end"], inplace=True)

    rul_series = rul_raw.set_index("engine_id")["RUL_end"]

    return train_df, test_df, rul_series


def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'RUL' column to a training-style DataFrame (has full run-to-failure)."""
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].max()
    df["RUL"] = df.apply(lambda r: max_cycle[r["engine_id"]] - r["cycle"], axis=1)
    df["RUL"] = df["RUL"].clip(upper=MAX_RUL)
    return df


def fit_scaler(df: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on SELECTED_FEATURES from a training DataFrame."""
    scaler = StandardScaler()
    scaler.fit(df[SELECTED_FEATURES])
    return scaler


def normalize(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Apply scaler transform to SELECTED_FEATURES in-place (copy returned)."""
    df = df.copy()
    df[SELECTED_FEATURES] = scaler.transform(df[SELECTED_FEATURES])
    return df


def make_windows(df: pd.DataFrame, window_size: int, flatten: bool = False):
    """Create sliding windows per engine.

    Args:
        df          : normalized DataFrame with SELECTED_FEATURES + 'engine_id' + 'cycle' + 'RUL'
        window_size : number of cycles per window
        flatten     : if True, returns shape (N, window_size * 15) -- for RF/MLP
                      if False, returns shape (N, window_size, 15) -- for LSTM

    Returns:
        X    : np.ndarray
        y    : np.ndarray of RUL values
        eids : np.ndarray of engine_ids (one per window, for the last cycle)
    """
    X_list, y_list, eid_list = [], [], []
    for eid, grp in df.groupby("engine_id"):
        grp = grp.sort_values("cycle")
        vals = grp[SELECTED_FEATURES].values  # shape: (n_cycles, 15)
        ruls = grp["RUL"].values
        if len(grp) < window_size:
            continue
        for i in range(window_size - 1, len(grp)):
            window = vals[i - window_size + 1 : i + 1]
            X_list.append(window.flatten() if flatten else window)
            y_list.append(ruls[i])
            eid_list.append(eid)

    return np.array(X_list), np.array(y_list), np.array(eid_list)
