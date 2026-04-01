#!/usr/bin/env python3
# rul-dashboard/notebook_export.py
"""
One-time script: trains RF, MLP, and LSTM on FD001 with best hyperparameters
and saves weights to models/.

Run from rul-dashboard/:
    python notebook_export.py

Outputs:
    models/scaler.pkl
    models/rf_model.pkl
    models/mlp_model.pt
    models/best_lstm.pt
    models/lstm_config.json
    models/train_losses.json
    models/val_losses.json
"""
import os, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.preprocessing import (
    load_data, fit_scaler, normalize, make_windows,
    WINDOW_SIZE_LSTM, WINDOW_SIZE_RF_MLP, SELECTED_FEATURES
)
from utils.models import LSTMRegressor, MLP, AsymmetricMSELoss

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ── 1. Load & preprocess data ─────────────────────────────────────────────────
print("\n[1/4] Loading data...")
train_df, test_df, rul_series = load_data(DATA_DIR)

scaler   = fit_scaler(train_df)
train_df = normalize(train_df, scaler)
test_df  = normalize(test_df,  scaler)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
print("  Scaler saved.")

# Train/val split by engine_id (80/20)
engine_ids  = train_df["engine_id"].unique()
np.random.seed(42)
val_ids     = set(np.random.choice(engine_ids, size=int(len(engine_ids) * 0.2), replace=False))
train_mask  = train_df["engine_id"].isin(val_ids)
val_df      = train_df[train_mask].copy()
tr_df       = train_df[~train_mask].copy()

# ── 2. Random Forest ──────────────────────────────────────────────────────────
print("\n[2/4] Training Random Forest...")
X_tr_rf, y_tr_rf, _ = make_windows(tr_df,  WINDOW_SIZE_RF_MLP, flatten=True)
X_vl_rf, y_vl_rf, _ = make_windows(val_df, WINDOW_SIZE_RF_MLP, flatten=True)

rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                           min_samples_split=2, random_state=42, n_jobs=-1)
rf.fit(X_tr_rf, y_tr_rf)
joblib.dump(rf, os.path.join(MODELS_DIR, "rf_model.pkl"))
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
preds = rf.predict(X_vl_rf)
print(f"  RF  val  MAE={mean_absolute_error(y_vl_rf, preds):.3f} "
      f"RMSE={np.sqrt(mean_squared_error(y_vl_rf, preds)):.3f} "
      f"R²={r2_score(y_vl_rf, preds):.4f}")
print("  RF model saved.")

# ── 3. MLP ────────────────────────────────────────────────────────────────────
print("\n[3/4] Training MLP...")
X_tr_mlp = torch.tensor(X_tr_rf, dtype=torch.float32)
y_tr_mlp = torch.tensor(y_tr_rf, dtype=torch.float32)
X_vl_mlp = torch.tensor(X_vl_rf, dtype=torch.float32)
y_vl_mlp = torch.tensor(y_vl_rf, dtype=torch.float32)

mlp         = MLP(input_size=WINDOW_SIZE_RF_MLP * len(SELECTED_FEATURES),
                  hidden_layers=[64, 128, 64], dropout_rate=0.2).to(DEVICE)
optimizer   = torch.optim.Adam(mlp.parameters(), lr=1e-3)
criterion   = nn.MSELoss()
loader      = DataLoader(TensorDataset(X_tr_mlp, y_tr_mlp), batch_size=256, shuffle=True)
best_mlp_wt = None
best_val    = float("inf")
patience    = 0

for epoch in range(100):
    mlp.train()
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(mlp(xb).squeeze(-1), yb)
        loss.backward()
        optimizer.step()
    mlp.eval()
    with torch.no_grad():
        val_loss = criterion(mlp(X_vl_mlp.to(DEVICE)).squeeze(-1), y_vl_mlp.to(DEVICE)).item()
    if val_loss < best_val:
        best_val    = val_loss
        best_mlp_wt = {k: v.clone() for k, v in mlp.state_dict().items()}
        patience    = 0
    else:
        patience += 1
        if patience >= 10:
            print(f"  Early stop at epoch {epoch+1}")
            break

mlp.load_state_dict(best_mlp_wt)
torch.save(best_mlp_wt, os.path.join(MODELS_DIR, "mlp_model.pt"))
print("  MLP model saved.")

# ── 4. LSTM ───────────────────────────────────────────────────────────────────
print("\n[4/4] Training best LSTM (hidden=256, seq_len=75, dropout=0.4, lr=5e-4, alpha=1.5)...")
X_tr_lt, y_tr_lt, _ = make_windows(tr_df,  WINDOW_SIZE_LSTM, flatten=False)
X_vl_lt, y_vl_lt, _ = make_windows(val_df, WINDOW_SIZE_LSTM, flatten=False)

X_tr_t = torch.tensor(X_tr_lt, dtype=torch.float32)
y_tr_t = torch.tensor(y_tr_lt, dtype=torch.float32)
X_vl_t = torch.tensor(X_vl_lt, dtype=torch.float32)
y_vl_t = torch.tensor(y_vl_lt, dtype=torch.float32)

INPUT_SIZE = len(SELECTED_FEATURES)  # 15
lstm       = LSTMRegressor(INPUT_SIZE, hidden_size=256, num_layers=1,
                           dropout_rate=0.4, use_attention=False).to(DEVICE)
optimizer  = torch.optim.Adam(lstm.parameters(), lr=5e-4, weight_decay=0)
criterion  = AsymmetricMSELoss(alpha=1.5)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=5, factor=0.5, min_lr=1e-6)
loader     = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=256, shuffle=True)

best_lstm_wt  = None
best_val_lstm = float("inf")
patience      = 0
train_losses, val_losses = [], []

for epoch in range(100):
    lstm.train()
    ep_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(lstm(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm.parameters(), max_norm=1.0)
        optimizer.step()
        ep_loss += loss.item() * len(xb)
    ep_loss /= len(X_tr_t)
    train_losses.append(ep_loss)

    lstm.eval()
    with torch.no_grad():
        val_loss = criterion(lstm(X_vl_t.to(DEVICE)), y_vl_t.to(DEVICE)).item()
    val_losses.append(val_loss)
    scheduler.step(val_loss)

    if val_loss < best_val_lstm:
        best_val_lstm = val_loss
        best_lstm_wt  = {k: v.clone() for k, v in lstm.state_dict().items()}
        patience      = 0
    else:
        patience += 1
        if patience >= 10:
            print(f"  Early stop at epoch {epoch+1}")
            break

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}  train={ep_loss:.4f}  val={val_loss:.4f}")

lstm.load_state_dict(best_lstm_wt)
torch.save(best_lstm_wt, os.path.join(MODELS_DIR, "best_lstm.pt"))

lstm_config = {"input_size": INPUT_SIZE, "hidden_size": 256, "num_layers": 1,
               "dropout_rate": 0.4, "use_attention": False}
with open(os.path.join(MODELS_DIR, "lstm_config.json"), "w") as f:
    json.dump(lstm_config, f, indent=2)
with open(os.path.join(MODELS_DIR, "train_losses.json"), "w") as f:
    json.dump(train_losses, f)
with open(os.path.join(MODELS_DIR, "val_losses.json"), "w") as f:
    json.dump(val_losses, f)

print("  LSTM model + losses saved.")
print("\nAll models exported successfully.")
print(f"  models/scaler.pkl, rf_model.pkl, mlp_model.pt, best_lstm.pt")
