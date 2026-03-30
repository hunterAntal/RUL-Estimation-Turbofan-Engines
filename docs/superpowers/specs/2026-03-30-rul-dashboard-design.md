# RUL Prediction Dashboard — Design Spec
**Date:** 2026-03-30
**Project:** ESOF-4011 Applied Computational Intelligence — Lakehead University
**Authors:** Felix Ikokwu & Hunter Antal

---

## Overview

A Streamlit web dashboard that lets users explore the NASA C-MAPSS FD001 dataset, compare three trained RUL prediction models (RF, MLP, LSTM), run live LSTM predictions on test engines, and deep-dive into LSTM training details. Built for the 10% bonus rubric item: "web-based application user interface to support data-mining queries and result visualization."

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Navigation | `st.tabs()` top-tab bar | Native Streamlit — no CSS hacks, matches user-selected layout |
| Code structure | Single `app.py` + `utils/` | Simplest to run/share; avoids Streamlit multi-page sidebar quirks |
| Visual theme | Bold/high-contrast Lakehead | Gold-accent card tops, white numbers, cobalt blue — high-impact for demo |
| Model export | `notebook_export.py` standalone script | Fully reproducible; no dependency on notebook state |
| Dataset source | Copy from kagglehub cache to `data/` | Already downloaded at `~/.cache/kagglehub/datasets/behrad3d/nasa-cmaps/versions/1/CMaps/` |

---

## File Structure

```
rul-dashboard/
├── app.py                   # Entry point: page config, header, st.tabs(), renders each tab
├── utils/
│   ├── preprocessing.py     # load_data(), compute_rul(), normalize(), make_windows()
│   ├── models.py            # LSTMRegressor + MLP class definitions (from notebook)
│   └── visualization.py    # Shared Plotly chart builder functions
├── models/
│   ├── best_lstm.pt         # saved by notebook_export.py
│   ├── mlp_model.pt
│   └── rf_model.pkl
├── data/
│   ├── train_FD001.txt      # copied from kagglehub cache
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── notebook_export.py       # one-time script: retrains + saves model weights
├── requirements.txt
└── README.md
```

---

## Architecture

### `app.py`

- `st.set_page_config(layout="wide", page_title="RUL Dashboard", page_icon="⚙️")`
- Custom CSS injected via `st.markdown` for Lakehead theme: dark background `#060e1a`, cobalt blue `#00427A`, blaze gold `#FFC20E`
- Header bar: "TURBOFAN RUL DASHBOARD" with Lakehead Faculty of Engineering subtitle and author names
- `tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset Explorer", "📈 Model Comparison", "🔮 RUL Predictor", "🔍 Model Deep Dive"])`
- Each tab calls a `render_*()` function defined in the same file, importing from `utils/`

### Caching

- `@st.cache_data` on `load_data()` — loads and preprocesses train/test CSVs once per session
- `@st.cache_resource` on `load_lstm_model()`, `load_mlp_model()`, `load_rf_model()` — loads `.pt`/`.pkl` files once per process

### Graceful degradation

If any model file is missing:
- **Model Comparison** tab: shows hardcoded metrics table + bar charts (no scatter/residuals — those require predictions)
- **RUL Predictor** tab: shows `st.warning("Model weights not found. Run notebook_export.py to enable live predictions.")`
- **Model Deep Dive** tab: shows architecture cards and hardcoded hyperparameter table; hides training curve chart

---

## Tab 1 — Dataset Explorer

**Components:**
- 4 KPI metric cards: Total Engines (100), Total Cycles (20,631), Selected Features (15), RUL Cap (125)
- Engine selector: `st.selectbox("Select Engine", range(1, 101))`
- Feature display: two-column layout — 15 selected (green checkmark) vs 9 removed (red ✗) with reason (low variance / low correlation)
- Sensor time-series chart: `st.selectbox` to pick a sensor, Plotly line chart of that sensor's values over cycles for the selected engine
- RUL degradation curve: Plotly line chart showing actual RUL (piecewise linear, capped at 125) vs cycle number for the selected engine

**Data flow:** `load_data()` → filter by `engine_id` → plot

---

## Tab 2 — Model Comparison

**Components:**
- Metrics table (hardcoded from report, always visible):

| Model | MAE | RMSE | R² | Parameters | FLOPs |
|---|---|---|---|---|---|
| Random Forest | 12.378 | 17.18 | 0.612 | 1,020,590 | 52,384,000 |
| MLP | 13.39 | 17.285 | 0.6072 | 18,177 | 455,950,336 | <!-- from final_model_comparison.csv -->
| **LSTM** ⭐ | **9.608** | **12.951** | **0.8482** | **296,065** | **126,178,997,888** |

- Grouped bar charts: MAE / RMSE / R² side-by-side (Plotly `go.Bar`, cobalt + gold + white color series)
- Scatter plots (requires model files): 3-column layout, one scatter per model, predicted vs actual, identity line in gold
- Residual distributions (requires model files): Plotly histogram overlay, all 3 models on same chart

**Note:** Predictions for scatter/residuals are generated at load time using test set + loaded models, cached with `@st.cache_data`.

---

## Tab 3 — RUL Predictor

**Components:**
- Test engine selector: `st.selectbox("Select Test Engine", range(1, 101))`
- Gauge visualization: custom Plotly `go.Indicator` gauge with three color zones:
  - Green: > 60 cycles
  - Yellow: 20–60 cycles
  - Red: < 20 cycles
- Two metric columns: Predicted RUL · Actual RUL
- Accuracy context: `st.metric("Prediction Error", f"±{abs(pred - actual):.1f} cycles")`
- Full prediction chart: Plotly line chart of predicted vs actual RUL across all cycles for the selected test engine (requires model)

**Live inference flow:**
1. Load test engine cycles from `test_FD001.txt`
2. Normalize using scaler fit on training data
3. Create sliding windows (`seq_len=75`)
4. Run `best_lstm.forward(windows)` → predictions
5. Compare final prediction to ground truth from `RUL_FD001.txt`

---

## Tab 4 — Model Deep Dive

**Components:**
- LSTM architecture card: hidden_size=256, num_layers=1, dropout=0.4, seq_len=75, lr=5e-4, asymmetric α=1.5, use_attention=False
- Sanity LSTM vs Best LSTM side-by-side comparison table (hardcoded metrics)
- Training curve: Plotly line chart, train loss + val loss vs epoch (data loaded from `tuning_log.csv` or hardcoded if unavailable)
- Hyperparameter heatmap: Plotly `go.Heatmap`, RMSE values across hidden_size × seq_len grid (from `tuning_log.csv`)
- CDF of absolute errors: Plotly line chart comparing Sanity LSTM vs Best LSTM (requires model files for live computation, fallback to static data)

---

## `utils/preprocessing.py`

```python
SELECTED_FEATURES = [
    'operational_setting_1',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
    'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
    'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]
REMOVED_FEATURES = [
    'operational_setting_2', 'operational_setting_3',
    'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10',
    'sensor_16', 'sensor_18', 'sensor_19'
]
MAX_RUL = 125
WINDOW_SIZE_LSTM = 75   # best LSTM
WINDOW_SIZE_RF_MLP = 30 # RF and MLP use 30×15=450-dim flat vector

# Functions:
# load_data(data_dir) -> (train_df, test_df, rul_series)
# compute_rul(df) -> df with 'rul' column, capped at MAX_RUL
# fit_scaler(train_df) -> StandardScaler
# normalize(df, scaler) -> normalized df
# make_windows(df, window_size, flatten=False) -> (X_np, y_np, engine_ids)
#   flatten=True returns shape (N, window*features) for RF/MLP
#   flatten=False returns shape (N, window, features) for LSTM
```

---

## `utils/models.py`

Exact class definitions copied from notebook (cells 63 and 84):
- `class MLP(nn.Module)` — architecture: Input(450) → FC(128) → BN → ReLU → Dropout → FC(64) → BN → ReLU → Dropout → FC(32) → BN → ReLU → Dropout → FC(1)
- `class LSTMRegressor(nn.Module)` — signature: `__init__(self, input_size, hidden_size, num_layers, dropout_rate, use_attention=False)`

---

## `notebook_export.py`

Standalone script. Run once from `rul-dashboard/` after activating the project venv.

**Steps:**
1. Load and preprocess data (same pipeline as notebook)
2. Fit StandardScaler on training data; save as `models/scaler.pkl`
3. Train best LSTM config (hidden=256, layers=1, dropout=0.4, seq_len=75, lr=5e-4, α=1.5) for 100 epochs
4. Save `models/best_lstm.pt` (state dict) + `models/lstm_config.json` (hyperparams)
5. Train best MLP config (hidden_layers=[128, 64, 32], dropout=0.3, lr=0.001) for 100 epochs
6. Save `models/mlp_model.pt`
7. Train RF (n_estimators=200, max_depth=20 or best from tuning_log.csv)
8. Save `models/rf_model.pkl`
9. Save `models/train_losses.json` and `models/val_losses.json` for training curves

---

## Visual Theme (CSS)

```css
/* Background */
.stApp { background-color: #060e1a; }

/* Metric cards */
[data-testid="metric-container"] {
  background: #0d1e35;
  border: 1px solid #00427A;
  border-top: 2px solid #FFC20E;
  border-radius: 6px;
  padding: 12px;
}

/* Tab active state */
[data-baseweb="tab"][aria-selected="true"] {
  background: #FFC20E;
  color: #00427A;
  font-weight: 800;
}

/* Plotly charts: dark background, gold accents */
/* Applied via layout= in each go.Figure() call */
```

Plotly template: `plotly_dark` with overrides — `paper_bgcolor="#060e1a"`, `plot_bgcolor="#0d1e35"`, primary color `#FFC20E`, secondary `#00427A`.

---

## `requirements.txt`

```
streamlit>=1.32
plotly>=5.20
pandas>=2.0
numpy<2
scikit-learn>=1.4
torch>=2.0
joblib>=1.3
```

---

## `notebook_export.py` RF Hyperparameters

RF best config comes from `tuning_log.csv` if it exists, otherwise defaults to: `n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1`.

---

## Out of Scope

- User authentication
- Deployment (Heroku, Streamlit Cloud, etc.)
- Uploading custom sensor data
- Model retraining from the UI
