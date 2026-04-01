# Turbofan RUL Dashboard

Streamlit dashboard for the ESOF-4011 Applied Computational Intelligence project.
Visualizes RUL predictions for NASA C-MAPSS FD001 using Random Forest, MLP, and LSTM models.

## Setup

### 1. Activate the project venv

```bash
cd /home/hunterantal/Dev/4011_RUL_Est_\ of_Turbofan_Engines
source venv/bin/activate
```

### 2. Install dashboard dependencies

```bash
cd rul-dashboard
pip install -r requirements.txt
```

### 3. Ensure dataset files are present

```bash
ls data/
# Expected: train_FD001.txt  test_FD001.txt  RUL_FD001.txt
```

If missing, copy from kagglehub cache:
```bash
CACHE=~/.cache/kagglehub/datasets/behrad3d/nasa-cmaps/versions/1/CMaps
cp $CACHE/train_FD001.txt $CACHE/test_FD001.txt $CACHE/RUL_FD001.txt data/
```

### 4. Export model weights (one-time, ~10 min on CPU)

```bash
python notebook_export.py
```

### 5. Run the dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Pages

| Tab | Content |
|---|---|
| 📊 Dataset Explorer | Engine selector, sensor time-series, RUL degradation curve, feature breakdown |
| 📈 Model Comparison | Metrics table, bar charts, scatter plots, residual distributions |
| 🔮 RUL Predictor | Live LSTM inference, RUL gauge, prediction vs actual trace |
| 🔍 Model Deep Dive | LSTM architecture, Sanity vs Best comparison, training curve, CDF |

## Notes

- Tabs 2–4 require model weights. Run `notebook_export.py` first.
- The dashboard works without weights: Tab 1 is fully functional, Tab 2 shows the hardcoded metrics table.
- Models are cached with `@st.cache_resource` — loaded once per Streamlit session.
