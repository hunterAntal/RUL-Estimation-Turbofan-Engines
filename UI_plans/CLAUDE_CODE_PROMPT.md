# Claude Code Prompt — RUL Prediction Dashboard

Read the PROJECT_CONTEXT.md file first for full project background.

## Task
Build a Streamlit web dashboard for our ESOF-4011 Applied Computational Intelligence project. This is a Remaining Useful Life (RUL) prediction system for turbofan engines using the NASA C-MAPSS dataset (FD001 subset). We have three trained models: Random Forest, MLP, and LSTM. The LSTM is our final selected model.

This UI is for a 10% bonus on our course grade. The rubric says: "Groups that design and implement a web-based, desktop, or mobile application user interface to support data-mining queries of the proposed solution and result visualization will be awarded up to 10% bonus marks."

## What the Dashboard Needs

### Page 1: Dataset Explorer
- Load and display the C-MAPSS FD001 dataset
- Show a summary table (number of engines, total cycles, features, etc.)
- Let the user select an engine ID from a dropdown
- Plot that engine's sensor readings over time (small multiples or selectable sensor)
- Show the RUL degradation curve for that engine
- Highlight the 15 selected features vs the 9 removed ones

### Page 2: Model Comparison
- Display a comparison table of all three models (RF, MLP, LSTM) with MAE, RMSE, R², parameters, FLOPs — use the exact numbers from PROJECT_CONTEXT.md
- Bar charts comparing the metrics side by side
- Show predicted vs actual scatter plot for each model (load from saved predictions or regenerate)
- Show residual distributions

### Page 3: RUL Predictor (Interactive)
- Let the user select a test engine
- Run the trained LSTM model on that engine's sensor data
- Display the predicted RUL vs actual RUL
- Show a gauge or progress bar visualization of remaining life
- Color-code: green (>60 cycles), yellow (20-60), red (<20)
- Show confidence context: "This prediction is within X cycles of actual"

### Page 4: Model Deep Dive
- Show the LSTM architecture details (hidden size, layers, dropout, etc.)
- Display the training curve (loss vs epoch)
- Show the hyperparameter search results
- Sanity LSTM vs Best LSTM comparison metrics
- CDF of absolute errors

## Technical Requirements
- Use Streamlit with st.sidebar for navigation
- Use Plotly for interactive charts (not matplotlib — we want hover, zoom, etc.)
- Use the Lakehead University color scheme: Cobalt blue (#00427A), Blaze gold (#FFC20E), white
- Dark theme preferred (st.set_page_config with dark theme)
- The app should work standalone — load data from CSV files, load model weights from .pt files
- Add the Lakehead Faculty of Engineering branding at the top
- Make it look polished, not like a default Streamlit app

## File Structure Expected
```
rul-dashboard/
├── app.py                  # Main Streamlit app
├── pages/                  # Multi-page Streamlit
│   ├── 1_Dataset_Explorer.py
│   ├── 2_Model_Comparison.py
│   ├── 3_RUL_Predictor.py
│   └── 4_Model_Deep_Dive.py
├── models/                 # Saved model weights
│   ├── best_lstm.pt
│   ├── mlp_model.pt
│   └── rf_model.pkl
├── data/                   # Dataset files
│   ├── train_FD001.txt
│   ├── test_FD001.txt
│   └── RUL_FD001.txt
├── utils/
│   ├── preprocessing.py    # Data loading, windowing, normalization
│   ├── models.py           # Model class definitions (LSTM, MLP)
│   └── visualization.py    # Shared plotting functions
├── requirements.txt
├── PROJECT_CONTEXT.md
└── README.md
```

## Model Definitions (PyTorch)

The LSTM model class looks like this:
```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=15, hidden_size=256, num_layers=1, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out).squeeze(-1)
```

The MLP model class:
```python
class MLPModel(nn.Module):
    def __init__(self, input_size=450, hidden_layers=[64, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_layers:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)
```

## Preprocessing Pipeline
```python
# Selected features (15 after feature selection)
SELECTED_FEATURES = [
    'operational_setting_1',
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8',
    'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14',
    'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21'
]

# RUL cap at 125 cycles (piecewise linear)
MAX_RUL = 125

# LSTM window size
WINDOW_SIZE_SANITY = 30
WINDOW_SIZE_BEST = 75

# Normalization: StandardScaler fit on training data only
```

## Data File Format
The C-MAPSS text files are space-delimited with no headers. Columns:
```
engine_id, cycle, op_setting_1, op_setting_2, op_setting_3, sensor_1, ..., sensor_21
```
The RUL file contains one RUL value per line (one per test engine).

## Important Notes
- If model weight files (.pt, .pkl) don't exist yet, the app should still work — show the comparison metrics from hardcoded values and display a message like "Upload model weights to enable live predictions"
- The dataset files (train_FD001.txt, test_FD001.txt, RUL_FD001.txt) can be downloaded from: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- Make the README explain how to set up and run the app
- Add a requirements.txt with: streamlit, plotly, pandas, numpy, scikit-learn, torch, joblib
