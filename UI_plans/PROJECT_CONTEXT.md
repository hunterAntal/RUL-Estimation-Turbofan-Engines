# RUL Prediction Project — Context for Claude Code

## Project Overview
- **Course**: ESOF-4011 Applied Computational Intelligence, Lakehead University
- **Authors**: Felix Ikokwu and Hunter Antal
- **Problem**: Predict Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset
- **Approach**: Supervised regression — predict number of operational cycles remaining before engine failure

## Dataset
- **Name**: NASA Turbofan Engine Degradation Simulation (C-MAPSS), FD001 subset
- **Source**: https://www.kaggle.com/datasets/behrad3d/nasa-cmaps
- **Structure**:
  - ~100 engines in training set (complete run-to-failure)
  - Separate test set with partial degradation histories
  - Each row = one engine cycle with: engine_id, cycle, 3 operational settings, 21 sensors
- **After preprocessing**: 24 raw features reduced to 15 normalized features
- **Features removed** (low variance/covariance/correlation): operational_setting_2, operational_setting_3, sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19
- **Normalization**: StandardScaler (mean=0, std=1)
- **LSTM windowing**: sliding window of 30 cycles (sanity) / 75 cycles (best model)
- **RUL cap**: 125 cycles (piecewise linear degradation label)

## Three Models Implemented

### 1. Random Forest Regression (Baseline)
- Input: flattened window (30 × 15 = 450-dim vector)
- Parameters: 1,020,590
- FLOPs per sample: 52,384,000
- **Test results**: MAE=12.378, RMSE=17.18, R²=0.612

### 2. Multilayer Perceptron (MLP)
- Architecture: [450] → [64, 128, 64] → [1]
- Input: same flattened window as RF
- Parameters: 18,177
- FLOPs per sample: 455,950,336
- Dropout: 0.3, LR: 0.001
- **Test results**: MAE=12.067, RMSE=16.892, R²=0.6249

### 3. LSTM (Final Selected Model)
- **Sanity LSTM**: hidden=64, seq_len=30, dropout=0.2, lr=0.001, params=24,961
  - Test: MAE=12.066, RMSE=16.960, R²=0.6745
- **Best LSTM**: hidden=256, seq_len=75, dropout=0.4, lr=5e-4, params=296,065
  - Test: MAE=9.942, RMSE=13.331, R²=0.8392, 62.4% within 10 cycles
- Asymmetric loss with α=1.5 (penalizes overprediction 50% more)
- ReduceLROnPlateau scheduler
- FLOPs per sample: 22,563,666,432

## Evaluation Metrics Used
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² (R-squared)
- Residual distributions
- CDF of absolute errors
- FLOPs per sample
- Parameter count

## Key Figures/Visualizations in the Report
1. System pipeline diagram
2. Raw data variance histograms (all 24 features)
3. Raw sensor data small multiples (time series per engine)
4. Pre-normalization vs post-normalization feature distributions
5. Predicted vs Actual scatter plots (per model, validation + test)
6. Residual distribution histograms (per model, validation + test)
7. Hyperparameter heatmaps (per model)
8. Training curves (loss vs epoch, MLP and LSTM)
9. Sanity vs Best LSTM comparison (scatter, residuals, CDF)

## Tech Stack
- Python, PyTorch, Scikit-Learn, NumPy, Pandas, Matplotlib
- Google Colab for training
- Overleaf for the IEEE report

## Why This UI Matters
The course rubric says: "Groups that design and implement a web-based, desktop, or mobile application user interface to support data-mining queries of the proposed solution and result visualization will be awarded up to 10% bonus marks."

The UI should allow users to:
- Explore the dataset
- Select engines and view sensor degradation
- Run RUL predictions using the trained LSTM
- Compare model performance visually
- View all key metrics and analysis from the report
