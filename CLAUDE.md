# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project comparing three approaches (Random Forest, MLP, LSTM) for **Remaining Useful Life (RUL) estimation of turbofan engines** using NASA's C-MAPS FD001 dataset.

The entire project lives in a single Jupyter notebook: `Project_WorkSpace.ipynb`.

## Running the Notebook

This notebook was developed for **Google Colab with T4 GPU**, but runs on CPU as well.

**Install dependencies (done inline in notebook cells):**
```bash
pip install kagglehub tqdm matplotlib pandas seaborn scikit-learn "numpy<2" torch
```

**Launch locally:**
```bash
jupyter notebook Project_WorkSpace.ipynb
```

The notebook downloads the dataset automatically from Kaggle (`nasa-cmaps`) via `kagglehub`. A Kaggle API key must be configured.

## Notebook Architecture (120 cells)

The notebook is organized into five sequential phases — run cells in order:

| Phase | Cells | Purpose |
|-------|-------|---------|
| Data Loading & Preprocessing | 1–26 | Download data, EDA, feature selection, normalization, windowing |
| Random Forest | 27–56 | Grid search, evaluation, feature importance |
| MLP | 57–76 | PyTorch feed-forward net, hyperparameter tuning |
| LSTM | 77–95 | PyTorch recurrent net on 30-cycle windows, hyperparameter tuning |
| Comparison & Evaluation | 96–119 | Summary table, scatter plots, metrics bar charts |

## Key Design Decisions

- **RUL capping at 125 cycles** — standard for FD001; engines are treated as healthy before degradation onset.
- **Train/test split by engine ID** — prevents data leakage between sequential cycles of the same engine.
- **Feature selection**: 7 sensors removed (sensors 1, 6, 16, 18, 19 + 2 others) due to low variance or low correlation with RUL; 18 of 27 features retained.
- **LSTM windowing**: 30-cycle sliding windows capture temporal degradation patterns; RF and MLP use single-cycle flat vectors.
- **MLP architecture**: Input → FC(128) → BN → ReLU → Dropout → FC(64) → BN → ReLU → Dropout → FC(32) → BN → ReLU → Dropout → FC(1)
- **LSTM architecture**: Stacked LSTM layers → last timestep → FC(64) → ReLU → Dropout → FC(1)

## Evaluation Metrics

Models are compared on: RMSE, MAE, R². Visualizations (saved as PNG) include training curves, residual distributions, hyperparameter heatmaps, and predicted vs. actual scatter plots.
