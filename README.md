# RUL Estimation of Turbofan Engines

Comparing three approaches — **Random Forest**, **MLP**, and **LSTM** — for Remaining Useful Life (RUL) estimation using NASA's C-MAPS FD001 dataset.

## Results

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Random Forest | — | — | — |
| MLP | — | — | — |
| LSTM (baseline) | 13.329 | — | 0.7989 |

> Results will be updated after the enhanced LSTM search completes.

## Key Design Choices

- **RUL capped at 125 cycles** — standard FD001 convention; engines treated as healthy before degradation onset
- **Train/test split by engine ID** — prevents leakage between sequential cycles of the same engine
- **18 of 27 features retained** — 7 sensors removed for low variance or low RUL correlation
- **30-cycle sliding windows** for LSTM; single-cycle flat vectors for RF and MLP

## LSTM Enhancements (v2)

On top of the baseline hyperparameter search (hidden size, layers, dropout, seq len, lr, weight decay), the following were added:

- **ReduceLROnPlateau** — halves LR when val RMSE plateaus (patience=5, factor=0.5)
- **Temporal attention** — soft attention over all LSTM timesteps instead of fixed last-step extraction
- **Asymmetric MSE loss** — penalizes underprediction more than overprediction, matching NASA's scoring convention

## Running

Developed for Google Colab (T4 GPU) but runs on CPU.

```bash
pip install kagglehub tqdm matplotlib pandas seaborn scikit-learn "numpy<2" torch
jupyter notebook Project_WorkSpace.ipynb
```

A Kaggle API key is required — the dataset downloads automatically via `kagglehub`.

## Dataset

[NASA C-MAPS FD001](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) — turbofan engine degradation simulation data.
