# rul-dashboard/utils/models.py
import torch
import torch.nn as nn


class AsymmetricMSELoss(nn.Module):
    """Penalizes underprediction (pred < target) by alpha, overprediction by 1.0.
    Matches NASA FD001 scoring convention: missing imminent failure is worse."""
    def __init__(self, alpha: float = 1.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err    = pred - target
        weight = torch.where(err < 0, self.alpha * torch.ones_like(err), torch.ones_like(err))
        return (weight * err ** 2).mean()


class MLP(nn.Module):
    """MLP for RUL regression.
    Architecture: Input → [FC → BN → ReLU → Dropout] × n_layers → FC(1)
    """
    def __init__(self, input_size: int, hidden_layers: list, dropout_rate: float):
        super().__init__()
        layers = []
        in_size = input_size
        for out_size in hidden_layers:
            layers.extend([
                nn.Linear(in_size, out_size),
                nn.BatchNorm1d(out_size),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate),
            ])
            in_size = out_size
        layers.append(nn.Linear(in_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class LSTMRegressor(nn.Module):
    """Stacked LSTM for RUL regression.
    Architecture: Input → LSTM → (attention or last timestep) → FC(64) → ReLU → Dropout → FC(1)
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout_rate: float, use_attention: bool = False):
        super().__init__()
        self.use_attention = use_attention
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        if use_attention:
            self.attn = nn.Linear(hidden_size, 1)
        self.fc_head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        if self.use_attention:
            scores  = self.attn(lstm_out)
            weights = torch.softmax(scores, dim=1)
            context = (weights * lstm_out).sum(dim=1)
        else:
            context = lstm_out[:, -1, :]
        return self.fc_head(context).squeeze(-1)
