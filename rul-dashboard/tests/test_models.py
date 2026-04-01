# rul-dashboard/tests/test_models.py
import torch
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.models import LSTMRegressor, MLP, AsymmetricMSELoss


def test_lstm_forward_shape():
    model = LSTMRegressor(input_size=15, hidden_size=64, num_layers=1, dropout_rate=0.0)
    x = torch.randn(8, 30, 15)  # batch=8, seq=30, features=15
    out = model(x)
    assert out.shape == (8,), f"expected (8,), got {out.shape}"


def test_lstm_best_config_forward():
    model = LSTMRegressor(input_size=15, hidden_size=256, num_layers=1,
                          dropout_rate=0.4, use_attention=False)
    x = torch.randn(4, 75, 15)
    out = model(x)
    assert out.shape == (4,)


def test_lstm_attention_forward():
    model = LSTMRegressor(input_size=15, hidden_size=64, num_layers=1,
                          dropout_rate=0.2, use_attention=True)
    x = torch.randn(4, 30, 15)
    out = model(x)
    assert out.shape == (4,)


def test_mlp_forward_shape():
    model = MLP(input_size=450, hidden_layers=[64, 128, 64], dropout_rate=0.0)
    x = torch.randn(8, 450)
    out = model(x)
    assert out.shape == (8, 1), f"expected (8, 1), got {out.shape}"


def test_asymmetric_loss_underprediction_penalized_more():
    loss_fn = AsymmetricMSELoss(alpha=1.5)
    # underprediction: pred < target → error < 0 → alpha applies
    pred_under = torch.tensor([5.0])
    pred_over  = torch.tensor([15.0])
    target     = torch.tensor([10.0])
    loss_under = loss_fn(pred_under, target).item()  # error=-5 → 1.5 * 25 = 37.5
    loss_over  = loss_fn(pred_over, target).item()   # error=+5 → 1.0 * 25 = 25.0
    assert loss_under > loss_over


def test_lstm_parameter_count():
    model = LSTMRegressor(input_size=15, hidden_size=256, num_layers=1,
                          dropout_rate=0.4, use_attention=False)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 296065, f"expected 296065 params, got {n_params}"
