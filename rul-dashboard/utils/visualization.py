# rul-dashboard/utils/visualization.py
"""Shared Plotly chart builders. Each function returns a go.Figure."""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Lakehead theme ────────────────────────────────────────────────────────────
COBALT   = "#00427A"
GOLD     = "#FFC20E"
WHITE    = "#FFFFFF"
BG_DARK  = "#060e1a"
BG_CARD  = "#0d1e35"
TEXT_DIM = "#6688aa"

_LAYOUT = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_CARD,
    font=dict(color=WHITE, family="system-ui"),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_LAYOUT)
    return fig


def sensor_time_series(engine_df, sensor: str) -> go.Figure:
    """Line chart of a single sensor reading over cycles for one engine."""
    fig = go.Figure(go.Scatter(
        x=engine_df["cycle"], y=engine_df[sensor],
        mode="lines", line=dict(color=GOLD, width=2),
        name=sensor,
    ))
    fig.update_layout(
        title=f"{sensor} — Engine {engine_df['engine_id'].iloc[0]}",
        xaxis_title="Cycle", yaxis_title="Sensor Value",
        **_LAYOUT,
    )
    return fig


def rul_degradation_curve(engine_df) -> go.Figure:
    """Piecewise RUL curve (capped at 125) for a single engine."""
    fig = go.Figure(go.Scatter(
        x=engine_df["cycle"], y=engine_df["RUL"],
        mode="lines+markers",
        line=dict(color=GOLD, width=2),
        marker=dict(size=3, color=GOLD),
        name="RUL",
    ))
    fig.add_hline(y=125, line_dash="dash", line_color=TEXT_DIM,
                  annotation_text="RUL cap (125)", annotation_position="top right")
    fig.update_layout(
        title=f"RUL Degradation — Engine {engine_df['engine_id'].iloc[0]}",
        xaxis_title="Cycle", yaxis_title="Remaining Useful Life",
        **_LAYOUT,
    )
    return fig


def metrics_bar_chart(models, mae_vals, rmse_vals, r2_vals) -> go.Figure:
    """Grouped bar chart: MAE / RMSE / R² for each model."""
    colors = [COBALT, GOLD, WHITE]
    fig = make_subplots(rows=1, cols=3, subplot_titles=["MAE", "RMSE", "R²"])

    for col, (metric, vals) in enumerate(
        [("MAE", mae_vals), ("RMSE", rmse_vals), ("R²", r2_vals)], start=1
    ):
        for i, (model, val) in enumerate(zip(models, vals)):
            fig.add_trace(go.Bar(
                name=model, x=[model], y=[val],
                marker_color=colors[i],
                text=[f"{val:.3f}"], textposition="outside",
                showlegend=(col == 1),
            ), row=1, col=col)

    fig.update_layout(
        barmode="group", title="Model Metrics Comparison",
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_CARD,
        font=dict(color=WHITE), margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(bgcolor=BG_CARD, bordercolor=COBALT),
    )
    fig.update_yaxes(gridcolor="#1e3a5f")
    return fig


def scatter_pred_vs_actual(y_true, y_pred, model_name: str) -> go.Figure:
    """Predicted vs actual scatter with identity line."""
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color=COBALT, size=4, opacity=0.6),
        name="Predictions",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color=GOLD, dash="dash", width=2),
        name="Perfect prediction",
    ))
    fig.update_layout(
        title=f"{model_name} — Predicted vs Actual",
        xaxis_title="Actual RUL", yaxis_title="Predicted RUL",
        **_LAYOUT,
    )
    return fig


def residuals_histogram(residuals_dict: dict) -> go.Figure:
    """Overlaid residual distributions for multiple models."""
    colors = [COBALT, GOLD, WHITE]
    fig = go.Figure()
    for (name, res), color in zip(residuals_dict.items(), colors):
        fig.add_trace(go.Histogram(
            x=res, name=name, opacity=0.6,
            marker_color=color, nbinsx=50,
        ))
    fig.update_layout(
        barmode="overlay", title="Residual Distributions",
        xaxis_title="Residual (Predicted − Actual)", yaxis_title="Count",
        **_LAYOUT,
        legend=dict(bgcolor=BG_CARD, bordercolor=COBALT),
    )
    return fig


def rul_gauge(predicted_rul: float) -> go.Figure:
    """Color-coded gauge: green >60, yellow 20–60, red <20."""
    color = "#2ecc71" if predicted_rul > 60 else ("#f39c12" if predicted_rul >= 20 else "#e74c3c")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_rul,
        title={"text": "Predicted RUL (cycles)", "font": {"color": WHITE}},
        number={"font": {"color": color, "size": 48}},
        gauge={
            "axis": {"range": [0, 125], "tickcolor": WHITE,
                     "tickfont": {"color": WHITE}},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": BG_CARD,
            "borderwidth": 1, "bordercolor": COBALT,
            "steps": [
                {"range": [0,  20], "color": "#2d0a0a"},
                {"range": [20, 60], "color": "#2d1a00"},
                {"range": [60,125], "color": "#0a2d0a"},
            ],
            "threshold": {"line": {"color": GOLD, "width": 3}, "value": predicted_rul},
        },
    ))
    fig.update_layout(paper_bgcolor=BG_DARK, font=dict(color=WHITE),
                      margin=dict(l=20, r=20, t=60, b=20), height=280)
    return fig


def prediction_trace(cycles, y_pred, y_true=None) -> go.Figure:
    """Line chart: LSTM predictions vs actual RUL over cycles."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cycles, y=y_pred, mode="lines",
        line=dict(color=GOLD, width=2), name="Predicted RUL",
    ))
    if y_true is not None:
        fig.add_trace(go.Scatter(
            x=cycles, y=y_true, mode="lines",
            line=dict(color=COBALT, width=2, dash="dot"), name="Actual RUL",
        ))
    fig.update_layout(
        title="RUL Prediction vs Actual", xaxis_title="Cycle",
        yaxis_title="RUL", **_LAYOUT,
    )
    return fig


def training_curve(train_losses: list, val_losses: list) -> go.Figure:
    """Train and validation loss over epochs."""
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode="lines",
                             line=dict(color=GOLD, width=2), name="Train Loss"))
    fig.add_trace(go.Scatter(x=epochs, y=val_losses,  mode="lines",
                             line=dict(color=COBALT, width=2), name="Val Loss"))
    fig.update_layout(
        title="LSTM Training Curve", xaxis_title="Epoch",
        yaxis_title="Loss", **_LAYOUT,
        legend=dict(bgcolor=BG_CARD, bordercolor=COBALT),
    )
    return fig


def cdf_absolute_errors(errors_dict: dict) -> go.Figure:
    """CDF of absolute errors for two or more models."""
    colors = [GOLD, COBALT]
    fig = go.Figure()
    for (name, errs), color in zip(errors_dict.items(), colors):
        sorted_e = np.sort(np.abs(errs))
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        fig.add_trace(go.Scatter(
            x=sorted_e, y=cdf, mode="lines",
            line=dict(color=color, width=2), name=name,
        ))
    fig.update_layout(
        title="CDF of Absolute Errors",
        xaxis_title="Absolute Error (cycles)", yaxis_title="CDF",
        **_LAYOUT,
        legend=dict(bgcolor=BG_CARD, bordercolor=COBALT),
    )
    return fig
