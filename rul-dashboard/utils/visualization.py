# rul-dashboard/utils/visualization.py
"""Shared Plotly chart builders. Each function returns a go.Figure."""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Colour scheme: Ivory / Fiery Terracotta / Gunmetal ───────────────────────
TERRACOTTA = "#e94f37"
IVORY      = "#f6f7eb"
GUNMETAL   = "#393e41"
BG_DARK    = "#2b2f31"       # slightly deeper gunmetal for page bg
BG_CARD    = "#393e41"       # gunmetal for card/plot bg
TEXT_DIM   = "#a8ada8"       # muted ivory-grey for gridlines / dim text

_LAYOUT = dict(
    paper_bgcolor=BG_DARK,
    plot_bgcolor=BG_CARD,
    font=dict(color=IVORY, family="system-ui", size=18),
    margin=dict(l=56, r=28, t=64, b=56),
    xaxis=dict(gridcolor="#4d5457", zerolinecolor="#4d5457",
               tickfont=dict(size=17), title_font=dict(size=19)),
    yaxis=dict(gridcolor="#4d5457", zerolinecolor="#4d5457",
               tickfont=dict(size=17), title_font=dict(size=19)),
    title_font=dict(size=23, color=IVORY),
)


def _apply_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(**_LAYOUT)
    return fig


def sensor_time_series(engine_df, sensor: str) -> go.Figure:
    """Line chart of a single sensor reading over cycles for one engine."""
    fig = go.Figure(go.Scatter(
        x=engine_df["cycle"], y=engine_df[sensor],
        mode="lines", line=dict(color=TERRACOTTA, width=3),
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
        line=dict(color=TERRACOTTA, width=3),
        marker=dict(size=4, color=TERRACOTTA),
        name="RUL",
    ))
    fig.add_hline(y=125, line_dash="dash", line_color=TEXT_DIM,
                  annotation_text="RUL cap (125)", annotation_position="top right",
                  annotation_font=dict(color=TEXT_DIM, size=13))
    fig.update_layout(
        title=f"RUL Degradation — Engine {engine_df['engine_id'].iloc[0]}",
        xaxis_title="Cycle", yaxis_title="Remaining Useful Life",
        **_LAYOUT,
    )
    return fig


def metrics_bar_chart(models, mae_vals, rmse_vals, r2_vals) -> go.Figure:
    """Grouped bar chart: MAE / RMSE / R² for each model."""
    # Terracotta for RF, Ivory for MLP, muted for LSTM
    colors = [TERRACOTTA, IVORY, TEXT_DIM]
    fig = make_subplots(rows=1, cols=3, subplot_titles=["MAE", "RMSE", "R²"])

    for col, (metric, vals) in enumerate(
        [("MAE", mae_vals), ("RMSE", rmse_vals), ("R²", r2_vals)], start=1
    ):
        for i, (model, val) in enumerate(zip(models, vals)):
            fig.add_trace(go.Bar(
                name=model, x=[model], y=[val],
                marker_color=colors[i],
                text=[f"{val:.3f}"], textposition="outside",
                textfont=dict(size=14),
                showlegend=(col == 1),
            ), row=1, col=col)

    fig.update_layout(
        barmode="group", title="Model Metrics Comparison",
        paper_bgcolor=BG_DARK, plot_bgcolor=BG_CARD,
        font=dict(color=IVORY, size=18), margin=dict(l=28, r=28, t=72, b=28),
        legend=dict(bgcolor=BG_CARD, bordercolor=TERRACOTTA, font=dict(size=17)),
        title_font=dict(size=23, color=IVORY),
    )
    fig.update_yaxes(gridcolor="#4d5457")
    fig.update_annotations(font_size=17)
    return fig


def scatter_pred_vs_actual(y_true, y_pred, model_name: str) -> go.Figure:
    """Predicted vs actual scatter with identity line."""
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(color=TERRACOTTA, size=5, opacity=0.65),
        name="Predictions",
    ))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color=IVORY, dash="dash", width=2),
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
    colors = [TERRACOTTA, IVORY, TEXT_DIM]
    fig = go.Figure()
    for (name, res), color in zip(residuals_dict.items(), colors):
        fig.add_trace(go.Histogram(
            x=res, name=name, opacity=0.65,
            marker_color=color, nbinsx=50,
        ))
    fig.update_layout(
        barmode="overlay", title="Residual Distributions",
        xaxis_title="Residual (Predicted − Actual)", yaxis_title="Count",
        **_LAYOUT,
        legend=dict(bgcolor=BG_CARD, bordercolor=TERRACOTTA, font=dict(size=17)),
    )
    return fig


def rul_gauge(predicted_rul: float) -> go.Figure:
    """Color-coded gauge: green >60, amber 20–60, red <20."""
    needle_color = "#5cb85c" if predicted_rul > 60 else ("#e0a800" if predicted_rul >= 20 else TERRACOTTA)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_rul,
        title={"text": "Predicted RUL (cycles)", "font": {"color": IVORY, "size": 22}},
        number={"font": {"color": needle_color, "size": 72}},
        gauge={
            "axis": {"range": [0, 125], "tickcolor": IVORY,
                     "tickfont": {"color": IVORY, "size": 17}},
            "bar": {"color": needle_color, "thickness": 0.3},
            "bgcolor": BG_CARD,
            "borderwidth": 1, "bordercolor": TERRACOTTA,
            "steps": [
                {"range": [0,  20], "color": "#4a2020"},
                {"range": [20, 60], "color": "#3d3020"},
                {"range": [60,125], "color": "#203320"},
            ],
            "threshold": {"line": {"color": IVORY, "width": 3}, "value": predicted_rul},
        },
    ))
    fig.update_layout(paper_bgcolor=BG_DARK, font=dict(color=IVORY, size=18),
                      margin=dict(l=28, r=28, t=72, b=28), height=340)
    return fig


def prediction_trace(cycles, y_pred, y_true=None) -> go.Figure:
    """Line chart: LSTM predictions vs actual RUL over cycles."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cycles, y=y_pred, mode="lines",
        line=dict(color=TERRACOTTA, width=3), name="Predicted RUL",
    ))
    if y_true is not None:
        fig.add_trace(go.Scatter(
            x=cycles, y=y_true, mode="lines",
            line=dict(color=IVORY, width=2, dash="dot"), name="Actual RUL",
        ))
    fig.update_layout(
        title="RUL Prediction vs Actual", xaxis_title="Cycle",
        yaxis_title="RUL", **_LAYOUT,
        legend=dict(bgcolor=BG_CARD, bordercolor=TERRACOTTA, font=dict(size=17)),
    )
    return fig


def training_curve(train_losses: list, val_losses: list) -> go.Figure:
    """Train and validation loss over epochs."""
    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_losses, mode="lines",
                             line=dict(color=TERRACOTTA, width=3), name="Train Loss"))
    fig.add_trace(go.Scatter(x=epochs, y=val_losses,  mode="lines",
                             line=dict(color=IVORY, width=2, dash="dot"), name="Val Loss"))
    fig.update_layout(
        title="LSTM Training Curve", xaxis_title="Epoch",
        yaxis_title="Loss", **_LAYOUT,
        legend=dict(bgcolor=BG_CARD, bordercolor=TERRACOTTA, font=dict(size=17)),
    )
    return fig


def cdf_absolute_errors(errors_dict: dict) -> go.Figure:
    """CDF of absolute errors for two or more models."""
    colors = [TERRACOTTA, IVORY]
    fig = go.Figure()
    for (name, errs), color in zip(errors_dict.items(), colors):
        sorted_e = np.sort(np.abs(errs))
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        fig.add_trace(go.Scatter(
            x=sorted_e, y=cdf, mode="lines",
            line=dict(color=color, width=3), name=name,
        ))
    fig.update_layout(
        title="CDF of Absolute Errors",
        xaxis_title="Absolute Error (cycles)", yaxis_title="CDF",
        **_LAYOUT,
        legend=dict(bgcolor=BG_CARD, bordercolor=TERRACOTTA, font=dict(size=17)),
    )
    return fig
