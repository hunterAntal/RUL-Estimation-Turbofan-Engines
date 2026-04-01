"""Tests for utils/visualization.py chart builders."""
import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.preprocessing import COLUMN_NAMES, SELECTED_FEATURES, compute_rul
from utils.visualization import (
    sensor_time_series,
    rul_degradation_curve,
    metrics_bar_chart,
    scatter_pred_vs_actual,
    residuals_histogram,
    rul_gauge,
    prediction_trace,
    training_curve,
    cdf_absolute_errors,
)
import plotly.graph_objects as go


def _make_engine_df(n_cycles=60):
    """Build a synthetic engine DataFrame matching C-MAPSS column layout."""
    rows = [[1, c] + [0.0] * 3 + [float(i % 10) for i in range(1, 22)] for c in range(1, n_cycles + 1)]
    df = pd.DataFrame(rows, columns=COLUMN_NAMES)
    return compute_rul(df)


class TestSensorTimeSeries:
    """Tests for sensor_time_series function."""

    def test_returns_figure(self):
        """sensor_time_series returns a plotly Figure."""
        df = _make_engine_df()
        fig = sensor_time_series(df, "sensor_2")
        assert isinstance(fig, go.Figure)

    def test_has_one_trace(self):
        """sensor_time_series creates exactly 1 trace."""
        df = _make_engine_df()
        fig = sensor_time_series(df, "sensor_3")
        assert len(fig.data) == 1

    def test_title_contains_sensor_name(self):
        """Title includes sensor name and engine ID."""
        df = _make_engine_df()
        sensor_name = "sensor_7"
        fig = sensor_time_series(df, sensor_name)
        assert sensor_name in fig.layout.title.text
        assert "Engine 1" in fig.layout.title.text

    def test_x_axis_is_cycle(self):
        """X-axis should be cycle column."""
        df = _make_engine_df(n_cycles=50)
        fig = sensor_time_series(df, "sensor_11")
        trace = fig.data[0]
        assert list(trace.x) == list(range(1, 51))

    def test_y_axis_is_sensor_values(self):
        """Y-axis should match sensor column values."""
        df = _make_engine_df(n_cycles=30)
        sensor = "sensor_4"
        fig = sensor_time_series(df, sensor)
        trace = fig.data[0]
        expected_y = df[sensor].values
        np.testing.assert_array_equal(trace.y, expected_y)


class TestRulDegradationCurve:
    """Tests for rul_degradation_curve function."""

    def test_returns_figure(self):
        """rul_degradation_curve returns a plotly Figure."""
        df = _make_engine_df()
        fig = rul_degradation_curve(df)
        assert isinstance(fig, go.Figure)

    def test_has_one_trace_plus_hline_shape(self):
        """rul_degradation_curve has 1 trace (RUL line) + hline shape (RUL cap)."""
        df = _make_engine_df()
        fig = rul_degradation_curve(df)
        # One scatter trace for the RUL line
        assert len(fig.data) == 1
        # The hline is stored in layout.shapes, not as a trace
        assert len(fig.layout.shapes) > 0

    def test_first_trace_is_scatter(self):
        """First trace should be RUL scatter plot."""
        df = _make_engine_df()
        fig = rul_degradation_curve(df)
        assert isinstance(fig.data[0], go.Scatter)

    def test_x_axis_is_cycle(self):
        """X-axis should be cycle."""
        df = _make_engine_df(n_cycles=40)
        fig = rul_degradation_curve(df)
        trace = fig.data[0]
        assert list(trace.x) == list(range(1, 41))

    def test_y_axis_is_rul(self):
        """Y-axis should be RUL values."""
        df = _make_engine_df()
        fig = rul_degradation_curve(df)
        trace = fig.data[0]
        expected_y = df["RUL"].values
        np.testing.assert_array_equal(trace.y, expected_y)

    def test_title_contains_engine_id(self):
        """Title should mention engine ID."""
        df = _make_engine_df()
        fig = rul_degradation_curve(df)
        assert "Engine 1" in fig.layout.title.text


class TestMetricsBarChart:
    """Tests for metrics_bar_chart function."""

    def test_returns_figure(self):
        """metrics_bar_chart returns a plotly Figure."""
        models = ["RF", "MLP", "LSTM"]
        mae_vals = [10.5, 9.2, 8.7]
        rmse_vals = [14.2, 13.1, 12.5]
        r2_vals = [0.88, 0.91, 0.92]
        fig = metrics_bar_chart(models, mae_vals, rmse_vals, r2_vals)
        assert isinstance(fig, go.Figure)

    def test_has_nine_traces(self):
        """metrics_bar_chart has 9 traces (3 models × 3 metrics subplots)."""
        models = ["RF", "MLP", "LSTM"]
        mae_vals = [10.5, 9.2, 8.7]
        rmse_vals = [14.2, 13.1, 12.5]
        r2_vals = [0.88, 0.91, 0.92]
        fig = metrics_bar_chart(models, mae_vals, rmse_vals, r2_vals)
        # 3 models × 3 metrics (MAE, RMSE, R²) = 9 traces
        assert len(fig.data) == 9

    def test_all_traces_are_bar_type(self):
        """All traces should be Bar objects."""
        models = ["RF", "MLP"]
        mae_vals = [10.5, 9.2]
        rmse_vals = [14.2, 13.1]
        r2_vals = [0.88, 0.91]
        fig = metrics_bar_chart(models, mae_vals, rmse_vals, r2_vals)
        for trace in fig.data:
            assert isinstance(trace, go.Bar)

    def test_title_is_model_metrics_comparison(self):
        """Title should be 'Model Metrics Comparison'."""
        models = ["RF", "MLP", "LSTM"]
        mae_vals = [10.5, 9.2, 8.7]
        rmse_vals = [14.2, 13.1, 12.5]
        r2_vals = [0.88, 0.91, 0.92]
        fig = metrics_bar_chart(models, mae_vals, rmse_vals, r2_vals)
        assert "Metrics Comparison" in fig.layout.title.text


class TestScatterPredVsActual:
    """Tests for scatter_pred_vs_actual function."""

    def test_returns_figure(self):
        """scatter_pred_vs_actual returns a plotly Figure."""
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([11.0, 19.0, 31.0, 39.0])
        fig = scatter_pred_vs_actual(y_true, y_pred, "RF")
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self):
        """scatter_pred_vs_actual has 2 traces (predictions + identity line)."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 31.0])
        fig = scatter_pred_vs_actual(y_true, y_pred, "MLP")
        assert len(fig.data) == 2

    def test_first_trace_is_scatter(self):
        """First trace should be scatter plot of predictions."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 31.0])
        fig = scatter_pred_vs_actual(y_true, y_pred, "LSTM")
        assert isinstance(fig.data[0], go.Scatter)

    def test_second_trace_is_identity_line(self):
        """Second trace should be identity line."""
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([11.0, 19.0, 31.0])
        fig = scatter_pred_vs_actual(y_true, y_pred, "RF")
        identity_trace = fig.data[1]
        assert isinstance(identity_trace, go.Scatter)
        assert len(identity_trace.x) == 2  # just start and end points

    def test_title_contains_model_name(self):
        """Title should include model name."""
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([11.0, 19.0])
        model_name = "MyModel"
        fig = scatter_pred_vs_actual(y_true, y_pred, model_name)
        assert model_name in fig.layout.title.text


class TestResidualsHistogram:
    """Tests for residuals_histogram function."""

    def test_returns_figure(self):
        """residuals_histogram returns a plotly Figure."""
        residuals_dict = {
            "RF": np.random.randn(100),
            "MLP": np.random.randn(100),
        }
        fig = residuals_histogram(residuals_dict)
        assert isinstance(fig, go.Figure)

    def test_number_of_traces_matches_dict_length(self):
        """Number of traces should equal number of models in dict."""
        residuals_dict = {
            "RF": np.random.randn(50),
            "MLP": np.random.randn(50),
            "LSTM": np.random.randn(50),
        }
        fig = residuals_histogram(residuals_dict)
        assert len(fig.data) == 3

    def test_all_traces_are_histogram_type(self):
        """All traces should be Histogram objects."""
        residuals_dict = {
            "RF": np.random.randn(100),
            "MLP": np.random.randn(100),
        }
        fig = residuals_histogram(residuals_dict)
        for trace in fig.data:
            assert isinstance(trace, go.Histogram)

    def test_title_is_residual_distributions(self):
        """Title should be 'Residual Distributions'."""
        residuals_dict = {"RF": np.random.randn(50)}
        fig = residuals_histogram(residuals_dict)
        assert "Residual" in fig.layout.title.text


class TestRulGauge:
    """Tests for rul_gauge function."""

    def test_returns_figure(self):
        """rul_gauge returns a plotly Figure."""
        fig = rul_gauge(50.0)
        assert isinstance(fig, go.Figure)

    def test_has_one_indicator_trace(self):
        """rul_gauge has 1 Indicator trace."""
        fig = rul_gauge(75.0)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Indicator)

    def test_rul_above_60_is_green(self):
        """RUL > 60 should use green color #5cb85c."""
        fig = rul_gauge(80.0)
        indicator = fig.data[0]
        assert indicator.gauge.bar.color == "#5cb85c"

    def test_rul_20_to_60_is_amber(self):
        """RUL in [20, 60] should use amber color #e0a800."""
        fig = rul_gauge(40.0)
        indicator = fig.data[0]
        assert indicator.gauge.bar.color == "#e0a800"

    def test_rul_below_20_is_red(self):
        """RUL < 20 should use terracotta color #e94f37."""
        fig = rul_gauge(10.0)
        indicator = fig.data[0]
        assert indicator.gauge.bar.color == "#e94f37"

    def test_rul_exactly_60_is_amber(self):
        """RUL = 60 is on boundary, should be amber."""
        fig = rul_gauge(60.0)
        indicator = fig.data[0]
        assert indicator.gauge.bar.color == "#e0a800"

    def test_rul_exactly_20_is_amber(self):
        """RUL = 20 is on lower boundary, should be amber."""
        fig = rul_gauge(20.0)
        indicator = fig.data[0]
        assert indicator.gauge.bar.color == "#e0a800"

    def test_gauge_value_matches_input(self):
        """Gauge value should match input RUL."""
        rul = 45.5
        fig = rul_gauge(rul)
        indicator = fig.data[0]
        assert indicator.value == rul


class TestPredictionTrace:
    """Tests for prediction_trace function."""

    def test_returns_figure(self):
        """prediction_trace returns a plotly Figure."""
        cycles = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([100.0, 95.0, 90.0, 85.0, 80.0])
        fig = prediction_trace(cycles, y_pred)
        assert isinstance(fig, go.Figure)

    def test_single_trace_when_no_y_true(self):
        """prediction_trace has 1 trace when y_true is None."""
        cycles = np.array([1, 2, 3])
        y_pred = np.array([100.0, 95.0, 90.0])
        fig = prediction_trace(cycles, y_pred)
        assert len(fig.data) == 1

    def test_two_traces_when_y_true_provided(self):
        """prediction_trace has 2 traces when y_true is provided."""
        cycles = np.array([1, 2, 3])
        y_pred = np.array([100.0, 95.0, 90.0])
        y_true = np.array([102.0, 96.0, 88.0])
        fig = prediction_trace(cycles, y_pred, y_true)
        assert len(fig.data) == 2

    def test_first_trace_is_prediction(self):
        """First trace should be predicted RUL."""
        cycles = np.array([1, 2, 3])
        y_pred = np.array([100.0, 95.0, 90.0])
        fig = prediction_trace(cycles, y_pred)
        trace = fig.data[0]
        assert isinstance(trace, go.Scatter)
        np.testing.assert_array_equal(trace.x, cycles)
        np.testing.assert_array_equal(trace.y, y_pred)

    def test_second_trace_is_actual_when_provided(self):
        """Second trace should be actual RUL when provided."""
        cycles = np.array([1, 2, 3])
        y_pred = np.array([100.0, 95.0, 90.0])
        y_true = np.array([102.0, 96.0, 88.0])
        fig = prediction_trace(cycles, y_pred, y_true)
        trace = fig.data[1]
        assert isinstance(trace, go.Scatter)
        np.testing.assert_array_equal(trace.y, y_true)

    def test_title_contains_rul_prediction(self):
        """Title should mention RUL prediction."""
        cycles = np.array([1, 2])
        y_pred = np.array([100.0, 95.0])
        fig = prediction_trace(cycles, y_pred)
        assert "RUL" in fig.layout.title.text.upper()


class TestTrainingCurve:
    """Tests for training_curve function."""

    def test_returns_figure(self):
        """training_curve returns a plotly Figure."""
        train_losses = [0.5, 0.4, 0.3, 0.25, 0.2]
        val_losses = [0.55, 0.45, 0.35, 0.3, 0.28]
        fig = training_curve(train_losses, val_losses)
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(self):
        """training_curve has 2 traces (train + val)."""
        train_losses = [0.5, 0.4, 0.3]
        val_losses = [0.55, 0.45, 0.35]
        fig = training_curve(train_losses, val_losses)
        assert len(fig.data) == 2

    def test_both_traces_are_scatter(self):
        """Both traces should be Scatter objects."""
        train_losses = [0.5, 0.4, 0.3]
        val_losses = [0.55, 0.45, 0.35]
        fig = training_curve(train_losses, val_losses)
        assert isinstance(fig.data[0], go.Scatter)
        assert isinstance(fig.data[1], go.Scatter)

    def test_x_axis_is_epoch_numbers(self):
        """X-axis should be epoch numbers starting at 1."""
        train_losses = [0.5, 0.4, 0.3, 0.25]
        val_losses = [0.55, 0.45, 0.35, 0.3]
        fig = training_curve(train_losses, val_losses)
        expected_epochs = [1, 2, 3, 4]
        np.testing.assert_array_equal(fig.data[0].x, expected_epochs)

    def test_train_values_match_input(self):
        """Train loss values should match input."""
        train_losses = [0.5, 0.4, 0.3]
        val_losses = [0.55, 0.45, 0.35]
        fig = training_curve(train_losses, val_losses)
        np.testing.assert_array_equal(fig.data[0].y, train_losses)

    def test_val_values_match_input(self):
        """Validation loss values should match input."""
        train_losses = [0.5, 0.4, 0.3]
        val_losses = [0.55, 0.45, 0.35]
        fig = training_curve(train_losses, val_losses)
        np.testing.assert_array_equal(fig.data[1].y, val_losses)

    def test_title_is_lstm_training_curve(self):
        """Title should mention LSTM training."""
        train_losses = [0.5, 0.4]
        val_losses = [0.55, 0.45]
        fig = training_curve(train_losses, val_losses)
        assert "Training" in fig.layout.title.text


class TestCdfAbsoluteErrors:
    """Tests for cdf_absolute_errors function."""

    def test_returns_figure(self):
        """cdf_absolute_errors returns a plotly Figure."""
        errors_dict = {
            "RF": np.random.randn(100),
            "MLP": np.random.randn(100),
        }
        fig = cdf_absolute_errors(errors_dict)
        assert isinstance(fig, go.Figure)

    def test_number_of_traces_limited_to_two_colors(self):
        """Number of traces is limited to 2 (only TERRACOTTA and IVORY colors available)."""
        errors_dict = {
            "RF": np.random.randn(50),
            "MLP": np.random.randn(50),
            "LSTM": np.random.randn(50),
        }
        fig = cdf_absolute_errors(errors_dict)
        # Only 2 colors are defined, so only first 2 models get traces
        assert len(fig.data) == 2

    def test_all_traces_are_scatter(self):
        """All traces should be Scatter objects."""
        errors_dict = {
            "RF": np.random.randn(100),
            "MLP": np.random.randn(100),
        }
        fig = cdf_absolute_errors(errors_dict)
        for trace in fig.data:
            assert isinstance(trace, go.Scatter)

    def test_x_axis_is_sorted_absolute_errors(self):
        """X-axis should be sorted absolute errors."""
        errors = np.array([5, -3, 8, -1, 2])
        errors_dict = {"Model": errors}
        fig = cdf_absolute_errors(errors_dict)
        trace = fig.data[0]
        expected_x = np.sort(np.abs(errors))
        np.testing.assert_array_equal(trace.x, expected_x)

    def test_y_axis_is_cumulative_probability(self):
        """Y-axis should be CDF values from 1/n to 1."""
        n = 10
        errors_dict = {"Model": np.random.randn(n)}
        fig = cdf_absolute_errors(errors_dict)
        trace = fig.data[0]
        expected_y = np.arange(1, n + 1) / n
        np.testing.assert_array_almost_equal(trace.y, expected_y)

    def test_title_is_cdf_of_absolute_errors(self):
        """Title should mention CDF of absolute errors."""
        errors_dict = {"RF": np.random.randn(50)}
        fig = cdf_absolute_errors(errors_dict)
        assert "CDF" in fig.layout.title.text
        assert "Error" in fig.layout.title.text
