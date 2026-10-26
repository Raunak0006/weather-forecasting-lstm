"""Automated Tests for ML Pipeline Components."""

import numpy as np
import pandas as pd
import pytest

from backend.ml.data_loader import (
    detect_temperature_column,
    detect_timestamp_column,
    get_dataset_info,
    load_csv,
    validate_dataset,
)
from backend.ml.evaluate import (
    calculate_mae,
    calculate_r2,
    calculate_rmse,
    persistence_baseline,
)
from backend.ml.preprocessing import (
    WeatherPreprocessor,
    check_temporal_frequency,
    create_sequences,
    handle_duplicates,
    handle_missing_values,
    parse_timestamps,
    split_data,
)


@pytest.fixture
def sample_raw_df():
    """Create sample DataFrame for testing."""
    dates = pd.date_range("2023-01-01", periods=100, freq="1H")
    temps = [20.0 + 5.0 * np.sin(i / 12.0 * np.pi) for i in range(100)]
    df = pd.DataFrame({"timestamp": dates, "temperature_celsius": temps})
    return df


def test_timestamp_parsing_and_sorting():
    """Verify timestamps are parsed and chronologically sorted."""
    df = pd.DataFrame(
        {
            "timestamp": ["2023-01-01 02:00:00", "2023-01-01 01:00:00"],
            "temperature_celsius": [15.0, 14.5],
        }
    )
    parsed = parse_timestamps(df, "timestamp")
    assert pd.api.types.is_datetime64_any_dtype(parsed["timestamp"])
    assert parsed["timestamp"].iloc[0] < parsed["timestamp"].iloc[1]


def test_handle_duplicates():
    """Verify duplicate timestamps are resolved."""
    df = pd.DataFrame(
        {
            "timestamp": ["2023-01-01 00:00:00", "2023-01-01 00:00:00"],
            "temperature_celsius": [10.0, 11.0],
        }
    )
    cleaned = handle_duplicates(df, "timestamp", keep="first")
    assert len(cleaned) == 1
    assert cleaned["temperature_celsius"].iloc[0] == 10.0


def test_handle_missing_values():
    """Verify missing values are imputed via interpolation."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=5, freq="1H"),
            "temperature_celsius": [10.0, np.nan, 14.0, np.nan, 18.0],
        }
    )
    imputed = handle_missing_values(df, "temperature_celsius", method="interpolate")
    assert imputed["temperature_celsius"].isna().sum() == 0
    assert imputed["temperature_celsius"].iloc[1] == 12.0
    assert imputed["temperature_celsius"].iloc[3] == 16.0


def test_sequence_creation():
    """Verify sliding window creates correct 3D input and target shapes."""
    data = np.arange(100, dtype=np.float32).reshape(-1, 1)
    lookback = 10
    horizon = 1
    X, y = create_sequences(data, lookback=lookback, horizon=horizon)

    expected_samples = 100 - lookback - horizon + 1
    assert X.shape == (expected_samples, lookback, 1)
    assert y.shape == (expected_samples, horizon)
    # Check that y is indeed the step following the lookback window
    np.testing.assert_array_equal(X[0, :, 0], np.arange(10))
    assert y[0, 0] == 10.0


def test_chronological_splitting():
    """Verify split_data does not shuffle and preserves temporal order."""
    X = np.arange(100).reshape(100, 1, 1)
    y = np.arange(100).reshape(100, 1)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(
        X, y, train_ratio=0.7, val_ratio=0.15
    )

    assert len(X_train) == 70
    assert len(X_val) == 15
    assert len(X_test) == 15
    # Strict temporal ordering
    assert y_train[-1, 0] < y_val[0, 0]
    assert y_val[-1, 0] < y_test[0, 0]


def test_scaler_fit_exclusively_on_train(sample_raw_df):
    """Verify that scaler is fitted ONLY on training split (no leakage)."""
    preprocessor = WeatherPreprocessor(lookback=12, horizon=1)
    processed = preprocessor.fit_transform(
        sample_raw_df,
        timestamp_col="timestamp",
        temperature_col="temperature_celsius",
        lookback=12,
        horizon=1,
        train_ratio=0.7,
        val_ratio=0.15,
    )

    # Scaler min and max must match the min and max of train_raw, NOT the whole dataset
    train_raw = processed["train_raw"]
    expected_min = float(train_raw.min())
    expected_max = float(train_raw.max())

    assert np.isclose(preprocessor.scaler.data_min_[0], expected_min)
    assert np.isclose(preprocessor.scaler.data_max_[0], expected_max)


def test_inverse_transform(sample_raw_df):
    """Verify inverse transformation accurately recovers original temperature values."""
    preprocessor = WeatherPreprocessor(lookback=12, horizon=1)
    preprocessor.fit_transform(
        sample_raw_df,
        timestamp_col="timestamp",
        temperature_col="temperature_celsius",
    )

    original_temps = np.array([[20.5], [25.0], [32.1]], dtype=np.float32)
    scaled = preprocessor.transform(original_temps)
    recovered = preprocessor.inverse_transform(scaled)

    np.testing.assert_allclose(original_temps, recovered, atol=1e-5)


def test_persistence_baseline():
    """Verify naive persistence baseline copies last observed step."""
    # Input shape: (samples=3, lookback=4, features=1)
    X_test = np.array(
        [
            [[10.0], [11.0], [12.0], [13.0]],
            [[15.0], [16.0], [17.0], [18.0]],
            [[20.0], [21.0], [22.0], [23.0]],
        ]
    )
    baseline_preds = persistence_baseline(X_test)
    expected = np.array([13.0, 18.0, 23.0])
    np.testing.assert_array_equal(baseline_preds, expected)


def test_evaluation_metrics():
    """Verify MAE, RMSE, and R2 calculation functions."""
    y_true = np.array([20.0, 22.0, 24.0, 26.0])
    y_pred = np.array([21.0, 22.0, 23.0, 27.0])

    # Absolute errors: 1, 0, 1, 1 -> Mean = 0.75
    mae = calculate_mae(y_true, y_pred)
    assert mae == 0.75

    # Squared errors: 1, 0, 1, 1 -> Mean = 0.75 -> RMSE = sqrt(0.75) ~ 0.8660
    rmse = calculate_rmse(y_true, y_pred)
    assert np.isclose(rmse, np.sqrt(0.75), atol=1e-3)

    # R2 should be positive and high
    r2 = calculate_r2(y_true, y_pred)
    assert 0.8 < r2 <= 1.0


def test_column_auto_detection():
    """Verify intelligent detection of time and temperature columns."""
    df1 = pd.DataFrame({"Date_Time": ["2023-01-01"], "Temp_C": [25.4]})
    assert detect_timestamp_column(df1) == "Date_Time"
    assert detect_temperature_column(df1) == "Temp_C"

    df2 = pd.DataFrame({"UTC_Timestamp": ["2023-01-01"], "air_temperature": [18.2]})
    assert detect_timestamp_column(df2) == "UTC_Timestamp"
    assert detect_temperature_column(df2) == "air_temperature"
