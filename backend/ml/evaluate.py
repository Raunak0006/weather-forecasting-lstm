"""
Evaluation Module for Weather Forecasting LSTM Model.

Calculates key forecasting metrics (MAE, RMSE, R2), establishes the naive
persistence baseline, and performs comparative performance evaluations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocessing import WeatherPreprocessor


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE) between ground truth and predictions.

    Parameters
    ----------
    actual : np.ndarray
        Ground truth actual values.
    predicted : np.ndarray
        Model forecasted values.

    Returns
    -------
    float
        MAE in original measurement units (°C).
    """
    y_true = np.asarray(actual).ravel()
    y_pred = np.asarray(predicted).ravel()
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    return round(float(mean_absolute_error(y_true, y_pred)), 4)


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE) between ground truth and predictions.

    Parameters
    ----------
    actual : np.ndarray
        Ground truth actual values.
    predicted : np.ndarray
        Model forecasted values.

    Returns
    -------
    float
        RMSE in original measurement units (°C).
    """
    y_true = np.asarray(actual).ravel()
    y_pred = np.asarray(predicted).ravel()
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    mse = mean_squared_error(y_true, y_pred)
    return round(float(np.sqrt(mse)), 4)


def calculate_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Coefficient of Determination (R² Score).

    Parameters
    ----------
    actual : np.ndarray
        Ground truth actual values.
    predicted : np.ndarray
        Model forecasted values.

    Returns
    -------
    float
        R² score bounded in (-inf, 1.0].
    """
    y_true = np.asarray(actual).ravel()
    y_pred = np.asarray(predicted).ravel()
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0

    # Guard against zero variance in ground truth
    variance = float(np.var(y_true))
    if variance < 1e-9:
        # If ground truth is constant, return 1.0 if prediction matches, else 0.0
        return 1.0 if np.allclose(y_true, y_pred) else 0.0

    score = r2_score(y_true, y_pred)
    return round(float(score), 4)


def persistence_baseline(test_data: np.ndarray) -> np.ndarray:
    """Compute naive persistence baseline predictions.

    For sliding window input sequences X_test of shape (N, lookback, features),
    the persistence baseline carries forward the most recent observed value
    at the end of the lookback window (X_test[:, -1, 0]).
    For 1D sequential series, it lags by 1 step.

    Parameters
    ----------
    test_data : np.ndarray
        Input sequences array X of shape (N, lookback, features) or 1D array.

    Returns
    -------
    np.ndarray
        Baseline predictions aligned with test targets.
    """
    arr = np.asarray(test_data)
    if arr.ndim == 3:
        # X array of shape (samples, lookback, features): last known value is index -1
        # Feature 0 is the target temperature
        return arr[:, -1, 0].copy()
    elif arr.ndim == 2:
        return arr[:, -1].copy()
    elif arr.ndim == 1:
        # 1D array: baseline prediction is previous value
        baseline = np.empty_like(arr)
        if len(arr) > 0:
            baseline[0] = arr[0]
            baseline[1:] = arr[:-1]
        return baseline
    else:
        raise ValueError(f"Unsupported array dimension for persistence baseline: {arr.ndim}")


def evaluate_model(
    model: tf.keras.Model,
    preprocessor: WeatherPreprocessor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Any]:
    """Evaluate a trained LSTM model against a test dataset in physical units (°C).

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    preprocessor : WeatherPreprocessor
        Fitted preprocessor for inverse transformations.
    X_test : np.ndarray
        Test sequence inputs of shape (samples, lookback, features).
    y_test : np.ndarray
        Test target values (scaled).

    Returns
    -------
    Dict[str, Any]
        Dictionary with MAE, RMSE, R2, actual values, and predicted values.
    """
    if len(X_test) == 0 or len(y_test) == 0:
        raise ValueError("X_test and y_test cannot be empty.")

    # 1. Forward inference on test sequences
    y_pred_scaled = model.predict(X_test, verbose=0)

    # 2. Invert scaling to return to true °C scale
    y_pred_actual = preprocessor.inverse_transform(y_pred_scaled)
    y_true_actual = preprocessor.inverse_transform(y_test)

    # 3. Calculate regression metrics
    mae = calculate_mae(y_true_actual, y_pred_actual)
    rmse = calculate_rmse(y_true_actual, y_pred_actual)
    r2 = calculate_r2(y_true_actual, y_pred_actual)

    # 4. Calculate Mean Bias Error (MBE): positive indicates overprediction, negative underprediction
    mbe = round(float(np.mean(np.asarray(y_pred_actual).ravel() - np.asarray(y_true_actual).ravel())), 4)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_bias_error": mbe,
        "actual_values": [round(float(v), 2) for v in np.asarray(y_true_actual).ravel()],
        "predicted_values": [round(float(v), 2) for v in np.asarray(y_pred_actual).ravel()],
        "sample_count": len(X_test),
    }


def compare_with_baseline(
    model_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare LSTM model performance against the persistence baseline.

    Parameters
    ----------
    model_metrics : Dict[str, Any]
        Metrics dict containing 'mae', 'rmse', 'r2'.
    baseline_metrics : Dict[str, Any]
        Baseline metrics dict containing 'mae', 'rmse', 'r2'.

    Returns
    -------
    Dict[str, Any]
        Comparative analysis including percentage error reductions and outperformance flag.
    """
    b_mae = float(baseline_metrics.get("mae", 0.0))
    m_mae = float(model_metrics.get("mae", 0.0))
    b_rmse = float(baseline_metrics.get("rmse", 0.0))
    m_rmse = float(model_metrics.get("rmse", 0.0))
    b_r2 = float(baseline_metrics.get("r2", 0.0))
    m_r2 = float(model_metrics.get("r2", 0.0))

    # Calculate percentage reduction in errors
    mae_improvement_pct = round(((b_mae - m_mae) / b_mae * 100.0) if b_mae > 0 else 0.0, 2)
    rmse_improvement_pct = round(((b_rmse - m_rmse) / b_rmse * 100.0) if b_rmse > 0 else 0.0, 2)
    r2_gain = round(m_r2 - b_r2, 4)

    is_better = m_rmse < b_rmse and m_mae < b_mae

    return {
        "model_mae": m_mae,
        "baseline_mae": b_mae,
        "mae_improvement_pct": mae_improvement_pct,
        "model_rmse": m_rmse,
        "baseline_rmse": b_rmse,
        "rmse_improvement_pct": rmse_improvement_pct,
        "model_r2": m_r2,
        "baseline_r2": b_r2,
        "r2_gain": r2_gain,
        "outperforms_baseline": is_better,
        "summary": (
            f"LSTM model achieves {mae_improvement_pct}% lower MAE and {rmse_improvement_pct}% "
            f"lower RMSE compared to persistence baseline (R²: {m_r2} vs {b_r2})."
            if is_better
            else f"LSTM model did not outperform persistence baseline on test partition."
        ),
    }


class ModelEvaluator:
    """Comprehensive evaluation manager for LSTM and persistence baselines."""

    def __init__(self) -> None:
        """Initialize ModelEvaluator."""
        self.last_model_results: Optional[Dict[str, Any]] = None
        self.last_baseline_results: Optional[Dict[str, Any]] = None

    def evaluate(
        self,
        model: tf.keras.Model,
        preprocessor: WeatherPreprocessor,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Evaluate trained model on test data in physical temperature units.

        Parameters
        ----------
        model : tf.keras.Model
            Trained LSTM model.
        preprocessor : WeatherPreprocessor
            Fitted preprocessor.
        X_test : np.ndarray
            Test sequences.
        y_test : np.ndarray
            Test targets.

        Returns
        -------
        Dict[str, Any]
            Model evaluation metrics and predicted arrays.
        """
        results = evaluate_model(model, preprocessor, X_test, y_test)
        self.last_model_results = results
        return results

    def baseline_evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        preprocessor: WeatherPreprocessor,
    ) -> Dict[str, Any]:
        """Evaluate naive persistence baseline on the same test sequence inputs.

        Parameters
        ----------
        X_test : np.ndarray
            Test sequence inputs.
        y_test : np.ndarray
            Ground truth test targets (scaled).
        preprocessor : WeatherPreprocessor
            Fitted preprocessor for inverse transforming.

        Returns
        -------
        Dict[str, Any]
            Baseline evaluation metrics and baseline predicted values.
        """
        # Baseline predicts last observed value in sequence (scaled)
        baseline_preds_scaled = persistence_baseline(X_test)
        if baseline_preds_scaled.ndim == 1:
            baseline_preds_scaled = baseline_preds_scaled.reshape(-1, 1)

        # Inverse transform to °C
        baseline_preds_actual = preprocessor.inverse_transform(baseline_preds_scaled)
        y_true_actual = preprocessor.inverse_transform(y_test)

        mae = calculate_mae(y_true_actual, baseline_preds_actual)
        rmse = calculate_rmse(y_true_actual, baseline_preds_actual)
        r2 = calculate_r2(y_true_actual, baseline_preds_actual)

        results = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "actual_values": [round(float(v), 2) for v in np.asarray(y_true_actual).ravel()],
            "predicted_values": [round(float(v), 2) for v in np.asarray(baseline_preds_actual).ravel()],
            "sample_count": len(X_test),
        }
        self.last_baseline_results = results
        return results

    def comparison(
        self,
        model_results: Optional[Dict[str, Any]] = None,
        baseline_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform comparison between model and baseline evaluation results.

        Parameters
        ----------
        model_results : Optional[Dict[str, Any]]
            Results from evaluate(). If None, uses cached results.
        baseline_results : Optional[Dict[str, Any]]
            Results from baseline_evaluate(). If None, uses cached results.

        Returns
        -------
        Dict[str, Any]
            Side-by-side comparison dict with improvement percentages.
        """
        m_res = model_results or self.last_model_results
        b_res = baseline_results or self.last_baseline_results

        if m_res is None or b_res is None:
            raise RuntimeError(
                "Both model_results and baseline_results must be provided or evaluated before comparison."
            )

        return compare_with_baseline(m_res, b_res)
