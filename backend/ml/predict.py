"""
Inference and Forecasting Module for Weather LSTM Model.

Supports single-step prediction, multi-step autoregressive recursive forecasting,
and end-to-end forecasting from Pandas DataFrames.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import tensorflow as tf

from .preprocessing import WeatherPreprocessor, handle_missing_values, parse_timestamps


class WeatherPredictor:
    """Performs inference using trained LSTM model and fitted WeatherPreprocessor."""

    def __init__(
        self,
        model: tf.keras.Model,
        preprocessor: WeatherPreprocessor,
    ) -> None:
        """Initialize WeatherPredictor.

        Parameters
        ----------
        model : tf.keras.Model
            Trained Keras model.
        preprocessor : WeatherPreprocessor
            Fitted preprocessor containing scaling parameters and lookback configuration.
        """
        if model is None:
            raise ValueError("model cannot be None.")
        if preprocessor is None or not preprocessor.is_fitted:
            raise ValueError("preprocessor must be provided and already fitted.")

        self.model = model
        self.preprocessor = preprocessor
        self.lookback = preprocessor.lookback

    def _prepare_input_window(
        self,
        input_sequence: Union[np.ndarray, List[float]],
    ) -> np.ndarray:
        """Validate and format input sequence into scaled shape (1, lookback, 1).

        Parameters
        ----------
        input_sequence : Union[np.ndarray, List[float]]
            Historical temperature sequence of length equal to self.lookback.

        Returns
        -------
        np.ndarray
            Scaled 3D array of shape (1, lookback, 1).
        """
        arr = np.asarray(input_sequence, dtype=np.float32).ravel()
        if len(arr) != self.lookback:
            raise ValueError(
                f"Expected input sequence of length {self.lookback}, got length {len(arr)}."
            )

        # Scale raw temperature values using preprocessor
        scaled_2d = self.preprocessor.transform(arr.reshape(-1, 1))
        return scaled_2d.reshape(1, self.lookback, 1)

    def predict_single(
        self,
        input_sequence: Union[np.ndarray, List[float]],
    ) -> float:
        """Predict the immediate next temperature step.

        Parameters
        ----------
        input_sequence : Union[np.ndarray, List[float]]
            Input temperature history sequence of length `lookback` (°C).

        Returns
        -------
        float
            Predicted temperature in degrees Celsius (°C).
        """
        window_3d = self._prepare_input_window(input_sequence)
        # Model forward pass
        scaled_pred = self.model.predict(window_3d, verbose=0)
        # Extract immediate next step (index 0 if horizon >= 1)
        next_step_scaled = float(scaled_pred[0, 0])
        # Invert scale back to physical °C
        inv_val = self.preprocessor.inverse_transform(next_step_scaled)
        return round(float(inv_val), 2)

    def predict_multi_step(
        self,
        input_sequence: Union[np.ndarray, List[float]],
        steps: int = 24,
    ) -> List[float]:
        """Perform recursive autoregressive forecasting for multiple future steps.

        At each future time step, the model predicts the next value, which is then
        appended to the working window while removing the oldest observation.

        Parameters
        ----------
        input_sequence : Union[np.ndarray, List[float]]
            Initial history sequence of length `lookback` (°C).
        steps : int
            Number of future hourly steps to forecast.

        Returns
        -------
        List[float]
            List of forecasted temperatures (°C) for each future step.

        Raises
        ------
        ValueError
            If steps < 1.
        """
        if steps < 1:
            raise ValueError(f"Forecast steps must be >= 1, got {steps}")

        # Prepare initial scaled window: shape (1, lookback, 1)
        curr_window = self._prepare_input_window(input_sequence).copy()
        scaled_predictions: List[float] = []

        for _ in range(steps):
            # Predict next step
            pred = self.model.predict(curr_window, verbose=0)
            next_scaled_val = float(pred[0, 0])
            scaled_predictions.append(next_scaled_val)

            # Slide window forward: shift left by 1 and insert new prediction at end
            curr_window = np.roll(curr_window, shift=-1, axis=1)
            curr_window[0, -1, 0] = next_scaled_val

        # Inverse transform all predictions simultaneously
        scaled_arr = np.array(scaled_predictions, dtype=np.float32).reshape(-1, 1)
        unscaled = self.preprocessor.inverse_transform(scaled_arr)

        return [round(float(val), 2) for val in unscaled.ravel()]

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        timestamp_col: Optional[str] = None,
        temperature_col: Optional[str] = None,
        steps: int = 24,
    ) -> pd.DataFrame:
        """Generate future forecasts using the most recent observations from a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Weather DataFrame.
        timestamp_col : Optional[str]
            Name of timestamp column. Defaults to preprocessor config.
        temperature_col : Optional[str]
            Name of temperature column. Defaults to preprocessor config.
        steps : int
            Number of steps ahead to predict. Default is 24.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'timestamp' and 'predicted_temperature_celsius'.

        Raises
        ------
        ValueError
            If DataFrame has fewer records than the required lookback.
        """
        ts_col = timestamp_col or self.preprocessor.timestamp_col
        temp_col = temperature_col or self.preprocessor.temperature_col

        if ts_col not in df.columns:
            raise KeyError(f"Timestamp column '{ts_col}' not found in DataFrame.")
        if temp_col not in df.columns:
            raise KeyError(f"Temperature column '{temp_col}' not found in DataFrame.")

        # Clean and sort chronologically
        df_clean = parse_timestamps(df, timestamp_col=ts_col)
        df_clean = handle_missing_values(df_clean, temperature_col=temp_col)

        if len(df_clean) < self.lookback:
            raise ValueError(
                f"DataFrame has {len(df_clean)} records, but lookback requires at least {self.lookback}."
            )

        # Extract the latest sequence of length lookback
        latest_slice = df_clean.iloc[-self.lookback :]
        input_sequence = latest_slice[temp_col].to_numpy(dtype=np.float32)

        # Generate predictions
        forecasts = self.predict_multi_step(input_sequence, steps=steps)

        # Infer temporal step size for future timestamps
        ts_series = pd.to_datetime(df_clean[ts_col])
        last_timestamp = ts_series.iloc[-1]

        # Calculate step delta (default to 1 hour if single row or non-uniform)
        if len(ts_series) >= 2:
            step_delta = ts_series.iloc[-1] - ts_series.iloc[-2]
            if step_delta.total_seconds() <= 0:
                step_delta = pd.Timedelta(hours=1)
        else:
            step_delta = pd.Timedelta(hours=1)

        future_timestamps = [
            last_timestamp + (i + 1) * step_delta for i in range(steps)
        ]

        return pd.DataFrame(
            {
                "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in future_timestamps],
                "predicted_temperature_celsius": forecasts,
            }
        )
