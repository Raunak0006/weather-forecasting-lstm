"""
Preprocessing Module for Weather Forecasting LSTM Pipeline.

Handles chronological timestamp parsing, missing-value imputation, deduplication,
sliding-window sequence creation, train/val/test sequential splitting, and
leakage-free MinMaxScaler preprocessing.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def parse_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Parse timestamp column to datetime, sort chronologically, and reset index.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    timestamp_col : str
        Name of timestamp column.

    Returns
    -------
    pd.DataFrame
        DataFrame with parsed datetime column, sorted in ascending order.

    Raises
    ------
    KeyError
        If timestamp_col is not present in df.
    ValueError
        If timestamps cannot be converted to valid datetime objects.
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found in DataFrame.")

    df_clean = df.copy()
    parsed = pd.to_datetime(df_clean[timestamp_col], errors="coerce")
    invalid_mask = parsed.isna()
    if invalid_mask.all():
        raise ValueError(f"All values in timestamp column '{timestamp_col}' are invalid or unparseable.")

    if invalid_mask.any():
        # Drop rows with unparseable timestamps
        df_clean = df_clean.loc[~invalid_mask].copy()
        parsed = parsed.loc[~invalid_mask]

    df_clean[timestamp_col] = parsed
    df_clean = df_clean.sort_values(by=timestamp_col).reset_index(drop=True)
    return df_clean


def handle_duplicates(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    keep: str = "first",
) -> pd.DataFrame:
    """Remove duplicate timestamp entries.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    timestamp_col : str
        Column to check for duplicate timestamps.
    keep : str
        'first', 'last', or False (drop all duplicates).

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    if timestamp_col not in df.columns:
        return df.copy()

    df_dedup = df.drop_duplicates(subset=[timestamp_col], keep=keep).reset_index(drop=True)
    return df_dedup


def handle_missing_values(
    df: pd.DataFrame,
    temperature_col: str = "temperature_celsius",
    method: str = "interpolate",
) -> pd.DataFrame:
    """Handle missing (NaN) values in temperature and meteorological columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    temperature_col : str
        Target temperature column name.
    method : str
        Imputation method: 'interpolate' (linear), 'ffill', 'bfill', or 'drop'.

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed or dropped missing values.
    """
    if temperature_col not in df.columns:
        raise KeyError(f"Target column '{temperature_col}' not found in DataFrame.")

    df_imputed = df.copy()

    if method == "drop":
        df_imputed = df_imputed.dropna(subset=[temperature_col]).reset_index(drop=True)
        return df_imputed

    # Ensure numeric conversion
    df_imputed[temperature_col] = pd.to_numeric(df_imputed[temperature_col], errors="coerce")

    if method == "interpolate":
        # Linear interpolation with boundary fallbacks
        df_imputed[temperature_col] = (
            df_imputed[temperature_col]
            .interpolate(method="linear", limit_direction="both")
            .bfill()
            .ffill()
        )
    elif method == "ffill":
        df_imputed[temperature_col] = df_imputed[temperature_col].ffill().bfill()
    elif method == "bfill":
        df_imputed[temperature_col] = df_imputed[temperature_col].bfill().ffill()
    else:
        raise ValueError(f"Unsupported missing value handling method: '{method}'.")

    # Also impute other numeric columns if present
    num_cols = df_imputed.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df_imputed[col].isna().any():
            df_imputed[col] = (
                df_imputed[col]
                .interpolate(method="linear", limit_direction="both")
                .bfill()
                .ffill()
            )

    return df_imputed


def check_temporal_frequency(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> Dict[str, Any]:
    """Inspect timestamp step sizes, regularity, and locate missing gaps.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with timestamp column.
    timestamp_col : str
        Column name for timestamps.

    Returns
    -------
    Dict[str, Any]
        Information on sampling frequency, regularity, and identified gaps.
    """
    if timestamp_col not in df.columns or len(df) < 2:
        return {
            "inferred_freq": None,
            "median_interval_seconds": None,
            "is_regular": True,
            "gaps_count": 0,
            "gap_details": [],
        }

    ts = pd.to_datetime(df[timestamp_col], errors="coerce").dropna().sort_values()
    diffs = ts.diff().dropna()
    diff_seconds = diffs.dt.total_seconds().to_numpy()

    if len(diff_seconds) == 0:
        return {
            "inferred_freq": None,
            "median_interval_seconds": None,
            "is_regular": True,
            "gaps_count": 0,
            "gap_details": [],
        }

    median_step = float(np.median(diff_seconds))
    # A gap occurs when the delta exceeds 1.5x the median step
    gap_threshold = 1.5 * median_step if median_step > 0 else 1.0
    gap_mask = diff_seconds > gap_threshold
    gap_indices = np.where(gap_mask)[0]

    gap_details: List[Dict[str, Any]] = []
    for idx in gap_indices[:50]:  # Limit details to 50 entries
        prev_time = ts.iloc[idx]
        next_time = ts.iloc[idx + 1]
        delta_sec = float(diff_seconds[idx])
        gap_details.append(
            {
                "gap_start": str(prev_time),
                "gap_end": str(next_time),
                "duration_hours": round(delta_sec / 3600.0, 2),
            }
        )

    inferred_freq: Optional[str] = None
    if 3500 <= median_step <= 3700:
        inferred_freq = "1H"
    elif 85000 <= median_step <= 87000:
        inferred_freq = "1D"
    elif 1700 <= median_step <= 1900:
        inferred_freq = "30min"
    else:
        inferred_freq = f"{median_step:.0f}s"

    return {
        "inferred_freq": inferred_freq,
        "median_interval_seconds": median_step,
        "is_regular": len(gap_indices) == 0,
        "gaps_count": int(len(gap_indices)),
        "gap_details": gap_details,
    }


def create_sequences(
    data: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct sliding-window sequences for recurrent neural networks.

    Parameters
    ----------
    data : np.ndarray
        Input 1D array of shape (N,) or 2D array of shape (N, F).
    lookback : int
        Number of historical time steps used for input window X.
    horizon : int
        Number of future time steps to forecast y.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        X: array of shape (samples, lookback, features)
        y: array of shape (samples, horizon)

    Raises
    ------
    ValueError
        If inputs are invalid or data length is insufficient for lookback + horizon.
    """
    if lookback < 1:
        raise ValueError(f"lookback must be at least 1, got {lookback}")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")

    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    total_len = len(arr)
    required_len = lookback + horizon
    if total_len < required_len:
        raise ValueError(
            f"Data length ({total_len}) is smaller than required lookback + horizon ({required_len})."
        )

    num_samples = total_len - lookback - horizon + 1
    num_features = arr.shape[1]

    # Preallocate output arrays for speed and memory efficiency
    X = np.empty((num_samples, lookback, num_features), dtype=np.float32)
    y = np.empty((num_samples, horizon), dtype=np.float32)

    for i in range(num_samples):
        X[i] = arr[i : i + lookback]
        # Target is the next 'horizon' values of feature 0 (target temperature)
        y[i] = arr[i + lookback : i + lookback + horizon, 0]

    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform sequential chronological train/validation/test split.

    Preserves strict temporal order to prevent lookahead data leakage.

    Parameters
    ----------
    X : np.ndarray
        Input sequences of shape (samples, lookback, features).
    y : np.ndarray
        Target values of shape (samples, horizon).
    train_ratio : float
        Proportion of samples allocated to training set.
    val_ratio : float
        Proportion of samples allocated to validation set.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_train, y_train, X_val, y_val, X_test, y_test

    Raises
    ------
    ValueError
        If train_ratio + val_ratio >= 1.0 or samples are insufficient.
    """
    if train_ratio <= 0.0 or val_ratio <= 0.0:
        raise ValueError("train_ratio and val_ratio must be strictly positive.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("Sum of train_ratio and val_ratio must be strictly less than 1.0.")

    n_samples = len(X)
    if n_samples < 3:
        raise ValueError(f"Insufficient samples ({n_samples}) to split into 3 subsets.")

    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    # Ensure at least 1 sample in each split
    train_end = max(1, train_end)
    val_end = max(train_end + 1, val_end)
    val_end = min(n_samples - 1, val_end)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


class WeatherPreprocessor:
    """Preprocesses weather data and handles scaling without data leakage.

    Fits MinMaxScaler exclusively on training partition and applies consistent
    transformations and sequence generation across validation and test partitions.
    """

    def __init__(
        self,
        lookback: int = 24,
        horizon: int = 1,
        feature_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Initialize WeatherPreprocessor with sequence lengths and scaling range."""
        self.lookback = lookback
        self.horizon = horizon
        self.feature_range = feature_range

        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.timestamp_col: str = "timestamp"
        self.temperature_col: str = "temperature_celsius"
        self.is_fitted: bool = False
        self.feature_names: List[str] = []

    def fit_transform(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        temperature_col: str = "temperature_celsius",
        lookback: Optional[int] = None,
        horizon: Optional[int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Dict[str, Any]:
        """Clean, scale without leakage, and create sequences for train/val/test splits.

        Parameters
        ----------
        df : pd.DataFrame
            Raw weather DataFrame.
        timestamp_col : str
            Name of timestamp column.
        temperature_col : str
            Name of temperature column.
        lookback : Optional[int]
            Lookback sequence length (overrides init value if provided).
        horizon : Optional[int]
            Forecasting horizon steps (overrides init value if provided).
        train_ratio : float
            Training split fraction.
        val_ratio : float
            Validation split fraction.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test'
            - 'train_raw', 'val_raw', 'test_raw'
            - 'processed_df'
            - 'scaler'
        """
        if lookback is not None:
            self.lookback = lookback
        if horizon is not None:
            self.horizon = horizon

        self.timestamp_col = timestamp_col
        self.temperature_col = temperature_col

        # 1. Clean timestamps and sort
        df_clean = parse_timestamps(df, timestamp_col=self.timestamp_col)
        df_clean = handle_duplicates(df_clean, timestamp_col=self.timestamp_col)
        df_clean = handle_missing_values(df_clean, temperature_col=self.temperature_col)

        # 2. Extract series array
        raw_values = df_clean[self.temperature_col].to_numpy(dtype=np.float32).reshape(-1, 1)

        # 3. Calculate chronological split indices on raw series
        n_raw = len(raw_values)
        if n_raw < (self.lookback + self.horizon + 5):
            raise ValueError(
                f"Dataset length {n_raw} is too short for lookback={self.lookback} and horizon={self.horizon}."
            )

        train_cutoff = int(n_raw * train_ratio)
        val_cutoff = int(n_raw * (train_ratio + val_ratio))

        train_raw = raw_values[:train_cutoff]
        val_raw = raw_values[train_cutoff:val_cutoff]
        test_raw = raw_values[val_cutoff:]

        # 4. FIT SCALER EXCLUSIVELY ON TRAINING DATA (No data leakage)
        self.scaler.fit(train_raw)
        self.is_fitted = True
        self.feature_names = [self.temperature_col]

        # 5. Transform all partitions with fitted scaler
        scaled_all = self.scaler.transform(raw_values)

        # 6. Create sequences on the full continuous series to preserve window continuity
        X_all, y_all = create_sequences(scaled_all, lookback=self.lookback, horizon=self.horizon)

        # 7. Split sequence arrays chronologically
        X_train, y_train, X_val, y_val, X_test, y_test = split_data(
            X_all, y_all, train_ratio=train_ratio, val_ratio=val_ratio
        )

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "train_raw": train_raw,
            "val_raw": val_raw,
            "test_raw": test_raw,
            "processed_df": df_clean,
            "scaler": self.scaler,
            "config": {
                "lookback": self.lookback,
                "horizon": self.horizon,
                "timestamp_col": self.timestamp_col,
                "temperature_col": self.temperature_col,
                "train_ratio": train_ratio,
                "val_ratio": val_ratio,
            },
        }

    def transform(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Transform temperature data using the fitted scaler.

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            Data containing temperature values.

        Returns
        -------
        np.ndarray
            Scaled array of shape (-1, 1).

        Raises
        ------
        RuntimeError
            If preprocessor has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("WeatherPreprocessor must be fitted before calling transform.")

        if isinstance(data, pd.DataFrame):
            if self.temperature_col not in data.columns:
                raise KeyError(f"Column '{self.temperature_col}' not found in DataFrame.")
            arr = data[self.temperature_col].to_numpy(dtype=np.float32).reshape(-1, 1)
        else:
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)

        return self.scaler.transform(arr)

    def inverse_transform(
        self,
        scaled_values: Union[np.ndarray, List[float], float],
    ) -> np.ndarray:
        """Invert scaled values back to the original physical scale (°C).

        Handles scalars, 1D, 2D, and 3D input arrays gracefully.

        Parameters
        ----------
        scaled_values : Union[np.ndarray, List[float], float]
            Scaled temperature values.

        Returns
        -------
        np.ndarray
            Original scale temperatures (°C) matching the input dimension/shape.

        Raises
        ------
        RuntimeError
            If preprocessor has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("WeatherPreprocessor must be fitted before calling inverse_transform.")

        arr = np.asarray(scaled_values, dtype=np.float32)
        original_shape = arr.shape

        if arr.ndim == 0:
            flat = arr.reshape(1, 1)
            inv = self.scaler.inverse_transform(flat)
            return np.float32(inv[0, 0])

        if arr.ndim == 1:
            flat = arr.reshape(-1, 1)
            inv = self.scaler.inverse_transform(flat)
            return inv.reshape(original_shape)

        if arr.ndim == 2:
            # If 2D (samples, horizon), invert each horizon step
            n_samples, n_steps = arr.shape
            flat = arr.reshape(-1, 1)
            inv = self.scaler.inverse_transform(flat)
            return inv.reshape(n_samples, n_steps)

        if arr.ndim == 3:
            # 3D: (samples, timesteps, features)
            n_samples, n_timesteps, n_features = arr.shape
            flat = arr.reshape(-1, n_features)
            inv = self.scaler.inverse_transform(flat)
            return inv.reshape(n_samples, n_timesteps, n_features)

        flat = arr.reshape(-1, 1)
        inv = self.scaler.inverse_transform(flat)
        return inv.reshape(original_shape)

    def save(self, filepath: str) -> None:
        """Serialize preprocessor configuration and scaler state to disk.

        Parameters
        ----------
        filepath : str
            Target file path (supports .joblib or .pkl).
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            "scaler": self.scaler,
            "lookback": self.lookback,
            "horizon": self.horizon,
            "feature_range": self.feature_range,
            "timestamp_col": self.timestamp_col,
            "temperature_col": self.temperature_col,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, filepath)

    @classmethod
    def load(cls, filepath: str) -> "WeatherPreprocessor":
        """Load a saved preprocessor from disk.

        Parameters
        ----------
        filepath : str
            Path to saved preprocessor file.

        Returns
        -------
        WeatherPreprocessor
            Restored and fitted preprocessor instance.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found at: {filepath}")

        payload = joblib.load(filepath)
        instance = cls(
            lookback=payload.get("lookback", 24),
            horizon=payload.get("horizon", 1),
            feature_range=payload.get("feature_range", (0.0, 1.0)),
        )
        instance.scaler = payload["scaler"]
        instance.timestamp_col = payload.get("timestamp_col", "timestamp")
        instance.temperature_col = payload.get("temperature_col", "temperature_celsius")
        instance.is_fitted = payload.get("is_fitted", True)
        instance.feature_names = payload.get("feature_names", [])
        return instance
