"""
Data Loading and Validation Module for Weather Forecasting.

Provides robust CSV loading, schema validation, column auto-detection,
and dataset summary diagnostics.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def detect_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the timestamp column in a DataFrame.

    Checks prioritized column name candidates, followed by parsing heuristics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.

    Returns
    -------
    Optional[str]
        Detected column name or None if not found.
    """
    candidates = [
        "timestamp",
        "datetime",
        "date_time",
        "date",
        "time",
        "dt",
        "time_stamp",
        "utc_timestamp",
        "observation_time",
    ]
    lower_to_col = {col.strip().lower(): col for col in df.columns}

    # Direct match on common names
    for cand in candidates:
        if cand in lower_to_col:
            return lower_to_col[cand]

    # Partial match heuristic
    for col_lower, original_col in lower_to_col.items():
        if "time" in col_lower or "date" in col_lower:
            return original_col

    # Heuristic: test object/string columns with pd.to_datetime on a sample
    sample_size = min(len(df), 50)
    if sample_size > 0:
        for col in df.columns:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                try:
                    parsed = pd.to_datetime(df[col].iloc[:sample_size], errors="coerce")
                    if parsed.notna().sum() >= sample_size * 0.8:
                        return col
                except Exception:
                    continue

    return None


def detect_temperature_column(df: pd.DataFrame) -> Optional[str]:
    """Auto-detect the target temperature column in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.

    Returns
    -------
    Optional[str]
        Detected column name or None if not found.
    """
    candidates = [
        "temperature_celsius",
        "temperature",
        "temp_celsius",
        "temp_c",
        "temp",
        "t2m",
        "t_c",
        "air_temperature",
        "air_temp",
        "tavg",
    ]
    lower_to_col = {col.strip().lower(): col for col in df.columns}

    # Direct match on common names
    for cand in candidates:
        if cand in lower_to_col:
            return lower_to_col[cand]

    # Partial match heuristic: contains 'temp'
    for col_lower, original_col in lower_to_col.items():
        if "temp" in col_lower:
            return original_col

    # Fallback: check for first numeric column
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col

    return None


def load_csv(filepath: str, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV file with automatic separator and encoding fallback.

    Parameters
    ----------
    filepath : str
        Absolute or relative path to CSV file.
    **kwargs
        Additional arguments passed to pd.read_csv.

    Returns
    -------
    pd.DataFrame
        Loaded pandas DataFrame.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty or cannot be parsed as a valid tabular dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Weather dataset file not found at: {filepath}")

    if os.path.getsize(filepath) == 0:
        raise ValueError(f"Dataset file is empty (0 bytes): {filepath}")

    encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252"]
    separators = [",", ";", "\t", r"\s+"]

    last_error: Optional[Exception] = None

    for enc in encodings:
        for sep in separators:
            try:
                opts = {"encoding": enc, "sep": sep}
                opts.update(kwargs)
                df = pd.read_csv(filepath, **opts)
                if not df.empty and len(df.columns) >= 1:
                    # Strip leading/trailing whitespaces from column names
                    df.columns = [str(c).strip() for c in df.columns]
                    return df
            except Exception as e:
                last_error = e
                continue

    raise ValueError(
        f"Unable to parse CSV file '{filepath}' with standard encodings and delimiters. "
        f"Last error: {last_error}"
    )


def validate_dataset(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    temperature_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Validate dataset structure, schema, column presence, and data integrity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    timestamp_col : Optional[str]
        Name of timestamp column. If None, auto-detected.
    temperature_col : Optional[str]
        Name of temperature target column. If None, auto-detected.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing validation results, errors, warnings, and detected columns.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if df is None or not isinstance(df, pd.DataFrame):
        return {
            "is_valid": False,
            "errors": ["Input must be a valid pandas DataFrame."],
            "warnings": [],
            "row_count": 0,
            "column_count": 0,
            "detected_timestamp_col": None,
            "detected_temperature_col": None,
            "missing_values": {},
            "has_duplicates": False,
            "temporal_ordering": False,
            "details": {},
        }

    row_count, col_count = df.shape
    if row_count == 0:
        errors.append("Dataset has 0 rows (empty dataset).")
    if col_count == 0:
        errors.append("Dataset has 0 columns.")

    # 1. Resolve columns
    detected_ts = timestamp_col or detect_timestamp_column(df)
    detected_temp = temperature_col or detect_temperature_column(df)

    if detected_ts is None or detected_ts not in df.columns:
        errors.append("Timestamp column not found or could not be detected.")
    if detected_temp is None or detected_temp not in df.columns:
        errors.append("Temperature column not found or could not be detected.")

    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    has_duplicates = False
    temporal_ordering = True

    # 2. Timestamp-specific validations
    if detected_ts and detected_ts in df.columns:
        ts_series = df[detected_ts]
        null_ts = int(ts_series.isna().sum())
        if null_ts > 0:
            errors.append(f"Timestamp column '{detected_ts}' contains {null_ts} null/empty values.")

        try:
            parsed_ts = pd.to_datetime(ts_series, errors="coerce")
            unparseable_count = int(parsed_ts.isna().sum() - null_ts)
            if unparseable_count > 0:
                errors.append(
                    f"Timestamp column '{detected_ts}' has {unparseable_count} unparseable datetime entries."
                )
            else:
                has_duplicates = bool(parsed_ts.duplicated().any())
                if has_duplicates:
                    dup_count = int(parsed_ts.duplicated().sum())
                    warnings.append(
                        f"Found {dup_count} duplicate timestamps in '{detected_ts}'."
                    )

                # Check monotonic ordering
                if not parsed_ts.is_monotonic_increasing:
                    temporal_ordering = False
                    warnings.append(
                        f"Timestamps in '{detected_ts}' are not strictly sorted in chronological order."
                    )
        except Exception as e:
            errors.append(f"Failed to parse datetime entries in '{detected_ts}': {str(e)}")

    # 3. Temperature column validations
    if detected_temp and detected_temp in df.columns:
        temp_series = df[detected_temp]
        null_temp = int(temp_series.isna().sum())
        if null_temp > 0:
            warnings.append(
                f"Temperature column '{detected_temp}' contains {null_temp} missing (NaN) values."
            )

        numeric_temp = pd.to_numeric(temp_series, errors="coerce")
        non_numeric = int(numeric_temp.isna().sum() - null_temp)
        if non_numeric > 0:
            errors.append(
                f"Temperature column '{detected_temp}' contains {non_numeric} non-numeric values."
            )
        elif numeric_temp.notna().any():
            # Check for constant variance or unrealistic values
            t_min = float(numeric_temp.min())
            t_max = float(numeric_temp.max())
            t_std = float(numeric_temp.std()) if len(numeric_temp) > 1 else 0.0

            if t_std < 1e-6:
                warnings.append(
                    f"Temperature column '{detected_temp}' has near-zero variance (constant data)."
                )
            if t_min < -80.0 or t_max > 70.0:
                warnings.append(
                    f"Temperature values outside typical meteorological range: [{t_min:.1f}°C, {t_max:.1f}°C]."
                )

    # Sequence viability check
    min_required_rows = 30
    if row_count > 0 and row_count < min_required_rows:
        warnings.append(
            f"Dataset has only {row_count} rows. Recommended minimum is {min_required_rows} rows for LSTM sequencing."
        )

    is_valid = len(errors) == 0

    return {
        "is_valid": is_valid,
        "errors": errors,
        "warnings": warnings,
        "row_count": row_count,
        "column_count": col_count,
        "detected_timestamp_col": detected_ts,
        "detected_temperature_col": detected_temp,
        "missing_values": missing_counts,
        "has_duplicates": has_duplicates,
        "temporal_ordering": temporal_ordering,
        "details": {
            "columns": list(df.columns),
            "dtypes": {c: str(dtype) for c, dtype in df.dtypes.items()},
        },
    }


def get_dataset_info(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    temperature_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract comprehensive metadata, temporal frequency, and summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to analyze.
    timestamp_col : Optional[str]
        Name of timestamp column. If None, auto-detected.
    temperature_col : Optional[str]
        Name of temperature target column. If None, auto-detected.

    Returns
    -------
    Dict[str, Any]
        Dictionary with dataset metadata, descriptive stats, and time bounds.
    """
    ts_col = timestamp_col or detect_timestamp_column(df)
    temp_col = temperature_col or detect_temperature_column(df)

    info: Dict[str, Any] = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "column_dtypes": {c: str(dtype) for c, dtype in df.dtypes.items()},
        "timestamp_column": ts_col,
        "temperature_column": temp_col,
        "missing_summary": {c: int(df[c].isna().sum()) for c in df.columns},
        "time_range": None,
        "inferred_frequency": None,
        "numerical_statistics": {},
    }

    # Temporal range analysis
    if ts_col and ts_col in df.columns:
        try:
            parsed_ts = pd.to_datetime(df[ts_col], errors="coerce").dropna().sort_values()
            if not parsed_ts.empty:
                start_ts = parsed_ts.iloc[0]
                end_ts = parsed_ts.iloc[-1]
                duration_days = (end_ts - start_ts).total_seconds() / 86400.0

                # Compute median sampling interval
                diffs = parsed_ts.diff().dropna()
                median_seconds = diffs.dt.total_seconds().median() if not diffs.empty else None

                inferred_freq = None
                if median_seconds is not None:
                    if 3500 <= median_seconds <= 3700:
                        inferred_freq = "1H"
                    elif 85000 <= median_seconds <= 87000:
                        inferred_freq = "1D"
                    elif 1700 <= median_seconds <= 1900:
                        inferred_freq = "30min"
                    else:
                        inferred_freq = f"{median_seconds:.0f}s"

                info["time_range"] = {
                    "start": str(start_ts),
                    "end": str(end_ts),
                    "duration_days": round(duration_days, 2),
                    "median_interval_seconds": median_seconds,
                }
                info["inferred_frequency"] = inferred_freq
        except Exception:
            pass

    # Numerical statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        series = df[col].dropna()
        if not series.empty:
            stats[col] = {
                "min": round(float(series.min()), 2),
                "max": round(float(series.max()), 2),
                "mean": round(float(series.mean()), 2),
                "std": round(float(series.std()), 2) if len(series) > 1 else 0.0,
                "median": round(float(series.median()), 2),
            }
    info["numerical_statistics"] = stats

    return info
