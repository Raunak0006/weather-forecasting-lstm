#!/usr/bin/env python3
"""
Synthetic Weather Dataset Generator for New Delhi, India.

Generates realistic hourly weather data (temperature, humidity, pressure, wind speed)
incorporating seasonal cycles, diurnal patterns, multi-day weather persistence,
and intentional missing-data gaps for testing preprocessing pipelines.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def generate_new_delhi_weather(
    start_date: str = "2020-01-01 00:00:00",
    end_date: str = "2023-12-31 23:00:00",
    seed: int = 42,
    insert_gaps: bool = True,
) -> pd.DataFrame:
    """Generate realistic hourly synthetic weather dataset for New Delhi.

    Parameters
    ----------
    start_date : str
        Start timestamp (inclusive), format 'YYYY-MM-DD HH:MM:SS'.
    end_date : str
        End timestamp (inclusive), format 'YYYY-MM-DD HH:MM:SS'.
    seed : int
        Random seed for reproducibility.
    insert_gaps : bool
        Whether to randomly drop a few hours to simulate sensor dropouts.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, temperature_celsius,
        humidity_percent, pressure_hpa, wind_speed_kmh.
    """
    rng = np.random.default_rng(seed)

    # 1. Complete hourly timeline
    date_range = pd.date_range(start=start_date, end=end_date, freq="h")
    total_hours = len(date_range)
    if total_hours == 0:
        raise ValueError("Date range produced 0 timestamps.")

    # Day of year and hour of day arrays
    day_of_year = date_range.dayofyear.to_numpy()
    hour_of_day = date_range.hour.to_numpy()

    # -------------------------------------------------------------
    # 2. Temperature modeling (°C)
    # New Delhi Climatology:
    # - Winter (Jan): Daily mean ~13°C (Min ~6°C, Max ~20°C)
    # - Pre-summer (Apr): Daily mean ~30°C (Min ~22°C, Max ~38°C)
    # - Peak Summer (Late May - June): Daily mean ~35°C (Min ~28°C, Max ~42-45°C)
    # - Monsoon (Jul-Aug): Daily mean ~31°C with damped diurnal range
    # - Autumn (Oct-Nov): Transitioning from 28°C down to 18°C
    # -------------------------------------------------------------
    # Annual cycle: peak heat ~ day 145-160 (late May/early June), trough ~ day 10 (early Jan)
    annual_phase = 2.0 * np.pi * (day_of_year - 150) / 365.25
    annual_mean_temp = 25.5 + 10.5 * np.cos(annual_phase)  # range ~15°C to 36°C

    # Diurnal amplitude varies by season (larger in dry summer/winter ~12-14°C, smaller in humid monsoon ~6-8°C)
    monsoon_factor = np.exp(-0.5 * ((day_of_year - 215) / 35.0) ** 2)  # Peak monsoon in early August
    diurnal_amp = 6.0 - 2.5 * monsoon_factor + 1.0 * np.cos(2.0 * np.pi * (day_of_year - 120) / 365.25)
    diurnal_amp = np.clip(diurnal_amp, 3.5, 7.5)

    # Diurnal cycle: minimum at ~05:30 (sunrise), maximum at ~15:00
    diurnal_phase = 2.0 * np.pi * (hour_of_day - 15.0) / 24.0
    diurnal_cycle = diurnal_amp * np.cos(diurnal_phase)

    # Autoregressive weather anomalies (heatwaves, cold waves, cloudy multi-day spells)
    ar_noise = np.zeros(total_hours)
    phi = 0.96  # High persistence across consecutive hours
    innovations = rng.normal(loc=0.0, scale=0.45, size=total_hours)
    for t in range(1, total_hours):
        ar_noise[t] = phi * ar_noise[t - 1] + innovations[t]

    temperature = annual_mean_temp + diurnal_cycle + ar_noise
    # Add slight high-frequency micro-variability
    temperature += rng.normal(loc=0.0, scale=0.25, size=total_hours)
    # Clip to realistic physical bounds for New Delhi (historical min ~2°C, max ~48°C)
    temperature = np.clip(temperature, 3.0, 47.5)

    # -------------------------------------------------------------
    # 3. Relative Humidity modeling (%)
    # Inversely correlated with temperature, high in monsoon and winter fog mornings
    # -------------------------------------------------------------
    # Base humidity: very high in monsoon (~75%), moderate in winter (~65%), low in summer (~25%)
    base_humidity = 48.0 + 30.0 * monsoon_factor - 18.0 * np.cos(annual_phase)
    # Diurnal variation: humidity peaks around sunrise (coldest), drops mid-afternoon
    humidity_diurnal = -18.0 * np.cos(diurnal_phase) * (1.0 - 0.4 * monsoon_factor)
    # Inverse response to temperature anomalies
    humidity_noise = -1.2 * ar_noise + rng.normal(loc=0.0, scale=2.5, size=total_hours)

    humidity = base_humidity + humidity_diurnal + humidity_noise
    humidity = np.clip(humidity, 10.0, 98.0)

    # -------------------------------------------------------------
    # 4. Atmospheric Surface Pressure modeling (hPa)
    # New Delhi elevation ~216m, typical sea level reduced pressure ~1000-1018 hPa
    # Higher in winter (~1015-1020 hPa), lower in summer/monsoon (~996-1005 hPa)
    # Semi-diurnal atmospheric solar tide (~1.5 hPa cycle, peaks at 10:00 & 22:00)
    # -------------------------------------------------------------
    annual_pressure = 1008.0 - 6.5 * np.cos(annual_phase)  # winter high, summer low
    semi_diurnal_tide = 1.2 * np.cos(4.0 * np.pi * (hour_of_day - 10.0) / 24.0)
    pressure_synoptic = -0.35 * ar_noise + rng.normal(loc=0.0, scale=0.4, size=total_hours)

    pressure = annual_pressure + semi_diurnal_tide + pressure_synoptic
    pressure = np.clip(pressure, 988.0, 1026.0)

    # -------------------------------------------------------------
    # 5. Wind Speed modeling (km/h)
    # Higher in summer afternoons (Loo winds, 15-30 km/h), lighter in winter nights (2-8 km/h)
    # -------------------------------------------------------------
    base_wind = 10.0 + 3.5 * np.cos(annual_phase + 0.5)
    diurnal_wind = 4.0 * np.cos(2.0 * np.pi * (hour_of_day - 14.0) / 24.0)
    wind_noise = np.abs(rng.normal(loc=0.0, scale=3.5, size=total_hours))

    wind_speed = base_wind + diurnal_wind + wind_noise
    # Occasional wind gusts / dust storms in May-June
    dust_storm_prob = 0.003 * np.clip((day_of_year - 120) / 45.0, 0, 1)
    storm_mask = rng.random(total_hours) < dust_storm_prob
    wind_speed[storm_mask] += rng.uniform(15.0, 30.0, size=np.sum(storm_mask))
    wind_speed = np.clip(wind_speed, 1.0, 55.0)

    # Build DataFrame
    df = pd.DataFrame(
        {
            "timestamp": date_range.strftime("%Y-%m-%d %H:%M:%S"),
            "temperature_celsius": np.round(temperature, 2),
            "humidity_percent": np.round(humidity, 1),
            "pressure_hpa": np.round(pressure, 1),
            "wind_speed_kmh": np.round(wind_speed, 1),
        }
    )

    # -------------------------------------------------------------
    # 6. Intentional Missing Data (Gaps) for Testing Pipeline
    # Introduce small realistic sensor dropouts (1-4 consecutive hours)
    # -------------------------------------------------------------
    if insert_gaps:
        # Predefined intentional gaps across different seasons
        # e.g., 2020 summer sensor maintenance, 2021 winter fog power glitch, etc.
        gap_indices_to_drop: List[int] = []

        # 6 specific realistic gap segments:
        candidate_gaps: List[Tuple[str, int]] = [
            ("2020-04-12 14:00:00", 2),   # 2-hour gap in summer 2020
            ("2020-09-03 08:00:00", 3),   # 3-hour monsoon gap
            ("2021-01-18 02:00:00", 4),   # 4-hour winter fog dropout
            ("2021-07-22 18:00:00", 2),   # 2-hour rainstorm outage
            ("2022-03-10 11:00:00", 1),   # 1-hour single missing sensor ping
            ("2022-11-05 06:00:00", 3),   # 3-hour autumn gap
            ("2023-05-28 15:00:00", 2),   # 2-hour summer peak gap
            ("2023-10-14 20:00:00", 1),   # 1-hour sensor restart
        ]

        timestamp_series = df["timestamp"].to_numpy()
        ts_to_idx = {ts: idx for idx, ts in enumerate(timestamp_series)}

        for start_ts, gap_len in candidate_gaps:
            if start_ts in ts_to_idx:
                start_i = ts_to_idx[start_ts]
                end_i = min(start_i + gap_len, total_hours)
                gap_indices_to_drop.extend(range(start_i, end_i))

        if gap_indices_to_drop:
            df = df.drop(index=list(set(gap_indices_to_drop))).reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic hourly weather dataset for New Delhi."
    )
    default_output_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )
    default_output_path = os.path.join(default_output_dir, "sample_weather_data.csv")

    parser.add_argument(
        "--output",
        type=str,
        default=default_output_path,
        help="Path where output CSV will be saved.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2020-01-01 00:00:00",
        help="Start timestamp (YYYY-MM-DD HH:MM:SS).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2023-12-31 23:00:00",
        help="End timestamp (YYYY-MM-DD HH:MM:SS).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible noise generation.",
    )
    parser.add_argument(
        "--no-gaps",
        action="store_true",
        help="Do not insert missing rows / sensor gaps.",
    )

    args = parser.parse_args()

    # Ensure destination directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating synthetic New Delhi weather dataset from {args.start_date} to {args.end_date}...")
    df = generate_new_delhi_weather(
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed,
        insert_gaps=not args.no_gaps,
    )

    df.to_csv(args.output, index=False)
    print(f"Dataset successfully created at: {args.output}")
    print(f"Total records: {len(df):,}")
    print(f"Features: {list(df.columns)}")
    print("Sample rows:")
    print(df.head(3))
    print("...")
    print(df.tail(3))


if __name__ == "__main__":
    main()
