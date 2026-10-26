# Synthetic Weather Dataset (New Delhi, India)

## Overview

This dataset contains realistic hourly synthetic meteorological observations for **New Delhi, India** spanning 4 full calendar years from **January 1, 2020 00:00:00** to **December 31, 2023 23:00:00**.

The synthetic dataset was engineered to faithfully mirror the subtropical, semi-arid continental climate of New Delhi (Köppen climate classification *Cwa/BSh*), capturing its distinctive seasonal shifts, strong diurnal cycles, and physical meteorological correlations.

---

## Dataset Specifications

- **Location:** New Delhi, India ($28.61^\circ\text{N}, 77.23^\circ\text{E}$, elevation $\approx 216\,\text{m}$)
- **Time Period:** `2020-01-01 00:00:00` to `2023-12-31 23:00:00`
- **Temporal Resolution:** Hourly ($1\,\text{h}$)
- **Total Expected Timestamps:** $1,461 \text{ days} \times 24 \text{ hours} = 35,064$ time steps
- **Actual Records:** $35,046$ rows (with $18$ intentional missing observations dropped across 8 gap segments for data engineering validation)
- **Format:** CSV with header row, comma-separated values

---

## Feature Definitions

| Column Name | Type | Unit | Description | Realistic Range | Mean (Approx.) |
|:---|:---:|:---:|:---|:---:|:---:|
| `timestamp` | String / Datetime | `YYYY-MM-DD HH:MM:SS` | Observation timestamp (IST / local time) | `2020-01-01` to `2023-12-31` | — |
| `temperature_celsius` | Float | $^\circ\text{C}$ | Ambient dry-bulb air temperature at 2 meters | $3.0^\circ\text{C}$ to $47.5^\circ\text{C}$ | $25.5^\circ\text{C}$ |
| `humidity_percent` | Float | $\%$ | Relative humidity | $10.0\%$ to $98.0\%$ | $55.0\%$ |
| `pressure_hpa` | Float | $\text{hPa}$ | Surface atmospheric barometric pressure | $988.0\,\text{hPa}$ to $1026.0\,\text{hPa}$ | $1008.0\,\text{hPa}$ |
| `wind_speed_kmh` | Float | $\text{km/h}$ | Wind speed at 10 meters above ground | $1.0\,\text{km/h}$ to $55.0\,\text{km/h}$ | $12.5\,\text{km/h}$ |

---

## Climatological Dynamics Modeled

1. **Seasonal Cycles (Annual Wave)**
   - **Winter (December – January):** Daily mean temperatures $\approx 12\text{--}15^\circ\text{C}$, night-time minimums down to $4\text{--}8^\circ\text{C}$, afternoon highs around $18\text{--}22^\circ\text{C}$. High early morning humidity ($80\text{--}95\%$) corresponding to Delhi winter radiation fog. High barometric pressure ($1015\text{--}1022\,\text{hPa}$).
   - **Pre-monsoon / Peak Summer (April – June):** Intense dry heat. Peak temperatures regularly reaching $40\text{--}46^\circ\text{C}$ during late May and early June. Relative humidity plummets to $15\text{--}30\%$. Afternoon hot westerly dust winds (*Loo*). Lower barometric pressure ($996\text{--}1004\,\text{hPa}$).
   - **Monsoon (July – September):** Influx of Arabian Sea and Bay of Bengal moisture. Temperatures moderate to $28\text{--}35^\circ\text{C}$, but relative humidity stays elevated between $70\text{--}95\%$. Diurnal temperature variation is damped due to cloud cover.
   - **Post-monsoon / Autumn (October – November):** Rapid cooling from October ($32^\circ\text{C}$) to November ($20^\circ\text{C}$), clear skies, dropping wind speeds.

2. **Diurnal Cycles (Daily Wave)**
   - Temperature reaches minimum just around sunrise ($05:00\text{--}06:00\,\text{IST}$) and maximum in mid-afternoon ($14:30\text{--}15:30\,\text{IST}$).
   - Diurnal temperature amplitude ranges between $6^\circ\text{C}$ (humid monsoon) and $14^\circ\text{C}$ (dry summer/winter).
   - Semi-diurnal atmospheric thermal solar tide ($\approx 1.2\,\text{hPa}$ oscillation peaking at $10:00$ and $22:00$).

3. **Weather Persistence & Noise**
   - Incorporates autoregressive $\text{AR}(1)$ noise ($\phi = 0.96$) to simulate persistent synoptic weather patterns (such as 3–5 day Western Disturbances, winter cold waves, and pre-monsoon heatwaves) rather than independent uncorrelated noise.

---

## Intentional Missing Data (Gaps)

To evaluate the robustness of missing-data handlers, imputation routines, and time-series resamplers, the dataset includes 8 intentional gap segments simulating real-world sensor dropouts and maintenance windows:

| Gap Start Timestamp | Duration | Season | Context |
|:---|:---:|:---|:---|
| `2020-04-12 14:00:00` | 2 hours | Summer 2020 | Periodic sensor maintenance |
| `2020-09-03 08:00:00` | 3 hours | Monsoon 2020 | Heavy rainfall telecommunication blip |
| `2021-01-18 02:00:00` | 4 hours | Winter 2021 | Fog-related solar battery drain |
| `2021-07-22 18:00:00` | 2 hours | Monsoon 2021 | Storm outage |
| `2022-03-10 11:00:00` | 1 hour | Spring 2022 | Single dropped transmission packet |
| `2022-11-05 06:00:00` | 3 hours | Autumn 2022 | Station recalibration |
| `2023-05-28 15:00:00` | 2 hours | Summer 2023 | High temperature thermal shutdown |
| `2023-10-14 20:00:00` | 1 hour | Autumn 2023 | Firmware restart |

**Recommended Pipeline Handling:**
Use `handle_missing_values(df, temperature_col, method='interpolate')` or time-indexed interpolation (`pd.DataFrame.interpolate(method='time')`) to smoothly impute missing records before sequence generation.

---

## How to Regenerate the Dataset

You can regenerate or customize the dataset using the generator script:

```bash
# Generate default 4-year dataset with standard gaps:
python scripts/generate_dataset.py

# Generate without missing gaps:
python scripts/generate_dataset.py --no-gaps

# Custom output destination and seed:
python scripts/generate_dataset.py --output data/custom_delhi_weather.csv --seed 123
```
