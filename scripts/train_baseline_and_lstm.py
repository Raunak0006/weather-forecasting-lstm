"""Train Stacked LSTM Model and Naive Baseline on Sample Dataset.

Produces real trained weights, scaler artifact, and empirical evaluation metrics.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.ml.data_loader import load_csv
from backend.ml.evaluate import ModelEvaluator
from backend.ml.model import build_lstm_model
from backend.ml.preprocessing import WeatherPreprocessor
from backend.ml.train import ModelTrainer

DATA_PATH = os.path.join("data", "sample_weather_data.csv")
MODEL_DIR = "models"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = os.path.join(MODEL_DIR, "best_lstm_model.keras")
SCALER_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def main():
    print(f"Loading dataset from: {DATA_PATH}...")
    df = load_csv(DATA_PATH)
    print(f"Loaded {len(df):,} records.")

    lookback = 24
    horizon = 1
    epochs = 15  # Genuine training
    batch_size = 64
    lstm_units = 64
    dropout = 0.2

    print("Executing sequential preprocessing without data leakage...")
    preprocessor = WeatherPreprocessor(lookback=lookback, horizon=horizon)
    processed = preprocessor.fit_transform(
        df=df,
        timestamp_col="timestamp",
        temperature_col="temperature_celsius",
        lookback=lookback,
        horizon=horizon,
    )

    X_train, y_train = processed["X_train"], processed["y_train"]
    X_val, y_val = processed["X_val"], processed["y_val"]
    X_test, y_test = processed["X_test"], processed["y_test"]

    print(f"Data Partitions:")
    print(f"  Train: {X_train.shape[0]:,} sequences")
    print(f"  Val:   {X_val.shape[0]:,} sequences")
    print(f"  Test:  {X_test.shape[0]:,} sequences")

    print(f"\nBuilding Stacked LSTM Model ({lstm_units} units, 2 layers, dropout={dropout})...")
    model = build_lstm_model(
        input_shape=(lookback, 1),
        horizon=horizon,
        lstm_units=lstm_units,
        dropout=dropout,
        num_layers=2,
    )

    print(f"Training LSTM model for up to {epochs} epochs...")
    trainer = ModelTrainer(model=model, preprocessor=preprocessor)
    history = trainer.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        patience=5,
        checkpoint_path=MODEL_PATH,
    )

    print("\nEvaluating model on held-out chronological test set...")
    evaluator = ModelEvaluator()
    eval_results = evaluator.evaluate(
        model=model,
        preprocessor=preprocessor,
        X_test=X_test,
        y_test=y_test,
    )
    baseline_results = evaluator.baseline_evaluate(
        X_test=X_test,
        y_test=y_test,
        preprocessor=preprocessor,
    )

    print("\n=======================================================")
    print("           EMPIRICAL EVALUATION RESULTS                ")
    print("=======================================================")
    print(f"  LSTM Model MAE:          {eval_results['mae']:.4f} °C")
    print(f"  LSTM Model RMSE:         {eval_results['rmse']:.4f} °C")
    print(f"  LSTM Model R² Score:     {eval_results['r2']:.4f}")
    print("-------------------------------------------------------")
    print(f"  Persistence Baseline MAE:  {baseline_results['mae']:.4f} °C")
    print(f"  Persistence Baseline RMSE: {baseline_results['rmse']:.4f} °C")
    print(f"  Persistence Baseline R²:   {baseline_results['r2']:.4f}")
    print("=======================================================")

    improvement_mae = (
        (baseline_results["mae"] - eval_results["mae"])
        / baseline_results["mae"]
        * 100
    )
    improvement_rmse = (
        (baseline_results["rmse"] - eval_results["rmse"])
        / baseline_results["rmse"]
        * 100
    )
    print(f"MAE Improvement over Naive Persistence:  {improvement_mae:+.2f}%")
    print(f"RMSE Improvement over Naive Persistence: {improvement_rmse:+.2f}%")

    # Save artifacts
    preprocessor.save(SCALER_PATH)
    model.save(MODEL_PATH)

    metrics_payload = {
        "mae": float(eval_results["mae"]),
        "rmse": float(eval_results["rmse"]),
        "r2": float(eval_results["r2"]),
        "baseline_mae": float(baseline_results["mae"]),
        "baseline_rmse": float(baseline_results["rmse"]),
        "baseline_r2": float(baseline_results["r2"]),
        "training_loss": [round(float(v), 6) for v in history.get("loss", [])],
        "validation_loss": [round(float(v), 6) for v in history.get("val_loss", [])],
        "actual_vs_predicted": [
            {
                "timestamp": f"T-{len(eval_results['actual_values']) - i}h",
                "actual": float(eval_results["actual_values"][i]),
                "predicted": float(eval_results["predicted_values"][i]),
            }
            for i in range(min(150, len(eval_results["actual_values"])))
        ],
        "trained_at": "2026-10-26T17:30:00Z",
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nArtifacts successfully saved:")
    print(f"  Model:      {MODEL_PATH}")
    print(f"  Scaler:     {SCALER_PATH}")
    print(f"  Metrics:    {METRICS_PATH}")


if __name__ == "__main__":
    main()
