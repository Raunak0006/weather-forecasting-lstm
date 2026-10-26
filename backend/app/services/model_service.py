"""Model Service for Weather Forecasting LSTM Application.

Orchestrates dataset operations, model training, evaluation, persistence,
and inference across the FastAPI backend.
"""

from __future__ import annotations

import asyncio
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend.app.config import (
    ARTIFACTS_DIR,
    DATA_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATASET,
    DEFAULT_EPOCHS,
    DEFAULT_HORIZON,
    DEFAULT_LOOKBACK,
    MODEL_DIR,
)
from backend.ml.data_loader import (
    detect_temperature_column,
    detect_timestamp_column,
    get_dataset_info as ml_get_dataset_info,
    load_csv,
    validate_dataset,
)
from backend.ml.evaluate import ModelEvaluator
from backend.ml.model import build_lstm_model
from backend.ml.predict import WeatherPredictor
from backend.ml.preprocessing import WeatherPreprocessor
from backend.ml.train import ModelTrainer


class ModelService:
    """Singleton service managing dataset, training, evaluation, and inference."""

    _instance: Optional[ModelService] = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.dataset_df: Optional[pd.DataFrame] = None
        self.dataset_path: Optional[str] = None
        self.timestamp_col: str = "timestamp"
        self.temperature_col: str = "temperature_celsius"
        self.location: str = "New Delhi, India"

        # ML Components
        self.model = None
        self.preprocessor: Optional[WeatherPreprocessor] = None
        self.trainer: Optional[ModelTrainer] = None
        self.evaluator: Optional[ModelEvaluator] = None
        self.predictor: Optional[WeatherPredictor] = None

        # State & Telemetry
        self.training_status: Dict[str, Any] = {
            "status": "idle",
            "progress": 0.0,
            "current_epoch": 0,
            "total_epochs": 0,
            "start_time": None,
            "end_time": None,
            "message": "Ready to train",
        }
        self.metrics: Optional[Dict[str, Any]] = None
        self.training_history: Optional[Dict[str, List[float]]] = None
        self.last_trained_at: Optional[str] = None
        self._training_thread: Optional[threading.Thread] = None

        # Paths
        self.saved_model_path = os.path.join(str(MODEL_DIR), "best_lstm_model.keras")
        self.saved_scaler_path = os.path.join(str(ARTIFACTS_DIR), "preprocessor.joblib")
        self.metrics_path = os.path.join(str(ARTIFACTS_DIR), "metrics.json")

    @classmethod
    def get_instance(cls) -> ModelService:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def auto_detect_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        ts_col = detect_timestamp_column(df)
        temp_col = detect_temperature_column(df)
        return ts_col, temp_col

    def load_dataset(
        self,
        filepath: str,
        timestamp_col: Optional[str] = None,
        temperature_col: Optional[str] = None,
        location: str = "New Delhi, India",
    ) -> Dict[str, Any]:
        """Load and validate dataset from CSV path."""
        df = load_csv(filepath)
        detected_ts, detected_temp = self.auto_detect_columns(df)

        ts_to_use = timestamp_col or detected_ts or "timestamp"
        temp_to_use = temperature_col or detected_temp or "temperature_celsius"

        validation = validate_dataset(df, timestamp_col=ts_to_use, temperature_col=temp_to_use)

        self.dataset_df = df
        self.dataset_path = filepath
        self.timestamp_col = ts_to_use
        self.temperature_col = temp_to_use
        self.location = location

        return {
            "records": len(df),
            "columns": list(df.columns),
            "timestamp_col": self.timestamp_col,
            "temperature_col": self.temperature_col,
            "validation": validation,
        }

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return metadata, summary statistics, and time range."""
        if self.dataset_df is None:
            if os.path.exists(str(DEFAULT_DATASET)):
                self.load_dataset(str(DEFAULT_DATASET))
            else:
                return {
                    "record_count": 0,
                    "columns": [],
                    "start_date": None,
                    "end_date": None,
                    "location": self.location,
                    "frequency": None,
                    "missing_count": 0,
                    "temperature_stats": {"min": 0.0, "max": 0.0, "mean": 0.0},
                }

        df = self.dataset_df
        ts_col = self.timestamp_col
        temp_col = self.temperature_col

        temp_series = df[temp_col].dropna() if temp_col in df.columns else pd.Series(dtype=float)
        min_temp = float(temp_series.min()) if not temp_series.empty else 0.0
        max_temp = float(temp_series.max()) if not temp_series.empty else 0.0
        mean_temp = float(temp_series.mean()) if not temp_series.empty else 0.0

        start_date = None
        end_date = None
        freq = "1H"
        missing_count = int(df[temp_col].isna().sum()) if temp_col in df.columns else 0

        if ts_col in df.columns:
            try:
                parsed_ts = pd.to_datetime(df[ts_col], errors="coerce").dropna().sort_values()
                if not parsed_ts.empty:
                    start_date = parsed_ts.iloc[0].isoformat()
                    end_date = parsed_ts.iloc[-1].isoformat()
            except Exception:
                pass

        return {
            "record_count": len(df),
            "columns": list(df.columns),
            "start_date": start_date,
            "end_date": end_date,
            "location": self.location,
            "frequency": freq,
            "missing_count": missing_count,
            "temperature_stats": {
                "min": round(min_temp, 2),
                "max": round(max_temp, 2),
                "mean": round(mean_temp, 2),
            },
        }

    def get_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """Fetch historical records for plotting and inspection."""
        if self.dataset_df is None:
            if os.path.exists(str(DEFAULT_DATASET)):
                self.load_dataset(str(DEFAULT_DATASET))
            else:
                return {"data": [], "total_records": 0, "location": self.location}

        df = self.dataset_df.copy()
        ts_col = self.timestamp_col
        temp_col = self.temperature_col

        if ts_col not in df.columns or temp_col not in df.columns:
            return {"data": [], "total_records": 0, "location": self.location}

        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.dropna(subset=[ts_col, temp_col]).sort_values(by=ts_col)

        if start_date:
            try:
                s_dt = pd.to_datetime(start_date)
                df = df[df[ts_col] >= s_dt]
            except Exception:
                pass

        if end_date:
            try:
                e_dt = pd.to_datetime(end_date)
                df = df[df[ts_col] <= e_dt]
            except Exception:
                pass

        total_matching = len(df)
        # Select records uniformly or take the most recent slice
        if total_matching > limit:
            df = df.iloc[-limit:]

        history_items = [
            {
                "timestamp": row[ts_col].isoformat() if hasattr(row[ts_col], "isoformat") else str(row[ts_col]),
                "temperature": round(float(row[temp_col]), 2),
            }
            for _, row in df.iterrows()
        ]

        return {
            "data": history_items,
            "total_records": total_matching,
            "location": self.location,
        }

    def set_column_mapping(self, timestamp_column: str, temperature_column: str) -> Dict[str, Any]:
        """Configure timestamp and target columns."""
        if self.dataset_df is None:
            raise ValueError("No dataset loaded.")

        if timestamp_column not in self.dataset_df.columns:
            raise ValueError(f"Column '{timestamp_column}' does not exist in dataset.")
        if temperature_column not in self.dataset_df.columns:
            raise ValueError(f"Column '{temperature_column}' does not exist in dataset.")

        self.timestamp_col = timestamp_column
        self.temperature_col = temperature_column

        return {
            "timestamp_column": self.timestamp_col,
            "temperature_column": self.temperature_col,
            "message": "Column mapping updated successfully.",
        }

    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger background training worker."""
        if self.training_status["status"] == "training":
            return {"message": "Training is already in progress.", "status": "training"}

        if self.dataset_df is None:
            if os.path.exists(str(DEFAULT_DATASET)):
                self.load_dataset(str(DEFAULT_DATASET))
            else:
                raise ValueError("No dataset loaded. Please upload or provide a weather dataset.")

        self.training_status = {
            "status": "training",
            "progress": 0.05,
            "current_epoch": 0,
            "total_epochs": int(config.get("epochs", DEFAULT_EPOCHS)),
            "start_time": datetime.utcnow().isoformat(),
            "end_time": None,
            "message": "Initializing preprocessing and sequence generation...",
        }

        # Run training in background thread
        thread = threading.Thread(target=self._run_training_job, args=(config,), daemon=True)
        self._training_thread = thread
        thread.start()

        return {"message": "Training started successfully.", "status": "training"}

    def _run_training_job(self, config: Dict[str, Any]) -> None:
        """Synchronous background execution of data preprocessing and LSTM training."""
        try:
            lookback = int(config.get("lookback", DEFAULT_LOOKBACK))
            horizon = int(config.get("horizon", DEFAULT_HORIZON))
            epochs = int(config.get("epochs", DEFAULT_EPOCHS))
            batch_size = int(config.get("batch_size", DEFAULT_BATCH_SIZE))
            lstm_units = int(config.get("lstm_units", 64))
            dropout = float(config.get("dropout", 0.2))

            # Step 1: Preprocessing
            self.training_status["message"] = "Preprocessing historical observations..."
            preprocessor = WeatherPreprocessor(lookback=lookback, horizon=horizon)
            processed = preprocessor.fit_transform(
                df=self.dataset_df,
                timestamp_col=self.timestamp_col,
                temperature_col=self.temperature_col,
                lookback=lookback,
                horizon=horizon,
            )

            X_train, y_train = processed["X_train"], processed["y_train"]
            X_val, y_val = processed["X_val"], processed["y_val"]
            X_test, y_test = processed["X_test"], processed["y_test"]

            self.training_status["progress"] = 0.2
            self.training_status["message"] = f"Building LSTM model ({lstm_units} units, {dropout} dropout)..."

            # Step 2: Build Model
            model = build_lstm_model(
                input_shape=(lookback, 1),
                horizon=horizon,
                lstm_units=lstm_units,
                dropout=dropout,
                num_layers=2,
            )

            # Step 3: Train
            self.training_status["progress"] = 0.3
            self.training_status["message"] = f"Training LSTM model for up to {epochs} epochs..."

            trainer = ModelTrainer(model=model, preprocessor=preprocessor)
            os.makedirs(os.path.dirname(self.saved_model_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.saved_scaler_path), exist_ok=True)

            history = trainer.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                patience=10,
                checkpoint_path=self.saved_model_path,
            )

            self.training_status["progress"] = 0.8
            self.training_status["message"] = "Evaluating model on chronological test dataset..."

            # Step 4: Evaluate
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

            # Format actual vs predicted sample points for charting (up to 150 points)
            y_test_inv = preprocessor.inverse_transform(y_test)
            y_pred_scaled = model.predict(X_test, verbose=0)
            y_pred_inv = preprocessor.inverse_transform(y_pred_scaled)

            num_plot_points = min(150, len(y_test_inv))
            # Sample evenly from test set
            indices = np.linspace(0, len(y_test_inv) - 1, num_plot_points, dtype=int)
            chart_points = []
            now = datetime.utcnow()
            for idx in indices:
                chart_points.append(
                    {
                        "timestamp": f"T-{len(y_test_inv) - idx}h",
                        "actual": round(float(y_test_inv[idx][0] if y_test_inv.ndim > 1 else y_test_inv[idx]), 2),
                        "predicted": round(float(y_pred_inv[idx][0] if y_pred_inv.ndim > 1 else y_pred_inv[idx]), 2),
                    }
                )

            training_loss = [round(float(v), 6) for v in history.get("loss", [])]
            validation_loss = [round(float(v), 6) for v in history.get("val_loss", [])]

            self.metrics = {
                "mae": eval_results["mae"],
                "rmse": eval_results["rmse"],
                "r2": eval_results["r2"],
                "baseline_mae": baseline_results["mae"],
                "baseline_rmse": baseline_results["rmse"],
                "baseline_r2": baseline_results["r2"],
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "actual_vs_predicted": chart_points,
                "trained_at": datetime.utcnow().isoformat(),
            }

            # Step 5: Save State
            self.model = model
            self.preprocessor = preprocessor
            self.trainer = trainer
            self.evaluator = evaluator
            self.predictor = WeatherPredictor(model=model, preprocessor=preprocessor)
            self.training_history = history
            self.last_trained_at = self.metrics["trained_at"]

            # Save scaler and weights
            preprocessor.save(self.saved_scaler_path)
            model.save(self.saved_model_path)

            self.training_status = {
                "status": "completed",
                "progress": 1.0,
                "current_epoch": len(training_loss),
                "total_epochs": epochs,
                "start_time": self.training_status.get("start_time"),
                "end_time": datetime.utcnow().isoformat(),
                "message": f"Training completed successfully! Test MAE: {eval_results['mae']}°C, RMSE: {eval_results['rmse']}°C, R²: {eval_results['r2']}",
            }

        except Exception as e:
            self.training_status = {
                "status": "failed",
                "progress": 0.0,
                "current_epoch": self.training_status.get("current_epoch", 0),
                "total_epochs": self.training_status.get("total_epochs", 0),
                "start_time": self.training_status.get("start_time"),
                "end_time": datetime.utcnow().isoformat(),
                "message": f"Training failed: {str(e)}",
            }

    def get_training_status(self) -> Dict[str, Any]:
        """Return current status of training."""
        return self.training_status

    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Return evaluation metrics or None if not trained."""
        return self.metrics

    def predict(self, horizon: int = 1) -> Dict[str, Any]:
        """Generate forward temperature prediction using the trained LSTM."""
        if self.model is None or self.preprocessor is None:
            # Try to load from saved checkpoint
            loaded = self.load_saved_model()
            if not loaded:
                raise ValueError("Model is not trained yet. Please train the model first.")

        if self.dataset_df is None:
            if os.path.exists(str(DEFAULT_DATASET)):
                self.load_dataset(str(DEFAULT_DATASET))
            else:
                raise ValueError("Dataset not loaded to generate forecast sequence.")

        df = self.dataset_df
        temp_col = self.temperature_col
        ts_col = self.timestamp_col

        if temp_col not in df.columns:
            raise ValueError(f"Temperature column '{temp_col}' not found.")

        # Extract the last lookback values
        clean_temps = df[temp_col].dropna().to_numpy()
        lookback = self.preprocessor.lookback
        if len(clean_temps) < lookback:
            raise ValueError(
                f"Dataset has only {len(clean_temps)} valid temperature records, but lookback is {lookback}."
            )

        last_window = clean_temps[-lookback:]

        # Auto-regressive recursive multi-step forecasting
        predictor = self.predictor or WeatherPredictor(self.model, self.preprocessor)
        predictions = predictor.predict_multi_step(last_window, steps=horizon)

        # Generate forecast timestamps
        last_ts = None
        if ts_col in df.columns:
            try:
                parsed_ts = pd.to_datetime(df[ts_col], errors="coerce").dropna()
                if not parsed_ts.empty:
                    last_ts = parsed_ts.iloc[-1]
            except Exception:
                pass

        if last_ts is None:
            last_ts = pd.Timestamp.now()

        prediction_items = []
        for i, val in enumerate(predictions):
            step_ts = last_ts + pd.Timedelta(hours=i + 1)
            prediction_items.append(
                {
                    "timestamp": step_ts.isoformat(),
                    "temperature": round(float(val), 2),
                }
            )

        return {
            "predictions": prediction_items,
            "model_used": "Stacked-LSTM (2 layers, 64 units)",
            "lookback_used": lookback,
            "horizon": horizon,
        }

    def load_saved_model(self) -> bool:
        """Attempt to restore saved model weights, preprocessor, and metrics from disk."""
        try:
            if os.path.exists(self.saved_model_path) and os.path.exists(self.saved_scaler_path):
                import json
                import tensorflow as tf

                self.preprocessor = WeatherPreprocessor.load(self.saved_scaler_path)
                self.model = tf.keras.models.load_model(self.saved_model_path)
                self.predictor = WeatherPredictor(self.model, self.preprocessor)

                if os.path.exists(self.metrics_path):
                    try:
                        with open(self.metrics_path, "r") as f:
                            self.metrics = json.load(f)
                    except Exception:
                        pass
                return True
        except Exception:
            pass
        return False
