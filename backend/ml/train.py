"""
Training Module for Weather Forecasting LSTM Model.

Manages model training workflows, reproducible seed configuration, callbacks
(EarlyStopping, ModelCheckpoint, ReduceLROnPlateau), training telemetry,
and model persistence.
"""

from __future__ import annotations

import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

from .config import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_PATIENCE, MODEL_DIR, SEED
from .preprocessing import WeatherPreprocessor


def set_reproducible_seeds(seed: int = SEED) -> None:
    """Set global random seeds across Python, NumPy, and TensorFlow for reproducibility.

    Parameters
    ----------
    seed : int
        Seed value to initialize random number generators.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class ModelTrainer:
    """Orchestrates LSTM training, validation, callbacks, and persistence."""

    def __init__(
        self,
        model: Optional[tf.keras.Model] = None,
        preprocessor: Optional[WeatherPreprocessor] = None,
    ) -> None:
        """Initialize ModelTrainer with an optional model and preprocessor.

        Parameters
        ----------
        model : Optional[tf.keras.Model]
            Compiled Keras LSTM model.
        preprocessor : Optional[WeatherPreprocessor]
            Fitted data preprocessor instance.
        """
        self.model: Optional[tf.keras.Model] = model
        self.preprocessor: Optional[WeatherPreprocessor] = preprocessor

        self.status: str = "idle"  # 'idle', 'training', 'completed', 'failed'
        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.training_duration_seconds: Optional[float] = None
        self.epochs_completed: int = 0
        self.best_val_loss: Optional[float] = None
        self.history: Dict[str, List[float]] = {}
        self.last_error: Optional[str] = None

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        patience: int = DEFAULT_PATIENCE,
        checkpoint_path: Optional[str] = None,
        seed: int = SEED,
    ) -> Dict[str, Any]:
        """Train the LSTM model on training sequences with validation monitoring.

        Configures:
        - EarlyStopping: halts training when validation loss stops improving and restores best weights.
        - ReduceLROnPlateau: decreases learning rate when validation loss plateaus.
        - ModelCheckpoint: saves best model checkpoint to disk during training.

        Parameters
        ----------
        X_train : np.ndarray
            Training sequence inputs of shape (samples, lookback, features).
        y_train : np.ndarray
            Training targets of shape (samples, horizon).
        X_val : np.ndarray
            Validation sequence inputs.
        y_val : np.ndarray
            Validation targets.
        epochs : int
            Maximum number of training epochs.
        batch_size : int
            Mini-batch size for gradient descent.
        patience : int
            Number of epochs with no improvement after which training stops.
        checkpoint_path : Optional[str]
            Filepath where best model checkpoint is saved.
        seed : int
            Random seed for training reproducibility.

        Returns
        -------
        Dict[str, Any]
            Training history dictionary and summary statistics.

        Raises
        ------
        RuntimeError
            If no model has been provided or built before calling train.
        ValueError
            If dataset dimensions are mismatched or empty.
        """
        if self.model is None:
            raise RuntimeError("Cannot train: model is not defined. Provide or build a model first.")

        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data arrays X_train and y_train cannot be empty.")
        if len(X_val) == 0 or len(y_val) == 0:
            raise ValueError("Validation data arrays X_val and y_val cannot be empty.")

        set_reproducible_seeds(seed)

        self.status = "training"
        self.last_error = None
        t_start = time.time()
        self.start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Configure callbacks
        callbacks: List[Callback] = [
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.5,
                patience=max(2, patience // 2),
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        if checkpoint_path is None:
            checkpoint_path = os.path.join(MODEL_DIR, "best_checkpoint.keras")

        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                verbose=1,
            )
        )

        try:
            fit_history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
                shuffle=False,  # Preserve sequential mini-batch temporal locality
            )

            t_end = time.time()
            self.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.training_duration_seconds = round(t_end - t_start, 2)
            self.epochs_completed = len(fit_history.epoch)

            # Extract metrics history (convert np.float to python float)
            self.history = {
                metric: [float(v) for v in vals]
                for metric, vals in fit_history.history.items()
            }

            if "val_loss" in self.history and len(self.history["val_loss"]) > 0:
                self.best_val_loss = float(min(self.history["val_loss"]))
            else:
                self.best_val_loss = None

            self.status = "completed"

            return {
                "status": self.status,
                "epochs_completed": self.epochs_completed,
                "best_val_loss": self.best_val_loss,
                "training_duration_seconds": self.training_duration_seconds,
                "start_time": self.start_time,
                "end_time": self.end_time,
                "history": self.history,
                "checkpoint_path": checkpoint_path,
            }

        except Exception as e:
            self.status = "failed"
            self.last_error = str(e)
            self.end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            raise e

    def save_model(self, filepath: str) -> None:
        """Save the trained Keras model to disk.

        Parameters
        ----------
        filepath : str
            Destination path (.keras or .h5 format).

        Raises
        ------
        RuntimeError
            If no model exists to save.
        """
        if self.model is None:
            raise RuntimeError("No model exists to save.")

        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.model.save(filepath)

    def load_model(self, filepath: str) -> tf.keras.Model:
        """Load a saved Keras model from disk.

        Parameters
        ----------
        filepath : str
            Path to saved model file.

        Returns
        -------
        tf.keras.Model
            Loaded Keras model.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found at: {filepath}")

        self.model = tf.keras.models.load_model(filepath)
        return self.model

    def get_training_status(self) -> Dict[str, Any]:
        """Return the current training status, timing, and latest metrics.

        Returns
        -------
        Dict[str, Any]
            Status dictionary.
        """
        return {
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "training_duration_seconds": self.training_duration_seconds,
            "epochs_completed": self.epochs_completed,
            "best_val_loss": self.best_val_loss,
            "last_error": self.last_error,
            "history_available": len(self.history) > 0,
        }
