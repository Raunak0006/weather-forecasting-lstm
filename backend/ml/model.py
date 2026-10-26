"""
LSTM Model Architecture Module for Weather Forecasting.

Constructs stacked Long Short-Term Memory (LSTM) recurrent neural network architectures
with configurable depth, units, dropout regularization, and forecasting horizons.
"""

from __future__ import annotations

import io
from typing import Optional, Sequence, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


def build_lstm_model(
    input_shape: Tuple[int, int],
    horizon: int = 1,
    lstm_units: int = 64,
    dropout: float = 0.2,
    num_layers: int = 2,
    learning_rate: float = 0.001,
) -> tf.keras.Model:
    """Construct and compile a stacked LSTM neural network for time-series forecasting.

    Architecture pattern:
    - Input: shape (lookback, features)
    - If num_layers == 1:
        LSTM(lstm_units, return_sequences=False) -> Dropout(dropout) -> Dense(horizon)
    - If num_layers >= 2:
        Layer 1..(num_layers - 1): LSTM(lstm_units, return_sequences=True) -> Dropout(dropout)
        Final LSTM layer: LSTM(lstm_units, return_sequences=False) -> Dropout(dropout)
        Output Layer: Dense(horizon)

    Parameters
    ----------
    input_shape : Tuple[int, int]
        Shape of a single input sequence: (lookback_steps, num_features).
    horizon : int
        Number of output time steps to forecast. Default is 1.
    lstm_units : int
        Number of hidden memory cells per LSTM layer. Default is 64.
    dropout : float
        Dropout rate between 0.0 and 1.0 for regularization. Default is 0.2.
    num_layers : int
        Number of stacked LSTM layers. Default is 2.
    learning_rate : float
        Learning rate for the Adam optimizer. Default is 0.001.

    Returns
    -------
    tf.keras.Model
        Compiled Keras LSTM Model.

    Raises
    ------
    ValueError
        If parameters are invalid (e.g., num_layers < 1, horizon < 1).
    """
    if len(input_shape) != 2:
        raise ValueError(
            f"input_shape must be a 2-tuple (lookback, features), got {input_shape}"
        )
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if lstm_units < 1:
        raise ValueError(f"lstm_units must be >= 1, got {lstm_units}")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")
    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")

    inputs = layers.Input(shape=input_shape, name="sequence_input")
    x = inputs

    # Stacked LSTM layers
    for layer_idx in range(num_layers):
        is_last_lstm = (layer_idx == num_layers - 1)
        return_seq = not is_last_lstm

        x = layers.LSTM(
            units=lstm_units,
            return_sequences=return_seq,
            name=f"lstm_layer_{layer_idx + 1}",
        )(x)

        if dropout > 0.0:
            x = layers.Dropout(
                rate=dropout,
                name=f"dropout_{layer_idx + 1}",
            )(x)

    # Output forecasting dense layer
    outputs = layers.Dense(
        units=horizon,
        activation="linear",
        name="forecast_output",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Weather_LSTM_Forecaster")

    # Compile with Adam optimizer, MSE loss, and MAE metric
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=["mae", "mse"],
    )

    return model


def build_model_summary(model: tf.keras.Model) -> str:
    """Generate a clean string representation of the Keras model architecture summary.

    Parameters
    ----------
    model : tf.keras.Model
        Compiled or uncompiled Keras model.

    Returns
    -------
    str
        Formatted architecture summary including layer shapes and parameter counts.
    """
    lines = []
    model.summary(print_fn=lambda line: lines.append(line))
    return "\n".join(lines)
