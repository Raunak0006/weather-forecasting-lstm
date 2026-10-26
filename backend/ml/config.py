"""
Configuration constants and directory paths for the Weather Forecasting ML pipeline.
"""

import os

# Base directory paths
# ML_DIR: .../weather-forecasting-lstm/backend/ml
ML_DIR = os.path.dirname(os.path.abspath(__file__))
# BACKEND_DIR: .../weather-forecasting-lstm/backend
BACKEND_DIR = os.path.dirname(ML_DIR)
# BASE_DIR: .../weather-forecasting-lstm
BASE_DIR = os.path.dirname(BACKEND_DIR)

# Storage directories
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
SCALER_DIR = os.path.join(BASE_DIR, "scalers")

# Default file paths
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "sample_weather_data.csv")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "best_lstm_model.keras")
DEFAULT_SCALER_PATH = os.path.join(SCALER_DIR, "weather_preprocessor.joblib")

# Ensure required storage directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SCALER_DIR, exist_ok=True)

# Default column names
DEFAULT_TIMESTAMP_COL = "timestamp"
DEFAULT_TARGET_COL = "temperature_celsius"

# Model and Sequence Defaults
DEFAULT_LOOKBACK = 24       # Number of prior hourly steps used as input sequence
DEFAULT_HORIZON = 1         # Number of future steps to predict
DEFAULT_LSTM_UNITS = 64     # Neurons per LSTM layer
DEFAULT_DROPOUT = 0.2       # Dropout probability
DEFAULT_NUM_LAYERS = 2      # Number of LSTM layers
DEFAULT_LEARNING_RATE = 0.001

# Training Hyperparameters
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 10
DEFAULT_MIN_DELTA = 1e-4

# Dataset Split Ratios (Train + Val + Test = 1.0)
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15

# Global reproducibility seed
SEED = 42
