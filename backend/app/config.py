import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
UPLOAD_DIR = DATA_DIR / "uploads"

# Create directories
for d in [DATA_DIR, MODEL_DIR, ARTIFACTS_DIR, UPLOAD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Default dataset
DEFAULT_DATASET = DATA_DIR / "sample_weather_data.csv"

# API settings
API_TITLE = "Weather Forecasting LSTM API"
API_VERSION = "1.0.0"
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

# Model settings
DEFAULT_LOOKBACK = 24
DEFAULT_HORIZON = 1
DEFAULT_EPOCHS = 100
DEFAULT_BATCH_SIZE = 32
