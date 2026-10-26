"""Pydantic schemas for the Weather Forecasting LSTM API."""

from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TemperatureStats(BaseModel):
    """Statistical summary of temperature values in dataset."""
    min: float = Field(..., description="Minimum recorded temperature in Celsius")
    max: float = Field(..., description="Maximum recorded temperature in Celsius")
    mean: float = Field(..., description="Mean recorded temperature in Celsius")

    model_config = ConfigDict(extra="ignore")


class DatasetInfoResponse(BaseModel):
    """Metadata and statistics for the loaded weather dataset."""
    record_count: int = Field(..., description="Total number of valid records in dataset")
    columns: List[str] = Field(..., description="List of columns available in the dataset")
    start_date: Optional[str] = Field(None, description="Start date/time of records (ISO format)")
    end_date: Optional[str] = Field(None, description="End date/time of records (ISO format)")
    location: str = Field("Weather Station", description="Location or sensor name")
    frequency: Optional[str] = Field(None, description="Inferred time frequency of observations")
    missing_count: int = Field(0, description="Total missing or null temperature observations")
    temperature_stats: TemperatureStats = Field(..., description="Temperature summary statistics")

    model_config = ConfigDict(extra="ignore")


class TrainRequest(BaseModel):
    """Hyperparameters and configuration for model training."""
    lookback: int = Field(24, ge=1, le=720, description="Historical timesteps per sequence window")
    horizon: int = Field(1, ge=1, le=168, description="Prediction horizon in future timesteps")
    epochs: int = Field(100, ge=1, le=1000, description="Total training epochs")
    batch_size: int = Field(32, ge=1, le=512, description="Batch size for gradient descent")
    lstm_units: int = Field(64, ge=8, le=512, description="Number of units in LSTM layers")
    dropout: float = Field(0.2, ge=0.0, le=0.8, description="Dropout rate for regularization")

    @field_validator("lookback", "horizon", "epochs", "batch_size", "lstm_units")
    @classmethod
    def validate_positive_ints(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Parameter must be greater than zero.")
        return v

    @field_validator("dropout")
    @classmethod
    def validate_dropout(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError("Dropout must be between 0.0 and 1.0.")
        return v

    model_config = ConfigDict(extra="ignore")


class TrainStatusResponse(BaseModel):
    """Current execution status of model training job."""
    status: str = Field(..., description="Status string: idle, training, completed, or failed")
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Normalized training progress (0.0 to 1.0)")
    current_epoch: int = Field(0, ge=0, description="Current epoch executed")
    total_epochs: int = Field(0, ge=0, description="Total epochs requested")
    start_time: Optional[str] = Field(None, description="ISO timestamp when training began")
    end_time: Optional[str] = Field(None, description="ISO timestamp when training finished")
    message: str = Field("", description="Current status message or error details")

    model_config = ConfigDict(extra="ignore")


class ActualVsPredictedItem(BaseModel):
    """A single evaluation record comparing actual vs predicted temperature."""
    timestamp: str = Field(..., description="Timestamp of observation")
    actual: float = Field(..., description="Actual recorded temperature")
    predicted: float = Field(..., description="Model predicted temperature")

    model_config = ConfigDict(extra="ignore")


class MetricsResponse(BaseModel):
    """Evaluation metrics comparing the LSTM model to a persistence baseline."""
    mae: float = Field(..., description="Mean Absolute Error of the trained LSTM model")
    rmse: float = Field(..., description="Root Mean Squared Error of the trained LSTM model")
    r2: float = Field(..., description="Coefficient of determination (R-squared)")
    baseline_mae: float = Field(..., description="MAE of baseline persistence model")
    baseline_rmse: float = Field(..., description="RMSE of baseline persistence model")
    baseline_r2: float = Field(..., description="R-squared of baseline persistence model")
    training_loss: List[float] = Field(default_factory=list, description="Training loss per epoch")
    validation_loss: List[float] = Field(default_factory=list, description="Validation loss per epoch")
    actual_vs_predicted: List[ActualVsPredictedItem] = Field(
        default_factory=list, description="Evaluation points comparing actual vs predicted values"
    )
    trained_at: str = Field(..., description="ISO timestamp when training finished")

    model_config = ConfigDict(extra="ignore")


class PredictRequest(BaseModel):
    """Request payload for generating temperature forecasts."""
    lookback: Optional[int] = Field(None, ge=1, le=720, description="Optional lookback override")
    horizon: int = Field(1, ge=1, le=168, description="Number of future steps to forecast")

    model_config = ConfigDict(extra="ignore")


class PredictionItem(BaseModel):
    """Single predicted future time point."""
    timestamp: str = Field(..., description="Forecasted step timestamp")
    temperature: float = Field(..., description="Forecasted temperature in Celsius")

    model_config = ConfigDict(extra="ignore")


class PredictResponse(BaseModel):
    """Forecast response containing predicted steps and model details."""
    predictions: List[PredictionItem] = Field(..., description="List of forecasted predictions")
    model_used: str = Field(..., description="Model identifier used for inference")
    lookback_used: int = Field(..., description="Number of historical steps used")
    horizon: int = Field(..., description="Number of forecast steps produced")

    model_config = ConfigDict(extra="ignore")


class HistoryItem(BaseModel):
    """Single historical temperature observation."""
    timestamp: str = Field(..., description="ISO observation timestamp")
    temperature: float = Field(..., description="Observed temperature in Celsius")

    model_config = ConfigDict(extra="ignore")


class HistoryResponse(BaseModel):
    """Historical weather observation records."""
    data: List[HistoryItem] = Field(..., description="List of historical temperature observations")
    total_records: int = Field(..., description="Total records matching criteria")
    location: str = Field("Weather Station", description="Weather station location")

    model_config = ConfigDict(extra="ignore")


class UploadResponse(BaseModel):
    """Response returned upon dataset file upload."""
    filename: str = Field(..., description="Uploaded file name")
    records: int = Field(..., description="Number of records parsed")
    columns: List[str] = Field(..., description="Detected columns in uploaded file")
    validation: dict[str, Any] = Field(default_factory=dict, description="Validation details")
    message: str = Field(..., description="Human-readable upload result message")

    model_config = ConfigDict(extra="ignore")


class ColumnMappingRequest(BaseModel):
    """Request payload to map timestamp and target temperature columns."""
    timestamp_column: str = Field(..., min_length=1, description="Column name for timestamp")
    temperature_column: str = Field(..., min_length=1, description="Column name for temperature")

    @field_validator("timestamp_column", "temperature_column")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Column name cannot be blank.")
        return stripped

    model_config = ConfigDict(extra="ignore")


class HealthResponse(BaseModel):
    """Health check status for the backend service."""
    status: str = Field("ok", description="Service status indicator")
    model_loaded: bool = Field(..., description="True if a trained model is ready in memory")
    dataset_loaded: bool = Field(..., description="True if a dataset is currently loaded")
    version: str = Field(..., description="Application semantic version")

    model_config = ConfigDict(extra="ignore")


class ErrorResponse(BaseModel):
    """Standardized API error response."""
    detail: str = Field(..., description="Error message details")

    model_config = ConfigDict(extra="ignore")


class ForecastResponse(BaseModel):
    """Dedicated forecast response endpoint schema."""
    predictions: List[PredictionItem] = Field(..., description="List of forecast points")
    generated_at: str = Field(..., description="ISO timestamp of generation")
    model_info: dict[str, Any] = Field(default_factory=dict, description="Metadata of model used")

    model_config = ConfigDict(extra="ignore")
