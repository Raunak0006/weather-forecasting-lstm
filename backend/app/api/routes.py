"""API Route Definitions for Weather Forecasting LSTM Service."""

from __future__ import annotations

import os
import shutil
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from backend.app.config import API_VERSION, UPLOAD_DIR
from backend.app.schemas.schemas import (
    ColumnMappingRequest,
    DatasetInfoResponse,
    ForecastResponse,
    HealthResponse,
    HistoryResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
    TrainRequest,
    TrainStatusResponse,
    UploadResponse,
)
from backend.app.services.model_service import ModelService

router = APIRouter(prefix="/api", tags=["Weather Forecasting"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Verify backend operational status and component readiness."""
    service = ModelService.get_instance()
    return HealthResponse(
        status="ok",
        model_loaded=(service.model is not None),
        dataset_loaded=(service.dataset_df is not None),
        version=API_VERSION,
    )


@router.get("/dataset/info", response_model=DatasetInfoResponse)
async def get_dataset_metadata():
    """Retrieve statistical metadata, frequency, and dimensions of current dataset."""
    service = ModelService.get_instance()
    info = service.get_dataset_info()
    return DatasetInfoResponse(**info)


@router.post("/data/upload", response_model=UploadResponse)
async def upload_dataset_file(file: UploadFile = File(...)):
    """Upload custom historical weather CSV dataset."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files (.csv) are accepted.",
        )

    os.makedirs(str(UPLOAD_DIR), exist_ok=True)
    destination_path = os.path.join(str(UPLOAD_DIR), file.filename)

    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save uploaded file: {str(e)}",
        )

    service = ModelService.get_instance()
    try:
        result = service.load_dataset(destination_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse and validate CSV: {str(e)}",
        )

    return UploadResponse(
        filename=file.filename,
        records=result["records"],
        columns=result["columns"],
        validation=result["validation"],
        message=f"Dataset uploaded successfully ({result['records']:,} rows).",
    )


@router.post("/data/columns")
async def update_column_mapping(mapping: ColumnMappingRequest):
    """Explicitly map timestamp and temperature target columns."""
    service = ModelService.get_instance()
    try:
        res = service.set_column_mapping(
            timestamp_column=mapping.timestamp_column,
            temperature_column=mapping.temperature_column,
        )
        return res
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.post("/model/train", response_model=TrainStatusResponse)
async def trigger_model_training(train_req: TrainRequest):
    """Trigger background model training job using configured hyperparameters."""
    service = ModelService.get_instance()
    try:
        service.start_training(train_req.model_dump())
        status_info = service.get_training_status()
        return TrainStatusResponse(**status_info)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/model/status", response_model=TrainStatusResponse)
async def get_training_progress():
    """Poll live progress, status string, and metrics of training process."""
    service = ModelService.get_instance()
    return TrainStatusResponse(**service.get_training_status())


@router.get("/model/metrics", response_model=MetricsResponse)
async def get_evaluation_metrics():
    """Retrieve test set evaluation metrics and persistence baseline comparisons."""
    service = ModelService.get_instance()
    metrics = service.get_metrics()
    if metrics is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model has not been trained or evaluated yet. Please train the model first.",
        )
    return MetricsResponse(**metrics)


@router.post("/predict", response_model=PredictResponse)
async def generate_prediction(req: PredictRequest):
    """Produce single or multi-step forward temperature predictions."""
    service = ModelService.get_instance()
    try:
        pred_result = service.predict(horizon=req.horizon)
        return PredictResponse(**pred_result)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@router.get("/history", response_model=HistoryResponse)
async def get_historical_observations(
    start_date: Optional[str] = Query(None, description="Optional start datetime filter"),
    end_date: Optional[str] = Query(None, description="Optional end datetime filter"),
    limit: int = Query(1000, ge=10, le=10000, description="Max observations to return"),
):
    """Retrieve historical observation time series for plotting."""
    service = ModelService.get_instance()
    history = service.get_history(start_date=start_date, end_date=end_date, limit=limit)
    return HistoryResponse(**history)


@router.get("/forecast", response_model=ForecastResponse)
async def get_forecast_results(
    horizon: int = Query(24, ge=1, le=168, description="Forecast steps ahead"),
):
    """Generate or retrieve forward forecasts."""
    service = ModelService.get_instance()
    try:
        pred_result = service.predict(horizon=horizon)
        return ForecastResponse(
            predictions=pred_result["predictions"],
            generated_at=datetime.utcnow().isoformat(),
            model_info={
                "model_used": pred_result["model_used"],
                "lookback_used": pred_result["lookback_used"],
                "horizon": pred_result["horizon"],
            },
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
