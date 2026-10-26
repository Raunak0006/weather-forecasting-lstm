"""FastAPI Main Application for Weather Forecasting LSTM."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.config import (
    API_TITLE,
    API_VERSION,
    CORS_ORIGINS,
    DEFAULT_DATASET,
)
from backend.app.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifespan context."""
    # Startup: Load default sample dataset and existing trained model if available
    service = ModelService.get_instance()
    default_csv = str(DEFAULT_DATASET)
    if os.path.exists(default_csv):
        try:
            service.load_dataset(default_csv)
            print(f"[Startup] Loaded default dataset from: {default_csv}")
        except Exception as e:
            print(f"[Startup] Warning: Could not auto-load default dataset: {e}")

    # Attempt to restore trained model if saved on disk
    try:
        if service.load_saved_model():
            print("[Startup] Loaded existing trained LSTM model from disk.")
    except Exception as e:
        print(f"[Startup] Info: No pre-trained model loaded ({e})")

    yield

    # Shutdown logic (if any)
    print("[Shutdown] Weather Forecasting API cleanly stopped.")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application instance."""
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description="Production API for Weather Temperature Forecasting using Stacked LSTM Deep Learning.",
        lifespan=lifespan,
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all for local development flexibility
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include all API endpoints
    app.include_router(router)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
