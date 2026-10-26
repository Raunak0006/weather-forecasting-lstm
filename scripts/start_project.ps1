# Weather Forecasting Using LSTM - Windows PowerShell Launcher
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "   Starting Weather Forecasting Using LSTM Project         " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$ROOT_DIR = Split-Path -Parent $PSScriptRoot

# 1. Start Backend FastAPI Server
Write-Host "[1/2] Launching FastAPI Backend on http://localhost:8000..." -ForegroundColor Green
Start-Process -FilePath "python" -ArgumentList "-m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload" -WorkingDirectory $ROOT_DIR

# Brief pause for backend startup
Start-Sleep -Seconds 2

# 2. Start Frontend Vite Development Server
Write-Host "[2/2] Launching React+Vite Frontend on http://localhost:5173..." -ForegroundColor Green
$FRONTEND_DIR = Join-Path $ROOT_DIR "frontend"
Start-Process -FilePath "npm.cmd" -ArgumentList "run dev" -WorkingDirectory $FRONTEND_DIR

Write-Host "`nAll services launched successfully!" -ForegroundColor Yellow
Write-Host "• Frontend Dashboard: http://localhost:5173" -ForegroundColor White
Write-Host "• Backend API Docs:   http://localhost:8000/docs" -ForegroundColor White
Write-Host "`nPress Ctrl+C in their respective windows to stop." -ForegroundColor Gray
