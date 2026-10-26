#!/bin/bash
# Weather Forecasting Using LSTM - Unix/Linux/macOS Launcher

echo "=========================================================="
echo "   Starting Weather Forecasting Using LSTM Project         "
echo "=========================================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# 1. Start Backend FastAPI Server
echo "[1/2] Launching FastAPI Backend on http://localhost:8000..."
cd "$ROOT_DIR"
python3 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

sleep 2

# 2. Start Frontend Vite Server
echo "[2/2] Launching React+Vite Frontend on http://localhost:5173..."
cd "$ROOT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "All services launched successfully!"
echo "• Frontend Dashboard: http://localhost:5173"
echo "• Backend API Docs:   http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers."

trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait
