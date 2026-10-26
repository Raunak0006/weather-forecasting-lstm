import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Cpu,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Brain,
  Sliders,
  Layers,
  Sparkles,
  ArrowRight,
  TrendingDown,
} from 'lucide-react';

import { getMetrics, getTrainingStatus, trainModel } from '../api/endpoints';
import { MetricsResponse, TrainConfig, TrainStatusResponse } from '../api/types';
import { StatusBadge } from '../components/StatusBadge';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorAlert } from '../components/ErrorAlert';

export const Training: React.FC = () => {
  const navigate = useNavigate();

  // Hyperparameters
  const [config, setConfig] = useState<TrainConfig>({
    lookback: 24,
    horizon: 1,
    epochs: 30, // Default 30 for swift convergence and snappy interactive demonstration
    batch_size: 64,
    lstm_units: 64,
    dropout: 0.2,
  });

  const [status, setStatus] = useState<TrainStatusResponse | null>(null);
  const [completedMetrics, setCompletedMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      const st = await getTrainingStatus();
      setStatus(st);

      if (st.status === 'completed') {
        try {
          const m = await getMetrics();
          setCompletedMetrics(m);
        } catch {
          // ignore
        }
      }
    } catch (err: any) {
      setError(err.message || 'Failed to query training status.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  // Poll while training is running
  useEffect(() => {
    let interval: any;
    if (status?.status === 'training') {
      interval = setInterval(() => {
        fetchStatus();
      }, 2500);
    }
    return () => clearInterval(interval);
  }, [status?.status]);

  const handleStartTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setError(null);
      setCompletedMetrics(null);
      const res = await trainModel(config);
      setStatus(res);
    } catch (err: any) {
      setError(err.message || 'Failed to start model training.');
    }
  };

  const isTraining = status?.status === 'training';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
            LSTM Model Training & Experimentation
          </h2>
          <p className="text-sm text-slate-500">
            Configure recurrent hyperparameters, initiate training, and track validation loss convergence
          </p>
        </div>

        <div className="flex items-center space-x-2 self-start md:self-auto">
          <StatusBadge status={status?.status || 'idle'} />
        </div>
      </div>

      {error && <ErrorAlert message={error} onRetry={fetchStatus} />}

      {/* Main Grid: Config Form & Live Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Hyperparameter Configuration */}
        <div className="lg:col-span-2 bg-white rounded-2xl shadow-sm border border-slate-200/80 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Sliders className="w-5 h-5 text-blue-600" />
            <h3 className="text-base font-bold text-slate-800">Model & Sequence Hyperparameters</h3>
          </div>

          <form onSubmit={handleStartTraining} className="space-y-5">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Lookback Window Length (Hours)
                </label>
                <input
                  type="number"
                  min={6}
                  max={168}
                  value={config.lookback}
                  onChange={(e) => setConfig({ ...config, lookback: Number(e.target.value) })}
                  disabled={isTraining}
                  className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs bg-slate-50 focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono disabled:opacity-50"
                />
                <p className="text-2xs text-slate-400 mt-1">
                  Number of prior observations fed as input sequence (default 24h)
                </p>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Target Forecast Horizon (Steps Ahead)
                </label>
                <input
                  type="number"
                  min={1}
                  max={48}
                  value={config.horizon}
                  onChange={(e) => setConfig({ ...config, horizon: Number(e.target.value) })}
                  disabled={isTraining}
                  className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs bg-slate-50 focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono disabled:opacity-50"
                />
                <p className="text-2xs text-slate-400 mt-1">
                  Direct output dense dimensions (default 1 step = +1 hour)
                </p>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  LSTM Hidden Layer Units
                </label>
                <select
                  value={config.lstm_units}
                  onChange={(e) => setConfig({ ...config, lstm_units: Number(e.target.value) })}
                  disabled={isTraining}
                  className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs bg-slate-50 focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono disabled:opacity-50"
                >
                  <option value={32}>32 Hidden Units</option>
                  <option value={64}>64 Hidden Units (Recommended)</option>
                  <option value={128}>128 Hidden Units</option>
                </select>
                <p className="text-2xs text-slate-400 mt-1">Cell memory capacity in stacked layers</p>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Dropout Regularization Rate
                </label>
                <select
                  value={config.dropout}
                  onChange={(e) => setConfig({ ...config, dropout: Number(e.target.value) })}
                  disabled={isTraining}
                  className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs bg-slate-50 focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono disabled:opacity-50"
                >
                  <option value={0.0}>0.0 (No Dropout)</option>
                  <option value={0.1}>0.1 (10% Dropout)</option>
                  <option value={0.2}>0.2 (20% Dropout - Standard)</option>
                  <option value={0.3}>0.3 (30% Dropout)</option>
                </select>
                <p className="text-2xs text-slate-400 mt-1">Prevents recurrent overfitting</p>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Maximum Epochs
                </label>
                <input
                  type="number"
                  min={5}
                  max={200}
                  value={config.epochs}
                  onChange={(e) => setConfig({ ...config, epochs: Number(e.target.value) })}
                  disabled={isTraining}
                  className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs bg-slate-50 focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono disabled:opacity-50"
                />
                <p className="text-2xs text-slate-400 mt-1">
                  EarlyStopping halts automatically if val_loss plateaus
                </p>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Mini-Batch Size
                </label>
                <select
                  value={config.batch_size}
                  onChange={(e) => setConfig({ ...config, batch_size: Number(e.target.value) })}
                  disabled={isTraining}
                  className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs bg-slate-50 focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono disabled:opacity-50"
                >
                  <option value={32}>32 samples</option>
                  <option value={64}>64 samples (Fast CPU)</option>
                  <option value={128}>128 samples</option>
                </select>
                <p className="text-2xs text-slate-400 mt-1">SGD optimization step size</p>
              </div>
            </div>

            <div className="pt-4 border-t border-slate-100 flex items-center justify-between">
              <span className="text-xs text-slate-400 flex items-center">
                <Brain className="w-3.5 h-3.5 mr-1 text-blue-500" />
                Loss: MSE • Optimizer: Adam (lr=0.001)
              </span>

              <button
                type="submit"
                disabled={isTraining}
                className={`px-6 py-2.5 rounded-xl font-bold text-xs shadow-sm transition-all flex items-center space-x-2 ${
                  isTraining
                    ? 'bg-amber-500 text-white cursor-wait animate-pulse'
                    : 'bg-blue-600 hover:bg-blue-700 text-white'
                }`}
              >
                {isTraining ? (
                  <>
                    <RotateCcw className="w-4 h-4 animate-spin" />
                    <span>Training Active...</span>
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 fill-current" />
                    <span>Train LSTM Model</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Right Col: Live Training Telemetry */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200/80 p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <Cpu className="w-5 h-5 text-blue-600" />
              <h3 className="text-base font-bold text-slate-800">Training Telemetry</h3>
            </div>

            {/* Progress bar */}
            <div className="space-y-2 mb-5">
              <div className="flex justify-between text-xs text-slate-600 font-semibold">
                <span>Progress</span>
                <span>{Math.round((status?.progress || 0) * 100)}%</span>
              </div>
              <div className="w-full bg-slate-100 rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                  style={{ width: `${Math.round((status?.progress || 0) * 100)}%` }}
                />
              </div>
            </div>

            {/* Status Details */}
            <div className="space-y-3 text-xs">
              <div className="p-3 bg-slate-50 rounded-xl flex items-center justify-between">
                <span className="text-slate-500">Status</span>
                <span className="font-bold text-slate-800 uppercase tracking-wide">
                  {status?.status || 'idle'}
                </span>
              </div>

              <div className="p-3 bg-slate-50 rounded-xl flex items-center justify-between">
                <span className="text-slate-500">Epochs Completed</span>
                <span className="font-bold text-slate-800">
                  {status?.current_epoch || 0} / {status?.total_epochs || 0}
                </span>
              </div>

              <div className="p-3 bg-slate-50 rounded-xl">
                <span className="text-slate-400 block text-2xs uppercase tracking-wider mb-1">
                  Log Message
                </span>
                <p className="text-slate-700 font-medium">
                  {status?.message || 'Awaiting training initiation.'}
                </p>
              </div>
            </div>
          </div>

          {/* Quick Result Summary after completion */}
          {completedMetrics && (
            <div className="mt-5 p-4 bg-emerald-50 border border-emerald-200 rounded-xl space-y-2">
              <div className="flex items-center space-x-1.5 text-emerald-800 font-bold text-xs">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>Training Successful!</span>
              </div>
              <div className="text-xs text-emerald-700 space-y-1">
                <p>Test MAE: <span className="font-bold">{completedMetrics.mae.toFixed(3)} °C</span></p>
                <p>Test RMSE: <span className="font-bold">{completedMetrics.rmse.toFixed(3)} °C</span></p>
                <p>Test R²: <span className="font-bold">{completedMetrics.r2.toFixed(4)}</span></p>
              </div>
              <button
                onClick={() => navigate('/performance')}
                className="w-full mt-2 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg text-xs font-semibold transition-colors flex items-center justify-center space-x-1"
              >
                <span>View Full Metrics</span>
                <ArrowRight className="w-3 h-3" />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
