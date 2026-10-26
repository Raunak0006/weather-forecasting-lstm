import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Thermometer,
  CloudRain,
  Database,
  Cpu,
  ArrowRight,
  TrendingUp,
  MapPin,
  Clock,
  Sparkles,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

import { getDatasetInfo, getForecast, getHistory, getTrainingStatus } from '../api/endpoints';
import { DatasetInfoResponse, ForecastResponse, HistoryItem, TrainStatusResponse } from '../api/types';
import { StatCard } from '../components/StatCard';
import { StatusBadge } from '../components/StatusBadge';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorAlert } from '../components/ErrorAlert';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  const [datasetInfo, setDatasetInfo] = useState<DatasetInfoResponse | null>(null);
  const [historyData, setHistoryData] = useState<HistoryItem[]>([]);
  const [trainingStatus, setTrainingStatus] = useState<TrainStatusResponse | null>(null);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [horizon, setHorizon] = useState<number>(24);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [infoRes, statusRes, histRes] = await Promise.all([
        getDatasetInfo(),
        getTrainingStatus(),
        getHistory({ limit: 48 }),
      ]);

      setDatasetInfo(infoRes);
      setTrainingStatus(statusRes);
      setHistoryData(histRes.data);

      // Attempt to fetch forecast if model is trained
      try {
        const forecastRes = await getForecast(horizon);
        setForecast(forecastRes);
      } catch {
        // Model might not be trained yet
        setForecast(null);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to load dashboard data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [horizon]);

  if (loading) {
    return <LoadingSpinner message="Loading Weather Intelligence Dashboard..." />;
  }

  const latestHistory = historyData.length > 0 ? historyData[historyData.length - 1] : null;
  const nextForecast = forecast && forecast.predictions.length > 0 ? forecast.predictions[0] : null;

  return (
    <div className="space-y-6">
      {/* Top Banner / Location Bar */}
      <div className="bg-gradient-to-r from-blue-700 via-blue-600 to-sky-600 rounded-2xl p-6 text-white shadow-md relative overflow-hidden">
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <div className="flex items-center space-x-2 text-blue-100 text-xs font-semibold uppercase tracking-wider mb-1">
              <MapPin className="w-3.5 h-3.5" />
              <span>{datasetInfo?.location || 'New Delhi Weather Station'}</span>
              <span>•</span>
              <Clock className="w-3.5 h-3.5" />
              <span>Resolution: {datasetInfo?.frequency || '1-Hour Hourly'}</span>
            </div>
            <h2 className="text-2xl font-extrabold tracking-tight">
              LSTM Deep Learning Temperature Forecasting
            </h2>
            <p className="text-blue-100 text-sm mt-1 max-w-2xl">
              Trained on chronological multi-year weather observations to predict future temperature
              dynamics using recurrent memory gates.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/training')}
              className="px-4 py-2 bg-white text-blue-700 font-semibold text-sm rounded-xl hover:bg-blue-50 transition-colors shadow-sm flex items-center space-x-2"
            >
              <Cpu className="w-4 h-4" />
              <span>Train Model</span>
            </button>
            <button
              onClick={() => navigate('/forecast')}
              className="px-4 py-2 bg-blue-800/60 hover:bg-blue-800 text-white font-semibold text-sm rounded-xl transition-colors border border-blue-400/30 flex items-center space-x-2"
            >
              <CloudRain className="w-4 h-4" />
              <span>View Forecast</span>
            </button>
          </div>
        </div>
      </div>

      {error && <ErrorAlert message={error} onRetry={fetchData} />}

      {/* KPI Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
        <StatCard
          title="Latest Observation"
          value={latestHistory ? `${latestHistory.temperature}°C` : 'N/A'}
          subtitle={
            latestHistory
              ? new Date(latestHistory.timestamp).toLocaleString(undefined, {
                  dateStyle: 'short',
                  timeStyle: 'short',
                })
              : 'No records'
          }
          icon={<Thermometer className="w-6 h-6" />}
        />

        <StatCard
          title={`Forecast (+1h)`}
          value={nextForecast ? `${nextForecast.temperature}°C` : 'Model Untrained'}
          subtitle={
            nextForecast
              ? `Target: ${new Date(nextForecast.timestamp).toLocaleTimeString([], {
                  hour: '2-digit',
                  minute: '2-digit',
                })}`
              : 'Train model to activate'
          }
          icon={<TrendingUp className="w-6 h-6" />}
        />

        <StatCard
          title="Total Records"
          value={datasetInfo ? datasetInfo.record_count.toLocaleString() : 0}
          subtitle={`Min: ${datasetInfo?.temperature_stats.min}°C | Max: ${datasetInfo?.temperature_stats.max}°C`}
          icon={<Database className="w-6 h-6" />}
        />

        <div className="bg-white rounded-xl shadow-md border border-slate-100 p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-slate-500">Model State</p>
            <StatusBadge status={trainingStatus?.status || 'idle'} />
          </div>
          <div className="mt-3">
            <p className="text-xs text-slate-500 line-clamp-2">
              {trainingStatus?.message || 'Ready for training jobs.'}
            </p>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between">
            <span className="text-xs text-slate-400">Architecture</span>
            <span className="text-xs font-semibold text-slate-700">Stacked LSTM (2L-64U)</span>
          </div>
        </div>
      </div>

      {/* Main Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent 48h Temperature Trend */}
        <div className="lg:col-span-2 bg-white p-6 rounded-2xl shadow-sm border border-slate-200/80">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 gap-2">
            <div>
              <h3 className="text-base font-bold text-slate-800">
                Recent Observed Temperature Trend (Last 48 Hours)
              </h3>
              <p className="text-xs text-slate-400">
                Hourly telemetry sequence feeding into the LSTM lookback window
              </p>
            </div>
            <span className="text-xs font-semibold px-2.5 py-1 bg-blue-50 text-blue-700 rounded-lg border border-blue-200">
              Historical Observations
            </span>
          </div>

          <div className="h-72 w-full">
            {historyData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(val) => {
                      const d = new Date(val);
                      return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}:00`;
                    }}
                    tick={{ fontSize: 11, fill: '#64748b' }}
                  />
                  <YAxis unit="°C" tick={{ fontSize: 11, fill: '#64748b' }} domain={['auto', 'auto']} />
                  <Tooltip
                    labelFormatter={(val: any) => (val ? new Date(String(val)).toLocaleString() : '')}
                    formatter={(val: any) => [`${val} °C`, 'Temperature']}
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      borderRadius: '8px',
                      color: '#fff',
                      fontSize: '12px',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="temperature"
                    stroke="#2563eb"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-400 text-sm">
                No historical data available. Please load a dataset.
              </div>
            )}
          </div>
        </div>

        {/* Quick Actions & Forecast Preview */}
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/80 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-bold text-slate-800">Forecast Horizon</h3>
              <div className="flex items-center space-x-1 bg-slate-100 p-1 rounded-lg">
                {[6, 12, 24].map((h) => (
                  <button
                    key={h}
                    onClick={() => setHorizon(h)}
                    className={`px-2.5 py-1 text-xs font-semibold rounded-md transition-colors ${
                      horizon === h
                        ? 'bg-blue-600 text-white shadow-2xs'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {h}h
                  </button>
                ))}
              </div>
            </div>

            {forecast ? (
              <div className="space-y-3">
                <div className="p-3 bg-blue-50 border border-blue-100 rounded-xl">
                  <div className="flex items-center justify-between text-xs text-blue-700 font-medium">
                    <span>Generated By Model:</span>
                    <span className="font-bold">{forecast.model_info.model_used as string}</span>
                  </div>
                  <div className="flex items-center justify-between text-xs text-blue-700 font-medium mt-1">
                    <span>Lookback Window:</span>
                    <span className="font-bold">{forecast.model_info.lookback_used as number} hours</span>
                  </div>
                </div>

                <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                  {forecast.predictions.slice(0, 5).map((item, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between p-2.5 rounded-lg bg-slate-50 border border-slate-100 text-xs"
                    >
                      <span className="text-slate-500 font-medium">
                        +{idx + 1}h ({new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })})
                      </span>
                      <span className="font-bold text-slate-800 text-sm">{item.temperature}°C</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : (
              <div className="p-6 bg-slate-50 rounded-xl border border-slate-200/60 text-center space-y-3 my-4">
                <Sparkles className="w-8 h-8 text-blue-500 mx-auto" />
                <h4 className="text-sm font-bold text-slate-800">Model Needs Training</h4>
                <p className="text-xs text-slate-500">
                  Train the LSTM neural network on the historical data to generate forward predictions.
                </p>
                <button
                  onClick={() => navigate('/training')}
                  className="w-full py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs rounded-lg transition-colors shadow-2xs"
                >
                  Start Training Now
                </button>
              </div>
            )}
          </div>

          <div className="pt-4 border-t border-slate-100 flex items-center justify-between">
            <button
              onClick={() => navigate('/history')}
              className="text-xs font-semibold text-blue-600 hover:text-blue-700 flex items-center space-x-1"
            >
              <span>Explore History</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => navigate('/performance')}
              className="text-xs font-semibold text-slate-600 hover:text-slate-900 flex items-center space-x-1"
            >
              <span>Model Metrics</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
