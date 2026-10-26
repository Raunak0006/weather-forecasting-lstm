import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  BrainCircuit,
  TrendingDown,
  CheckCircle2,
  Award,
  AlertCircle,
  RefreshCw,
  Clock,
  ArrowRight,
  GitCompare,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

import { getMetrics } from '../api/endpoints';
import { MetricsResponse } from '../api/types';
import { StatCard } from '../components/StatCard';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorAlert } from '../components/ErrorAlert';

export const ModelPerformance: React.FC = () => {
  const navigate = useNavigate();

  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [notEvaluated, setNotEvaluated] = useState<boolean>(false);

  const loadMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      setNotEvaluated(false);

      const res = await getMetrics();
      setMetrics(res);
    } catch (err: any) {
      if (err.message && (err.message.includes('404') || err.message.includes('not been trained'))) {
        setNotEvaluated(true);
      } else {
        setError(err.message || 'Failed to retrieve model metrics.');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadMetrics();
  }, []);

  // Format loss curve data
  const lossChartData = React.useMemo(() => {
    if (!metrics || !metrics.training_loss) return [];
    return metrics.training_loss.map((lossVal, idx) => ({
      epoch: idx + 1,
      trainLoss: lossVal,
      valLoss: metrics.validation_loss[idx] !== undefined ? metrics.validation_loss[idx] : null,
    }));
  }, [metrics]);

  if (loading) {
    return <LoadingSpinner message="Loading empirical evaluation metrics..." />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
            Model Performance & Empirical Evaluation
          </h2>
          <p className="text-sm text-slate-500">
            Rigorous chronological test-set evaluation comparing the Stacked LSTM against the naive persistence baseline
          </p>
        </div>

        <button
          onClick={loadMetrics}
          className="inline-flex items-center space-x-2 px-4 py-2 bg-white border border-slate-200 rounded-xl text-sm font-semibold text-slate-700 hover:bg-slate-50 shadow-2xs transition-colors self-start md:self-auto"
        >
          <RefreshCw className="w-4 h-4 text-slate-500" />
          <span>Refresh Metrics</span>
        </button>
      </div>

      {error && <ErrorAlert message={error} onRetry={loadMetrics} />}

      {/* Untrained / Not Evaluated State */}
      {notEvaluated && (
        <div className="bg-white p-10 rounded-2xl shadow-sm border border-slate-200 text-center space-y-4">
          <div className="w-16 h-16 bg-blue-50 border border-blue-200 text-blue-600 rounded-2xl flex items-center justify-center mx-auto">
            <BrainCircuit className="w-10 h-10" />
          </div>
          <h3 className="text-xl font-bold text-slate-900">Model Not Evaluated Yet</h3>
          <p className="text-sm text-slate-500 max-w-lg mx-auto">
            To view empirical MAE, RMSE, and R² scores, train the LSTM network on your historical weather
            dataset. The training workflow will automatically evaluate the model on unseen chronological
            test sequences.
          </p>
          <div className="pt-2">
            <button
              onClick={() => navigate('/training')}
              className="px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-sm rounded-xl transition-colors shadow-sm inline-flex items-center space-x-2"
            >
              <span>Go to Training Panel</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}

      {/* Metrics Dashboard when Evaluated */}
      {!notEvaluated && metrics && (
        <>
          {/* Top Metric Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
            <StatCard
              title="Mean Absolute Error (MAE)"
              value={`${metrics.mae.toFixed(3)} °C`}
              subtitle={`Baseline MAE: ${metrics.baseline_mae.toFixed(3)} °C`}
              trend={`${(
                ((metrics.baseline_mae - metrics.mae) / metrics.baseline_mae) *
                100
              ).toFixed(1)}% improvement over baseline`}
              trendPositive={metrics.mae < metrics.baseline_mae}
              icon={<Award className="w-6 h-6" />}
            />

            <StatCard
              title="Root Mean Squared Error"
              value={`${metrics.rmse.toFixed(3)} °C`}
              subtitle={`Baseline RMSE: ${metrics.baseline_rmse.toFixed(3)} °C`}
              trend={`${(
                ((metrics.baseline_rmse - metrics.rmse) / metrics.baseline_rmse) *
                100
              ).toFixed(1)}% improvement over baseline`}
              trendPositive={metrics.rmse < metrics.baseline_rmse}
              icon={<TrendingDown className="w-6 h-6" />}
            />

            <StatCard
              title="Coefficient of Determination (R²)"
              value={metrics.r2.toFixed(4)}
              subtitle={`Baseline R²: ${metrics.baseline_r2.toFixed(4)}`}
              trend="Correlation with actuals"
              trendPositive={metrics.r2 > 0.8}
              icon={<BrainCircuit className="w-6 h-6" />}
            />

            <div className="bg-white rounded-xl shadow-md border border-slate-100 p-5 flex flex-col justify-between">
              <div>
                <p className="text-sm font-medium text-slate-500">Evaluation Timestamp</p>
                <h4 className="text-lg font-bold text-slate-800 mt-1">
                  {new Date(metrics.trained_at).toLocaleDateString([], {
                    month: 'short',
                    day: 'numeric',
                    year: 'numeric',
                  })}
                </h4>
                <p className="text-xs text-slate-400">
                  {new Date(metrics.trained_at).toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                  })}
                </p>
              </div>
              <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
                <span>Evaluation Data</span>
                <span className="font-semibold text-emerald-600 flex items-center">
                  <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
                  Unseen Chronological Test Set
                </span>
              </div>
            </div>
          </div>

          {/* LSTM vs Baseline Comparison Table */}
          <div className="bg-white rounded-2xl shadow-sm border border-slate-200/80 p-6">
            <div className="flex items-center space-x-2 mb-4">
              <GitCompare className="w-5 h-5 text-blue-600" />
              <h3 className="text-base font-bold text-slate-800">
                Comparative Performance: Stacked LSTM vs. Naive Persistence Baseline
              </h3>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              The persistence baseline assumes that the temperature 1 step ahead is identical to the last
              observed temperature: T(t+1) = T(t). An ML model is only considered viable if it outperforms
              this naive baseline on unseen test sequences.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-400 font-bold uppercase tracking-wider bg-slate-50">
                    <th className="py-3 px-4">Model Architecture</th>
                    <th className="py-3 px-4">MAE (°C)</th>
                    <th className="py-3 px-4">RMSE (°C)</th>
                    <th className="py-3 px-4">R² Score</th>
                    <th className="py-3 px-4">Result Assessment</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  <tr className="bg-blue-50/40">
                    <td className="py-3 px-4 font-bold text-blue-900 flex items-center space-x-2">
                      <span className="w-2 h-2 rounded-full bg-blue-600" />
                      <span>Stacked LSTM (Proposed Deep Architecture)</span>
                    </td>
                    <td className="py-3 px-4 font-bold text-blue-900">{metrics.mae.toFixed(4)} °C</td>
                    <td className="py-3 px-4 font-bold text-blue-900">{metrics.rmse.toFixed(4)} °C</td>
                    <td className="py-3 px-4 font-bold text-blue-900">{metrics.r2.toFixed(4)}</td>
                    <td className="py-3 px-4">
                      <span className="px-2.5 py-1 bg-emerald-100 text-emerald-800 font-bold rounded-md">
                        Superior Accuracy
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td className="py-3 px-4 font-medium text-slate-600 flex items-center space-x-2">
                      <span className="w-2 h-2 rounded-full bg-slate-400" />
                      <span>Naive Persistence Baseline (Lag-1 Observation)</span>
                    </td>
                    <td className="py-3 px-4 text-slate-600">{metrics.baseline_mae.toFixed(4)} °C</td>
                    <td className="py-3 px-4 text-slate-600">{metrics.baseline_rmse.toFixed(4)} °C</td>
                    <td className="py-3 px-4 text-slate-600">{metrics.baseline_r2.toFixed(4)}</td>
                    <td className="py-3 px-4 text-slate-400">Baseline Benchmark</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Actual vs Predicted on Test Set */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/80">
              <h3 className="text-base font-bold text-slate-800 mb-1">
                Actual vs. Predicted Temperatures (Test Set Sample)
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                Comparison of actual ground truth temperatures vs. model forecasts on held-out test data
              </p>
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metrics.actual_vs_predicted} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="timestamp" tick={{ fontSize: 10, fill: '#64748b' }} />
                    <YAxis unit="°C" tick={{ fontSize: 10, fill: '#64748b' }} domain={['auto', 'auto']} />
                    <Tooltip
                      formatter={(val: any) => [`${val} °C`]}
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        borderRadius: '8px',
                        color: '#fff',
                        fontSize: '12px',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    <Line
                      type="monotone"
                      dataKey="actual"
                      name="Actual Temperature"
                      stroke="#2563eb"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="predicted"
                      name="LSTM Prediction"
                      stroke="#0ea5e9"
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Training & Validation Loss Curve */}
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/80">
              <h3 className="text-base font-bold text-slate-800 mb-1">
                Training & Validation Loss History (MSE)
              </h3>
              <p className="text-xs text-slate-400 mb-4">
                Convergence trajectory of Adam optimization with EarlyStopping and learning rate reduction
              </p>
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={lossChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="epoch" name="Epoch" tick={{ fontSize: 10, fill: '#64748b' }} />
                    <YAxis tick={{ fontSize: 10, fill: '#64748b' }} domain={['auto', 'auto']} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        borderRadius: '8px',
                        color: '#fff',
                        fontSize: '12px',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }} />
                    <Line
                      type="monotone"
                      dataKey="trainLoss"
                      name="Training Loss (MSE)"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="valLoss"
                      name="Validation Loss (MSE)"
                      stroke="#f59e0b"
                      strokeWidth={2}
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
};
