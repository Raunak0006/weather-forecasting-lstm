import React, { useEffect, useState } from 'react';
import {
  Calendar,
  Filter,
  RefreshCw,
  Info,
  Thermometer,
  Layers,
  Activity,
  AlertTriangle,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Brush,
} from 'recharts';

import { getDatasetInfo, getHistory } from '../api/endpoints';
import { DatasetInfoResponse, HistoryItem } from '../api/types';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorAlert } from '../components/ErrorAlert';

export const HistoricalData: React.FC = () => {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfoResponse | null>(null);
  const [historyData, setHistoryData] = useState<HistoryItem[]>([]);
  const [startDate, setStartDate] = useState<string>('');
  const [endDate, setEndDate] = useState<string>('');
  const [limit, setLimit] = useState<number>(500);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [infoRes, histRes] = await Promise.all([
        getDatasetInfo(),
        getHistory({
          start_date: startDate || undefined,
          end_date: endDate || undefined,
          limit: limit,
        }),
      ]);

      setDatasetInfo(infoRes);
      setHistoryData(histRes.data);
    } catch (err: any) {
      setError(err.message || 'Failed to load historical data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [limit]);

  const handleApplyFilter = (e: React.FormEvent) => {
    e.preventDefault();
    loadData();
  };

  const handleResetFilter = () => {
    setStartDate('');
    setEndDate('');
    setLimit(500);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
            Historical Weather Observations
          </h2>
          <p className="text-sm text-slate-500">
            Explore and filter past meteorological observations used for training the LSTM model
          </p>
        </div>
        <button
          onClick={loadData}
          className="inline-flex items-center space-x-2 px-4 py-2 bg-white border border-slate-200 rounded-xl text-sm font-semibold text-slate-700 hover:bg-slate-50 shadow-2xs transition-colors self-start md:self-auto"
        >
          <RefreshCw className="w-4 h-4 text-slate-500" />
          <span>Refresh Data</span>
        </button>
      </div>

      {error && <ErrorAlert message={error} onRetry={loadData} />}

      {/* Dataset Metadata Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-2xs">
          <div className="flex items-center space-x-2 text-slate-400 text-xs font-semibold mb-1">
            <Thermometer className="w-3.5 h-3.5 text-blue-600" />
            <span>Temperature Min/Max</span>
          </div>
          <p className="text-lg font-bold text-slate-800">
            {datasetInfo ? `${datasetInfo.temperature_stats.min}°C / ${datasetInfo.temperature_stats.max}°C` : 'N/A'}
          </p>
          <p className="text-xs text-slate-400">Mean: {datasetInfo?.temperature_stats.mean || 0}°C</p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-2xs">
          <div className="flex items-center space-x-2 text-slate-400 text-xs font-semibold mb-1">
            <Layers className="w-3.5 h-3.5 text-blue-600" />
            <span>Total Records</span>
          </div>
          <p className="text-lg font-bold text-slate-800">
            {datasetInfo ? datasetInfo.record_count.toLocaleString() : 0}
          </p>
          <p className="text-xs text-slate-400">Sampling: {datasetInfo?.frequency || 'Hourly'}</p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-2xs">
          <div className="flex items-center space-x-2 text-slate-400 text-xs font-semibold mb-1">
            <Activity className="w-3.5 h-3.5 text-blue-600" />
            <span>Observed Range</span>
          </div>
          <p className="text-xs font-bold text-slate-800 truncate">
            {datasetInfo?.start_date ? new Date(datasetInfo.start_date).toLocaleDateString() : 'N/A'} -{' '}
            {datasetInfo?.end_date ? new Date(datasetInfo.end_date).toLocaleDateString() : 'N/A'}
          </p>
          <p className="text-xs text-slate-400">Location: {datasetInfo?.location || 'New Delhi'}</p>
        </div>

        <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-2xs">
          <div className="flex items-center space-x-2 text-slate-400 text-xs font-semibold mb-1">
            <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />
            <span>Missing Observations</span>
          </div>
          <p className="text-lg font-bold text-slate-800">
            {datasetInfo ? datasetInfo.missing_count : 0}
          </p>
          <p className="text-xs text-slate-400">Handled via spline/linear interpolation</p>
        </div>
      </div>

      {/* Filter Controls */}
      <div className="bg-white p-5 rounded-2xl shadow-sm border border-slate-200/80">
        <form onSubmit={handleApplyFilter} className="flex flex-wrap items-end gap-4">
          <div className="flex-1 min-w-[180px]">
            <label className="block text-xs font-semibold text-slate-600 mb-1">Start Date</label>
            <div className="relative">
              <input
                type="datetime-local"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-slate-50/50"
              />
            </div>
          </div>

          <div className="flex-1 min-w-[180px]">
            <label className="block text-xs font-semibold text-slate-600 mb-1">End Date</label>
            <div className="relative">
              <input
                type="datetime-local"
                value={endDate}
                onChange={(e) => setEndDate(e.target.value)}
                className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-slate-50/50"
              />
            </div>
          </div>

          <div className="w-32">
            <label className="block text-xs font-semibold text-slate-600 mb-1">Display Limit</label>
            <select
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-500 bg-slate-50/50"
            >
              <option value={100}>100 pts</option>
              <option value={250}>250 pts</option>
              <option value={500}>500 pts</option>
              <option value={1000}>1,000 pts</option>
              <option value={2500}>2,500 pts</option>
            </select>
          </div>

          <div className="flex gap-2">
            <button
              type="submit"
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold text-xs rounded-xl shadow-2xs transition-colors flex items-center space-x-1.5"
            >
              <Filter className="w-3.5 h-3.5" />
              <span>Filter</span>
            </button>
            <button
              type="button"
              onClick={handleResetFilter}
              className="px-3 py-2 bg-slate-100 hover:bg-slate-200 text-slate-600 font-semibold text-xs rounded-xl transition-colors"
            >
              Reset
            </button>
          </div>
        </form>
      </div>

      {/* Main Interactive Chart */}
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200/80">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-bold text-slate-800">
              Historical Temperature Series ({historyData.length} records shown)
            </h3>
            <p className="text-xs text-slate-400">
              Drag the bottom timeline zoom slider to pan across historical periods
            </p>
          </div>
          <span className="text-xs text-slate-500 font-medium bg-slate-100 px-2.5 py-1 rounded-lg">
            Unit: Celsius (°C)
          </span>
        </div>

        {loading ? (
          <LoadingSpinner message="Querying dataset time-slice..." />
        ) : (
          <div className="h-96 w-full">
            {historyData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={historyData} margin={{ top: 10, right: 20, left: -10, bottom: 20 }}>
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
                    formatter={(val: any) => [`${val} °C`, 'Observed Temp']}
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
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                  <Brush
                    dataKey="timestamp"
                    height={30}
                    stroke="#94a3b8"
                    tickFormatter={(val) => {
                      const d = new Date(val);
                      return `${d.getMonth() + 1}/${d.getDate()}`;
                    }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-400 text-sm">
                No observations match the selected criteria.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
