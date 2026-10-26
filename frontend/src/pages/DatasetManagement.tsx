import React, { useEffect, useState, useRef } from 'react';
import {
  UploadCloud,
  FileText,
  CheckCircle,
  AlertTriangle,
  RefreshCw,
  Database,
  ArrowRight,
  Sliders,
} from 'lucide-react';

import { getDatasetInfo, setColumnMapping, uploadDataset } from '../api/endpoints';
import { DatasetInfoResponse, UploadResponse } from '../api/types';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorAlert } from '../components/ErrorAlert';

export const DatasetManagement: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [datasetInfo, setDatasetInfo] = useState<DatasetInfoResponse | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [uploading, setUploading] = useState<boolean>(false);

  // Column mapping states
  const [timestampCol, setTimestampCol] = useState<string>('timestamp');
  const [temperatureCol, setTemperatureCol] = useState<string>('temperature_celsius');
  const [mappingMessage, setMappingMessage] = useState<string | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadInfo = async () => {
    try {
      setLoading(true);
      setError(null);
      const res = await getDatasetInfo();
      setDatasetInfo(res);
      if (res.columns.length > 0) {
        if (res.columns.includes('timestamp')) setTimestampCol('timestamp');
        if (res.columns.includes('temperature_celsius')) setTemperatureCol('temperature_celsius');
      }
    } catch (err: any) {
      setError(err.message || 'Failed to fetch dataset info.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadInfo();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    try {
      setUploading(true);
      setError(null);
      const res = await uploadDataset(selectedFile);
      setUploadResult(res);
      setSelectedFile(null);
      await loadInfo();
    } catch (err: any) {
      setError(err.message || 'Failed to upload dataset.');
    } finally {
      setUploading(false);
    }
  };

  const handleSaveMapping = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      setError(null);
      await setColumnMapping({
        timestamp_column: timestampCol,
        temperature_column: temperatureCol,
      });
      setMappingMessage('Column mapping successfully saved and applied to pipeline.');
      await loadInfo();
      setTimeout(() => setMappingMessage(null), 4000);
    } catch (err: any) {
      setError(err.message || 'Failed to update column mapping.');
    }
  };

  if (loading) {
    return <LoadingSpinner message="Inspecting dataset configuration..." />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-900 tracking-tight">
            Dataset Management & Telemetry
          </h2>
          <p className="text-sm text-slate-500">
            Inspect, validate, upload custom CSVs, and configure time-series feature mappings
          </p>
        </div>

        <button
          onClick={loadInfo}
          className="inline-flex items-center space-x-2 px-4 py-2 bg-white border border-slate-200 rounded-xl text-sm font-semibold text-slate-700 hover:bg-slate-50 shadow-2xs transition-colors self-start md:self-auto"
        >
          <RefreshCw className="w-4 h-4 text-slate-500" />
          <span>Reload Details</span>
        </button>
      </div>

      {error && <ErrorAlert message={error} onRetry={loadInfo} />}

      {/* Current Dataset Overview Card */}
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200/80 p-6">
        <div className="flex items-center justify-between pb-4 border-b border-slate-100">
          <div className="flex items-center space-x-3">
            <div className="p-2.5 bg-blue-50 text-blue-600 rounded-xl border border-blue-100">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-slate-800">
                Active Weather Dataset: {datasetInfo?.location || 'New Delhi, India'}
              </h3>
              <p className="text-xs text-slate-400">
                Standardized hourly weather observations with temporal consistency validation
              </p>
            </div>
          </div>
          <span className="px-3 py-1 bg-emerald-50 text-emerald-700 text-xs font-bold rounded-full border border-emerald-200">
            Active in Memory
          </span>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-5">
          <div className="p-3 bg-slate-50 rounded-xl">
            <p className="text-xs text-slate-500">Total Valid Observations</p>
            <p className="text-lg font-extrabold text-slate-800 mt-0.5">
              {datasetInfo?.record_count.toLocaleString()}
            </p>
          </div>
          <div className="p-3 bg-slate-50 rounded-xl">
            <p className="text-xs text-slate-500">Temporal Frequency</p>
            <p className="text-lg font-extrabold text-slate-800 mt-0.5">
              {datasetInfo?.frequency || 'Hourly (1H)'}
            </p>
          </div>
          <div className="p-3 bg-slate-50 rounded-xl">
            <p className="text-xs text-slate-500">Observed Start Date</p>
            <p className="text-xs font-bold text-slate-800 mt-1 truncate">
              {datasetInfo?.start_date ? new Date(datasetInfo.start_date).toLocaleString() : 'N/A'}
            </p>
          </div>
          <div className="p-3 bg-slate-50 rounded-xl">
            <p className="text-xs text-slate-500">Observed End Date</p>
            <p className="text-xs font-bold text-slate-800 mt-1 truncate">
              {datasetInfo?.end_date ? new Date(datasetInfo.end_date).toLocaleString() : 'N/A'}
            </p>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-slate-100 flex flex-wrap gap-2 items-center">
          <span className="text-xs font-semibold text-slate-400">Available Features:</span>
          {datasetInfo?.columns.map((col) => (
            <span
              key={col}
              className="px-2.5 py-1 bg-slate-100 text-slate-700 rounded-md text-xs font-mono font-medium border border-slate-200"
            >
              {col}
            </span>
          ))}
        </div>
      </div>

      {/* 2-Column: Upload CSV & Column Mapping */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* CSV Upload */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200/80 p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 mb-3">
              <UploadCloud className="w-5 h-5 text-blue-600" />
              <h3 className="text-base font-bold text-slate-800">Upload Custom Weather CSV</h3>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Upload your own station data or historical meteorological CSV. The pipeline will automatically
              validate timestamps, identify missing values, and check continuity.
            </p>

            <div
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-slate-200 hover:border-blue-500 rounded-xl p-8 text-center cursor-pointer transition-colors bg-slate-50/50 hover:bg-blue-50/20"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv"
                onChange={handleFileChange}
                className="hidden"
              />
              <FileText className="w-10 h-10 text-slate-400 mx-auto mb-2" />
              <p className="text-sm font-semibold text-slate-700">
                {selectedFile ? selectedFile.name : 'Click to select CSV file'}
              </p>
              <p className="text-xs text-slate-400 mt-1">Accepts standard .csv with header rows</p>
            </div>

            {uploadResult && (
              <div className="mt-4 p-3 bg-emerald-50 border border-emerald-200 rounded-xl text-xs text-emerald-800 flex items-start space-x-2">
                <CheckCircle className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                <div>
                  <p className="font-bold">{uploadResult.message}</p>
                  <p className="text-2xs text-emerald-600 mt-0.5">
                    Columns: {uploadResult.columns.join(', ')}
                  </p>
                </div>
              </div>
            )}
          </div>

          <div className="mt-5">
            <button
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
              className={`w-full py-2.5 rounded-xl font-semibold text-xs shadow-2xs transition-colors flex items-center justify-center space-x-2 ${
                !selectedFile || uploading
                  ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {uploading ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Parsing & Validating...</span>
                </>
              ) : (
                <>
                  <UploadCloud className="w-3.5 h-3.5" />
                  <span>Upload & Ingest Dataset</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Column Mapping Configuration */}
        <div className="bg-white rounded-2xl shadow-sm border border-slate-200/80 p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center space-x-2 mb-3">
              <Sliders className="w-5 h-5 text-blue-600" />
              <h3 className="text-base font-bold text-slate-800">Target & Time Column Mapping</h3>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Designate which column represents the observation timestamps and which represents the
              temperature target variable to predict.
            </p>

            <form onSubmit={handleSaveMapping} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Timestamp Column
                </label>
                <select
                  value={timestampCol}
                  onChange={(e) => setTimestampCol(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs text-slate-700 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {datasetInfo?.columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1">
                  Temperature Target Column (°C)
                </label>
                <select
                  value={temperatureCol}
                  onChange={(e) => setTemperatureCol(e.target.value)}
                  className="w-full px-3 py-2 border border-slate-200 rounded-xl text-xs text-slate-700 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  {datasetInfo?.columns.map((col) => (
                    <option key={col} value={col}>
                      {col}
                    </option>
                  ))}
                </select>
              </div>

              {mappingMessage && (
                <div className="p-3 bg-blue-50 border border-blue-200 text-blue-800 rounded-xl text-xs flex items-center space-x-2">
                  <CheckCircle className="w-4 h-4 text-blue-600 shrink-0" />
                  <span>{mappingMessage}</span>
                </div>
              )}

              <button
                type="submit"
                className="w-full py-2.5 bg-slate-800 hover:bg-slate-900 text-white rounded-xl font-semibold text-xs shadow-2xs transition-colors"
              >
                Save Column Mapping
              </button>
            </form>
          </div>

          <div className="mt-5 p-3 bg-amber-50 rounded-xl border border-amber-200 text-2xs text-amber-800">
            <span className="font-bold">Leakage Protection:</span> When training begins, MinMaxScaler is
            fit exclusively on the training split ({'>'}70% chronologically) to prevent look-ahead bias.
          </div>
        </div>
      </div>
    </div>
  );
};
