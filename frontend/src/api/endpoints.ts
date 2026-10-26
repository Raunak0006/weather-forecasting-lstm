import { apiClient } from './client';
import {
  ColumnMappingRequest,
  DatasetInfoResponse,
  ForecastResponse,
  HealthResponse,
  HistoryParams,
  HistoryResponse,
  MetricsResponse,
  PredictConfig,
  PredictResponse,
  TrainConfig,
  TrainStatusResponse,
  UploadResponse,
} from './types';

export const getHealth = async (): Promise<HealthResponse> => {
  const res = await apiClient.get<HealthResponse>('/health');
  return res.data;
};

export const getDatasetInfo = async (): Promise<DatasetInfoResponse> => {
  const res = await apiClient.get<DatasetInfoResponse>('/dataset/info');
  return res.data;
};

export const getHistory = async (params?: HistoryParams): Promise<HistoryResponse> => {
  const res = await apiClient.get<HistoryResponse>('/history', { params });
  return res.data;
};

export const uploadDataset = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await apiClient.post<UploadResponse>('/data/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return res.data;
};

export const setColumnMapping = async (mapping: ColumnMappingRequest): Promise<void> => {
  await apiClient.post('/data/columns', mapping);
};

export const trainModel = async (config: TrainConfig): Promise<TrainStatusResponse> => {
  const res = await apiClient.post<TrainStatusResponse>('/model/train', config);
  return res.data;
};

export const getTrainingStatus = async (): Promise<TrainStatusResponse> => {
  const res = await apiClient.get<TrainStatusResponse>('/model/status');
  return res.data;
};

export const getMetrics = async (): Promise<MetricsResponse> => {
  const res = await apiClient.get<MetricsResponse>('/model/metrics');
  return res.data;
};

export const predict = async (config: PredictConfig): Promise<PredictResponse> => {
  const res = await apiClient.post<PredictResponse>('/predict', config);
  return res.data;
};

export const getForecast = async (horizon: number = 24): Promise<ForecastResponse> => {
  const res = await apiClient.get<ForecastResponse>('/forecast', {
    params: { horizon },
  });
  return res.data;
};
