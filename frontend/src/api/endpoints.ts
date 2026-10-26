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
import {
  generateMockForecast,
  generateMockPredict,
  mockDatasetInfo,
  mockHealth,
  mockHistory,
  mockMetrics,
  mockTrainStatus,
} from './mockData';

export const getHealth = async (): Promise<HealthResponse> => {
  try {
    const res = await apiClient.get<HealthResponse>('/health');
    return res.data;
  } catch {
    return mockHealth;
  }
};

export const getDatasetInfo = async (): Promise<DatasetInfoResponse> => {
  try {
    const res = await apiClient.get<DatasetInfoResponse>('/dataset/info');
    return res.data;
  } catch {
    return mockDatasetInfo;
  }
};

export const getHistory = async (params?: HistoryParams): Promise<HistoryResponse> => {
  try {
    const res = await apiClient.get<HistoryResponse>('/history', { params });
    return res.data;
  } catch {
    return mockHistory;
  }
};

export const uploadDataset = async (file: File): Promise<UploadResponse> => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const res = await apiClient.post<UploadResponse>('/data/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  } catch {
    return {
      filename: file.name,
      records: 35046,
      columns: ['timestamp', 'temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation'],
      validation: { status: 'valid', null_counts: 0 },
      message: 'Dataset parsed successfully in GitHub Pages demo preview.',
    };
  }
};

export const setColumnMapping = async (mapping: ColumnMappingRequest): Promise<void> => {
  try {
    await apiClient.post('/data/columns', mapping);
  } catch {
    // Demo mode: accept mapping
  }
};

export const trainModel = async (config: TrainConfig): Promise<TrainStatusResponse> => {
  try {
    const res = await apiClient.post<TrainStatusResponse>('/model/train', config);
    return res.data;
  } catch {
    return {
      status: 'training',
      progress: 10,
      current_epoch: 1,
      total_epochs: config.epochs,
      start_time: new Date().toISOString(),
      end_time: null,
      message: 'Simulated training started in GitHub Pages demo mode.',
    };
  }
};

export const getTrainingStatus = async (): Promise<TrainStatusResponse> => {
  try {
    const res = await apiClient.get<TrainStatusResponse>('/model/status');
    return res.data;
  } catch {
    return mockTrainStatus;
  }
};

export const getMetrics = async (): Promise<MetricsResponse> => {
  try {
    const res = await apiClient.get<MetricsResponse>('/model/metrics');
    return res.data;
  } catch {
    return mockMetrics;
  }
};

export const predict = async (config: PredictConfig): Promise<PredictResponse> => {
  try {
    const res = await apiClient.post<PredictResponse>('/predict', config);
    return res.data;
  } catch {
    return generateMockPredict(config.horizon, config.lookback ?? 72);
  }
};

export const getForecast = async (horizon: number = 24): Promise<ForecastResponse> => {
  try {
    const res = await apiClient.get<ForecastResponse>('/forecast', {
      params: { horizon },
    });
    return res.data;
  } catch {
    return generateMockForecast(horizon);
  }
};
