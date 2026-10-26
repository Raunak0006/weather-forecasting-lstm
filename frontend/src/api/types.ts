export interface TemperatureStats {
  min: number;
  max: number;
  mean: number;
}

export interface DatasetInfoResponse {
  record_count: number;
  columns: string[];
  start_date: string | null;
  end_date: string | null;
  location: string;
  frequency: string | null;
  missing_count: number;
  temperature_stats: TemperatureStats;
}

export interface TrainConfig {
  lookback: number;
  horizon: number;
  epochs: number;
  batch_size: number;
  lstm_units: number;
  dropout: number;
}

export type TrainingStatus = 'idle' | 'training' | 'completed' | 'failed';

export interface TrainStatusResponse {
  status: TrainingStatus | string;
  progress: number;
  current_epoch: number;
  total_epochs: number;
  start_time: string | null;
  end_time: string | null;
  message: string;
}

export interface ActualVsPredictedItem {
  timestamp: string;
  actual: number;
  predicted: number;
}

export interface MetricsResponse {
  mae: number;
  rmse: number;
  r2: number;
  baseline_mae: number;
  baseline_rmse: number;
  baseline_r2: number;
  training_loss: number[];
  validation_loss: number[];
  actual_vs_predicted: ActualVsPredictedItem[];
  trained_at: string;
}

export interface PredictConfig {
  lookback?: number;
  horizon: number;
}

export interface PredictionItem {
  timestamp: string;
  temperature: number;
}

export interface PredictResponse {
  predictions: PredictionItem[];
  model_used: string;
  lookback_used: number;
  horizon: number;
}

export interface ForecastResponse {
  predictions: PredictionItem[];
  generated_at: string;
  model_info: Record<string, unknown>;
}

export interface HistoryItem {
  timestamp: string;
  temperature: number;
}

export interface HistoryResponse {
  data: HistoryItem[];
  total_records: number;
  location: string;
}

export interface HistoryParams {
  start_date?: string;
  end_date?: string;
  limit?: number;
}

export interface UploadResponse {
  filename: string;
  records: number;
  columns: string[];
  validation: Record<string, unknown>;
  message: string;
}

export interface ColumnMappingRequest {
  timestamp_column: string;
  temperature_column: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  dataset_loaded: boolean;
  version: string;
}

export interface ErrorResponse {
  detail: string;
}
