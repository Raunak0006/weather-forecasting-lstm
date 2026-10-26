import {
  DatasetInfoResponse,
  ForecastResponse,
  HealthResponse,
  HistoryResponse,
  MetricsResponse,
  PredictResponse,
  TrainStatusResponse,
} from './types';

export const mockHealth: HealthResponse = {
  status: 'healthy',
  model_loaded: true,
  dataset_loaded: true,
  version: '1.0.0 (GitHub Pages Demo Mode)',
};

export const mockDatasetInfo: DatasetInfoResponse = {
  record_count: 35046,
  columns: ['timestamp', 'temperature', 'humidity', 'pressure', 'wind_speed', 'precipitation'],
  start_date: '2022-01-01 00:00:00',
  end_date: '2025-12-31 05:00:00',
  location: 'Global Weather Observation Station (Synthesized/Real-World Aligned)',
  frequency: 'Hourly (1H)',
  missing_count: 0,
  temperature_stats: {
    min: 14.8,
    max: 43.8,
    mean: 28.64,
  },
};

const generateMockHistory = (count: number = 100) => {
  const items = [];
  const baseTime = new Date('2025-12-31T05:00:00Z').getTime();
  for (let i = count - 1; i >= 0; i--) {
    const t = new Date(baseTime - i * 3600 * 1000);
    const hour = t.getUTCHours();
    // Diurnal sinusoidal temperature curve + seasonal shift + noise
    const temp = 28 + 7 * Math.sin(((hour - 9) * Math.PI) / 12) + Math.sin(i * 0.1) * 1.2;
    items.push({
      timestamp: t.toISOString().replace('.000Z', 'Z'),
      temperature: parseFloat(temp.toFixed(2)),
    });
  }
  return items;
};

export const mockHistory: HistoryResponse = {
  data: generateMockHistory(100),
  total_records: 35046,
  location: 'Weather Observation Station',
};

export const generateMockForecast = (horizon: number = 24): ForecastResponse => {
  const startTime = new Date('2025-12-31T06:00:00Z').getTime();
  const predictions = [];
  for (let h = 0; h < horizon; h++) {
    const t = new Date(startTime + h * 3600 * 1000);
    const hour = t.getUTCHours();
    const temp = 27.5 + 7.5 * Math.sin(((hour - 9) * Math.PI) / 12) + Math.cos(h * 0.2) * 0.8;
    predictions.push({
      timestamp: t.toISOString().replace('.000Z', 'Z'),
      temperature: parseFloat(temp.toFixed(2)),
    });
  }
  return {
    predictions,
    generated_at: new Date().toISOString(),
    model_info: {
      architecture: 'Stacked LSTM (64 -> 32 units, Dropout 0.2)',
      framework: 'TensorFlow / Keras 3.x',
      lookback: 72,
      target: 'temperature (Celsius)',
    },
  };
};

export const generateMockPredict = (horizon: number = 24, lookback: number = 72): PredictResponse => {
  const forecast = generateMockForecast(horizon);
  return {
    predictions: forecast.predictions,
    model_used: 'Stacked LSTM (64-32 Units)',
    lookback_used: lookback,
    horizon,
  };
};

export const mockMetrics: MetricsResponse = {
  mae: 3.5424,
  rmse: 4.5334,
  r2: 0.7318,
  baseline_mae: 0.8818,
  baseline_rmse: 1.0589,
  baseline_r2: 0.9854,
  training_loss: [0.0824, 0.0461, 0.0315, 0.0242, 0.0195, 0.0163, 0.0141, 0.0125, 0.0112, 0.0102],
  validation_loss: [0.0651, 0.0388, 0.0279, 0.0216, 0.0182, 0.0154, 0.0138, 0.0124, 0.0115, 0.0108],
  actual_vs_predicted: [
    { timestamp: "T-5254h", actual: 30.94, predicted: 22.49 },
    { timestamp: "T-5253h", actual: 29.09, predicted: 21.37 },
    { timestamp: "T-5252h", actual: 28.87, predicted: 20.57 },
    { timestamp: "T-5251h", actual: 28.27, predicted: 20.31 },
    { timestamp: "T-5250h", actual: 27.02, predicted: 20.53 },
    { timestamp: "T-5249h", actual: 27.29, predicted: 21.03 },
    { timestamp: "T-5248h", actual: 28.1, predicted: 21.87 },
    { timestamp: "T-5247h", actual: 28.0, predicted: 23.03 },
    { timestamp: "T-5246h", actual: 30.15, predicted: 24.3 },
    { timestamp: "T-5245h", actual: 31.98, predicted: 25.81 },
    { timestamp: "T-5244h", actual: 33.21, predicted: 27.48 },
    { timestamp: "T-5243h", actual: 35.71, predicted: 29.11 },
    { timestamp: "T-5242h", actual: 37.29, predicted: 30.74 },
    { timestamp: "T-5241h", actual: 37.98, predicted: 32.17 },
    { timestamp: "T-5240h", actual: 40.2, predicted: 33.15 },
    { timestamp: "T-5239h", actual: 41.79, predicted: 33.82 },
    { timestamp: "T-5238h", actual: 40.7, predicted: 34.18 },
    { timestamp: "T-5237h", actual: 40.35, predicted: 33.84 },
    { timestamp: "T-5236h", actual: 39.78, predicted: 32.86 },
    { timestamp: "T-5235h", actual: 38.15, predicted: 31.4 },
    { timestamp: "T-5234h", actual: 37.09, predicted: 29.52 },
    { timestamp: "T-5233h", actual: 36.08, predicted: 27.48 },
    { timestamp: "T-5232h", actual: 34.74, predicted: 25.55 },
    { timestamp: "T-5231h", actual: 33.06, predicted: 23.87 },
    { timestamp: "T-5230h", actual: 32.32, predicted: 22.5 },
    { timestamp: "T-5229h", actual: 29.96, predicted: 21.61 },
    { timestamp: "T-5228h", actual: 30.26, predicted: 21.0 },
    { timestamp: "T-5227h", actual: 29.55, predicted: 20.94 },
    { timestamp: "T-5226h", actual: 28.19, predicted: 21.33 },
    { timestamp: "T-5225h", actual: 28.07, predicted: 21.91 },
    { timestamp: "T-5224h", actual: 29.01, predicted: 22.68 },
    { timestamp: "T-5223h", actual: 30.12, predicted: 23.73 },
    { timestamp: "T-5222h", actual: 31.27, predicted: 25.03 },
    { timestamp: "T-5221h", actual: 32.41, predicted: 26.49 },
    { timestamp: "T-5220h", actual: 33.89, predicted: 27.98 },
    { timestamp: "T-5219h", actual: 36.0, predicted: 29.41 },
    { timestamp: "T-5218h", actual: 37.37, predicted: 30.81 },
    { timestamp: "T-5217h", actual: 38.93, predicted: 32.01 },
    { timestamp: "T-5216h", actual: 40.49, predicted: 32.95 },
    { timestamp: "T-5215h", actual: 41.17, predicted: 33.62 },
    { timestamp: "T-5214h", actual: 40.1, predicted: 33.88 },
    { timestamp: "T-5213h", actual: 39.48, predicted: 33.44 },
    { timestamp: "T-5212h", actual: 37.57, predicted: 32.36 },
    { timestamp: "T-5211h", actual: 37.29, predicted: 30.64 },
    { timestamp: "T-5210h", actual: 35.24, predicted: 28.7 },
    { timestamp: "T-5209h", actual: 32.11, predicted: 26.61 },
    { timestamp: "T-5208h", actual: 30.31, predicted: 24.36 },
    { timestamp: "T-5207h", actual: 28.51, predicted: 22.27 },
    { timestamp: "T-5206h", actual: 27.3, predicted: 20.54 },
    { timestamp: "T-5205h", actual: 26.44, predicted: 19.35 },
    { timestamp: "T-5204h", actual: 25.54, predicted: 18.78 },
    { timestamp: "T-5203h", actual: 25.66, predicted: 18.81 },
    { timestamp: "T-5202h", actual: 27.21, predicted: 19.44 },
    { timestamp: "T-5201h", actual: 28.34, predicted: 20.74 },
    { timestamp: "T-5200h", actual: 29.97, predicted: 22.56 },
    { timestamp: "T-5199h", actual: 31.73, predicted: 24.74 },
    { timestamp: "T-5198h", actual: 33.65, predicted: 27.08 },
    { timestamp: "T-5197h", actual: 35.57, predicted: 29.38 },
    { timestamp: "T-5196h", actual: 38.24, predicted: 31.44 },
    { timestamp: "T-5195h", actual: 39.79, predicted: 33.26 },
    { timestamp: "T-5194h", actual: 40.18, predicted: 34.6 },
    { timestamp: "T-5193h", actual: 41.25, predicted: 35.21 },
    { timestamp: "T-5192h", actual: 41.25, predicted: 35.15 },
    { timestamp: "T-5191h", actual: 40.94, predicted: 34.41 },
    { timestamp: "T-5190h", actual: 40.52, predicted: 33.1 },
    { timestamp: "T-5189h", actual: 40.03, predicted: 31.4 },
    { timestamp: "T-5188h", actual: 37.88, predicted: 29.53 },
    { timestamp: "T-5187h", actual: 36.02, predicted: 27.48 },
    { timestamp: "T-5186h", actual: 34.85, predicted: 25.41 },
    { timestamp: "T-5185h", actual: 32.6, predicted: 23.6 },
    { timestamp: "T-5184h", actual: 31.26, predicted: 22.05 },
    { timestamp: "T-5183h", actual: 28.74, predicted: 20.92 },
    { timestamp: "T-5182h", actual: 28.43, predicted: 20.08 },
    { timestamp: "T-5181h", actual: 28.2, predicted: 19.78 },
    { timestamp: "T-5180h", actual: 28.98, predicted: 20.04 },
    { timestamp: "T-5179h", actual: 29.23, predicted: 20.94 },
    { timestamp: "T-5178h", actual: 29.76, predicted: 22.29 },
    { timestamp: "T-5177h", actual: 29.37, predicted: 23.94 },
    { timestamp: "T-5176h", actual: 31.19, predicted: 25.53 },
    { timestamp: "T-5175h", actual: 31.9, predicted: 27.16 },
    { timestamp: "T-5174h", actual: 34.21, predicted: 28.64 },
    { timestamp: "T-5173h", actual: 35.66, predicted: 30.07 },
    { timestamp: "T-5172h", actual: 37.18, predicted: 31.32 },
    { timestamp: "T-5171h", actual: 39.02, predicted: 32.35 },
    { timestamp: "T-5170h", actual: 39.7, predicted: 33.16 },
    { timestamp: "T-5169h", actual: 40.75, predicted: 33.56 },
    { timestamp: "T-5168h", actual: 41.19, predicted: 33.6 },
    { timestamp: "T-5167h", actual: 41.18, predicted: 33.23 },
    { timestamp: "T-5166h", actual: 40.6, predicted: 32.48 },
    { timestamp: "T-5165h", actual: 39.56, predicted: 31.36 }
  ],
  trained_at: '2026-10-26T17:30:00Z',
};

export const mockTrainStatus: TrainStatusResponse = {
  status: 'completed',
  progress: 100,
  current_epoch: 15,
  total_epochs: 15,
  start_time: '2026-10-26T17:20:00Z',
  end_time: '2026-10-26T17:30:00Z',
  message: 'Model is trained and ready for inference.',
};
