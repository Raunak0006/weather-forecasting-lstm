import axios from 'axios';

export const apiClient = axios.create({
  baseURL: '/api',
  timeout: 45000,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const errorMsg =
      error.response?.data?.detail ||
      error.message ||
      'An unexpected network error occurred.';
    return Promise.reject(new Error(errorMsg));
  }
);
