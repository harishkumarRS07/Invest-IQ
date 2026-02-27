/**
 * API Service - InvestIQ
 * Axios instance with JWT interceptors, error handling, and all API methods.
 */
import axios from 'axios';
import * as SecureStore from 'expo-secure-store';

// ─── Config ─────────────────────────────────────────────────────────────────
// Change this to your backend server IP when running locally.
// e.g.  http://192.168.1.10:8000/api/v1  (find your IP with `ipconfig`)
export const BASE_URL = 'http://10.159.111.162:8000/api/v1'; // Wi-Fi development IP

const TOKEN_KEY = 'investiq_jwt';

// ─── Axios Instance ──────────────────────────────────────────────────────────
const api = axios.create({
    baseURL: BASE_URL,
    timeout: 60000, // 60s – ML inference for 5 stocks takes ~20-30s
    headers: {
        'Content-Type': 'application/json',
    },
});

// ─── Request Interceptor – attach JWT ────────────────────────────────────────
api.interceptors.request.use(
    async (config) => {
        const token = await SecureStore.getItemAsync(TOKEN_KEY);
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => Promise.reject(error)
);

// ─── Response Interceptor – normalize errors ─────────────────────────────────
api.interceptors.response.use(
    (response) => response,
    (error) => {
        const message =
            error?.response?.data?.detail ||
            error?.response?.data?.message ||
            error?.message ||
            'An unexpected error occurred';
        return Promise.reject(new Error(message));
    }
);

// ─── Token Helpers ────────────────────────────────────────────────────────────
export const saveToken = (token) => SecureStore.setItemAsync(TOKEN_KEY, token);
export const getToken = () => SecureStore.getItemAsync(TOKEN_KEY);
export const removeToken = () => SecureStore.deleteItemAsync(TOKEN_KEY);

// ─── Auth APIs ───────────────────────────────────────────────────────────────
export const authApi = {
    register: (email, password, name) =>
        api.post('/auth/register', { email, password, name }).then((r) => r.data),

    login: (email, password) =>
        api.post('/auth/login', { email, password }).then((r) => r.data),

    me: () => api.get('/auth/me').then((r) => r.data),
};

// ─── Stock APIs ───────────────────────────────────────────────────────────────
export const stockApi = {
    listTickers: () =>
        api.get('/tickers').then((r) => r.data.tickers),

    predict: (symbol) =>
        api.post('/predict', { symbol }).then((r) => r.data),

    batchSignals: (symbols) =>
        api.post('/signals/batch', { symbols }).then((r) => r.data.signals),

    sentiment: (symbol) =>
        api.post('/sentiment/analyze', { symbol }).then((r) => r.data),

    riskScore: (symbol) =>
        api.post('/risk/score', { symbol }).then((r) => r.data),
};

// ─── Portfolio APIs ───────────────────────────────────────────────────────────
export const portfolioApi = {
    optimize: (symbols, period = '1y') =>
        api.post('/portfolio/optimize', { symbols, period }).then((r) => r.data),
};

// ─── Health ───────────────────────────────────────────────────────────────────
export const healthApi = {
    check: () => api.get('/health').then((r) => r.data),
};

export default api;
