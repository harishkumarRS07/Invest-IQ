/**
 * API Service - InvestIQ
 * Axios instance with JWT interceptors, error handling, and all API methods.
 */
import axios from 'axios';
import * as SecureStore from 'expo-secure-store';
import Constants from 'expo-constants';
import { API_BASE_URL, API_PORT, API_BASE_URL_CANDIDATES, REQUEST_TIMEOUT_MS, NETWORK_ERROR_MESSAGE } from '../../config/api';

// ─── Config ───────────────────────────────────────────────────────────────────
const TOKEN_KEY = 'investiq_jwt';
export const BASE_URL = API_BASE_URL;
console.log(`[InvestIQ] API base URL: ${BASE_URL}`);

let activeBaseURL = API_BASE_URL;
let baseURLProbePromise = null;

function dedupeURLs(urls) {
    return [...new Set(urls.filter(Boolean))];
}

function getBaseURLCandidates() {
    const inferred = inferLanBaseURLFromExpoHost();
    return dedupeURLs([
        ...API_BASE_URL_CANDIDATES,
        inferred,
    ]);
}

function inferLanBaseURLFromExpoHost() {
    const hostUri = Constants?.expoConfig?.hostUri || Constants?.expoGoConfig?.debuggerHost;
    if (!hostUri) {
        return null;
    }

    const host = hostUri.split(':')[0];
    if (!host || host === 'localhost' || host === '127.0.0.1') {
        return null;
    }
    return `http://${host}:${API_PORT}/api/v1`;
}

async function checkBaseURL(url) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3500);
    try {
        const res = await fetch(`${url}/health`, { 
            method: 'GET', 
            signal: controller.signal,
            headers: {
                'ngrok-skip-browser-warning': 'true',
                'User-Agent': 'InvestIQ-App/1.0.0',
            }
        });
        return res.ok;
    } catch {
        return false;
    } finally {
        clearTimeout(timeout);
    }
}

async function ensureActiveBaseURL() {
    if (baseURLProbePromise) {
        return baseURLProbePromise;
    }

    baseURLProbePromise = (async () => {
        const candidates = getBaseURLCandidates();

        for (const candidate of candidates) {
            const ok = await checkBaseURL(candidate);
            if (ok) {
                activeBaseURL = candidate;
                console.log(`[InvestIQ] Active API base URL: ${activeBaseURL}`);
                return activeBaseURL;
            }
        }

        activeBaseURL = API_BASE_URL;
        return activeBaseURL;
    })();

    try {
        return await baseURLProbePromise;
    } finally {
        baseURLProbePromise = null;
    }
}

// ─── Axios Instance ──────────────────────────────────────────────────────────
const api = axios.create({
    baseURL: BASE_URL,
    timeout: REQUEST_TIMEOUT_MS,
    headers: {
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true',
        'User-Agent': 'InvestIQ-App/1.0.0',
    },
});

// ─── Request Interceptor – attach JWT ────────────────────────────────────────
api.interceptors.request.use(
    async (config) => {
        config.baseURL = await ensureActiveBaseURL();
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
    async (error) => {
        const originalRequest = error?.config || {};
        const status = error?.response?.status;

        // Retry on server/gateway errors (502, 503, 504)
        if ((status === 502 || status === 503 || status === 504) && !originalRequest._investIQRetried) {
            originalRequest._investIQRetried = true;
            const candidates = getBaseURLCandidates().filter((url) => url !== activeBaseURL);

            console.warn(`[InvestIQ] API status ${status}. Attempting failover from ${activeBaseURL}`);

            for (const candidate of candidates) {
                const ok = await checkBaseURL(candidate);
                if (ok) {
                    activeBaseURL = candidate;
                    originalRequest.baseURL = candidate;
                    console.warn(`[InvestIQ] Failover successful to ${candidate}`);
                    return api.request(originalRequest);
                }
            }
        }

        if (!error?.response) {
            return Promise.reject(new Error(NETWORK_ERROR_MESSAGE));
        }

        if (error?.code === 'ECONNABORTED') {
            return Promise.reject(new Error('Request timed out. Please check your network and try again.'));
        }

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
        api.post('/signals/batch', { symbols }).then((r) => {
            console.log('[API] Full batch signals response:', r);
            console.log('[API] Response data:', r.data);
            const signals = r.data.signals || r.data;
            console.log('[API] Extracted signals:', signals);
            return signals;
        }),

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

export const diagnosticsApi = {
    activeBaseURL: () => activeBaseURL,
    checkConnectivity: async () => {
        const url = await ensureActiveBaseURL();
        const res = await api.get('/health', { baseURL: url });
        return { url, data: res.data };
    },
};

// ─── News ─────────────────────────────────────────────────────────────────────
export const newsApi = {
    getNews: (ticker) => api.get(`/news?ticker=${ticker}`).then((r) => r.data),
};

export default api;
