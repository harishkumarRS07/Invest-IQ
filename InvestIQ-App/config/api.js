/**
 * Centralized API configuration for mobile devices (Expo Go / tunnel).
 * Use your computer's LAN IPv4 so physical devices can reach the backend.
 */

export const API_HOST = '10.114.213.162';
export const API_PORT = 5000;
export const API_BASE_URL = 'https://curly-friends-strive.loca.lt/api/v1';

const ENV_API_BASE_URL = process.env.EXPO_PUBLIC_API_BASE_URL;

export const API_BASE_URL_CANDIDATES = [
    API_BASE_URL,
    ENV_API_BASE_URL,
    `http://${API_HOST}:${API_PORT}/api/v1`,
    `http://127.0.0.1:${API_PORT}/api/v1`,
].filter(Boolean);

export const REQUEST_TIMEOUT_MS = 60000;

export const NETWORK_ERROR_MESSAGE =
    `Network error: cannot reach backend at ${API_BASE_URL}. ` +
    'Make sure the backend is running with host 0.0.0.0, port 5000, and your phone + computer are on the same Wi-Fi. Expo tunnel only tunnels Metro, not your backend API.';
