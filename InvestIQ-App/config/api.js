/**
 * Centralized API configuration for InvestIQ mobile app.
 * Supports ngrok tunnel, local IP (LAN), and localhost fallbacks.
 * 
 * Priority order:
 * 1. Environment variables (EXPO_PUBLIC_API_URL)
 * 2. ngrok URL (most stable for remote devices)
 * 3. Local IP / LAN (for devices on same network)
 * 4. localhost (for simulator/emulator)
 */

// ─── Network Configuration ────────────────────────────────────────────────────
export const API_HOST = '172.19.120.162';  // Your LAN IPv4 address
export const API_PORT = 8000;

// ngrok public URL - update with your current ngrok URL
// Get this from running: npx ngrok http 8000
// Or check: https://dashboard.ngrok.com
export const NGROK_URL = process.env.EXPO_PUBLIC_NGROK_URL || 'https://larraine-supervigorous-shiplessly.ngrok-free.dev/api/v1';

export const API_LAN_URL = `http://${API_HOST}:${API_PORT}/api/v1`;
export const API_LOCALHOST_URL = `http://127.0.0.1:${API_PORT}/api/v1`;

// ─── Primary API Base URL (ngrok recommended for Expo tunnel mode) ────────────
export const API_BASE_URL = process.env.EXPO_PUBLIC_API_URL || NGROK_URL;

// ─── Fallback candidates (tried in order) ────────────────────────────────────
// Order: ngrok → LAN → localhost
export const API_BASE_URL_CANDIDATES = [
    process.env.EXPO_PUBLIC_API_URL,
    NGROK_URL,
    API_LAN_URL,
    API_LOCALHOST_URL,
].filter(Boolean);

export const REQUEST_TIMEOUT_MS = 60000;

export const NETWORK_ERROR_MESSAGE =
    `Network error: cannot reach backend at ${API_BASE_URL}. ` +
    'Ensure:\n' +
    '1. Backend is running: uvicorn backend.app.main:app --host 0.0.0.0 --port 8000\n' +
    '2. ngrok tunnel is active: npx ngrok http 8000\n' +
    '3. Phone and computer are on the same network (for LAN fallback)\n' +
    '4. NGROK_URL in config/api.js matches your current ngrok URL';


