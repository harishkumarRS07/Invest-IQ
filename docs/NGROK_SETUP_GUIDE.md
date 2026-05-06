# InvestIQ ngrok Backend Tunnel Setup Guide

This guide explains how to use ngrok to expose your local FastAPI backend to your Expo React Native mobile app securely and reliably.

## 📋 Requirements

- **Backend**: FastAPI + Uvicorn running on port 8000
- **Frontend**: Expo React Native app
- **ngrok**: Account and CLI installed (https://ngrok.com)
- **curl** & **jq** (optional, for update scripts)

## 🚀 Quick Start

### 1. Install ngrok

Download from: https://ngrok.com/download

Verify installation:
```bash
ngrok --version
```

### 2. Authenticate ngrok

Use your auth token (provided in your account):
```bash
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

For this project, the token is already configured.

### 3. Start Backend

Ensure your FastAPI backend runs on port 8000 with `host="0.0.0.0"`:

```bash
cd d:\InvestIQ-main
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Verify it's running:
```bash
curl http://localhost:8000/api/v1/health
# Should return: {"status":"ok","version":"2.0.0"}
```

### 4. Start ngrok Tunnel

**Option A: Using automated Windows batch script (recommended)**

```bash
.\start_ngrok_tunnel.bat
```

This will:
- Start ngrok on port 8000
- Automatically extract and update the frontend config with the ngrok URL
- Display the ngrok Web UI link

**Option B: Manual ngrok startup**

```bash
ngrok http 8000
```

Watch the output for the HTTPS forwarding URL:
```
Forwarding    https://larraine-supervigorous-shiplessly.ngrok-free.dev -> http://localhost:8000
```

### 5. Update Frontend Config

If using manual startup, update the frontend with the ngrok URL:

**Windows:**
```powershell
.\update_ngrok_url.ps1
```

**macOS/Linux:**
```bash
./get_ngrok_url.sh
```

Or manually edit:
```javascript
// InvestIQ-App/config/api.js
export const NGROK_URL = 'https://larraine-supervigorous-shiplessly.ngrok-free.dev/api/v1';
```

### 6. Restart Expo App

```bash
cd InvestIQ-App
npx expo start --tunnel -c
```

### 7. Test Login

Open the app on your mobile device and try logging in. The app will:
1. Try ngrok URL first (fastest for mobile devices)
2. Fall back to LAN IP if ngrok is unavailable
3. Fall back to localhost on simulator/emulator

## 🔍 Monitoring & Debugging

### View ngrok Traffic
Open ngrok Web interface in your browser:
```
http://127.0.0.1:4040
```

This shows:
- All HTTP/HTTPS traffic through the tunnel
- Request/response headers and payloads
- Connection diagnostics

### Check ngrok Status
List active tunnels:
```bash
curl http://127.0.0.1:4040/api/tunnels
```

### Backend Health Check
```bash
curl -X GET http://localhost:8000/api/v1/health
```

### Authentication Test
```bash
# Without auth (should fail with 401)
curl -X GET https://larraine-supervigorous-shiplessly.ngrok-free.dev/api/v1/auth/me

# With auth (need valid JWT token)
curl -X GET https://larraine-supervigorous-shiplessly.ngrok-free.dev/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## ⚙️ Configuration

### Priority Order for API URLs

The frontend tries connections in this order:

1. **Environment Variable**: `EXPO_PUBLIC_API_URL`
   ```bash
   export EXPO_PUBLIC_API_URL="https://your-custom-url/api/v1"
   ```

2. **ngrok URL** (primary)
   ```javascript
   export const NGROK_URL = 'https://larraine-supervigorous-shiplessly.ngrok-free.dev/api/v1';
   ```

3. **Local IP / LAN**
   ```javascript
   export const API_LAN_URL = 'http://172.19.120.162:8000/api/v1';
   ```

4. **Localhost**
   ```javascript
   export const API_LOCALHOST_URL = 'http://127.0.0.1:8000/api/v1';
   ```

### Fallback Strategy

If one connection fails, the app automatically tries the next candidate. This ensures reliability even if ngrok temporarily disconnects.

## 🔐 Security Notes

- **ngrok free tier limitations**:
  - Some requests may be rate-limited
  - Session can reconnect periodically
  - For production, consider ngrok paid plan or self-hosted tunnel

- **HTTPS enforced**: All ngrok URLs use HTTPS for data encryption

- **Token management**: Auth tokens stored securely in Expo SecureStore

## 📊 Troubleshooting

### 503 Service Unavailable
```bash
# Check backend is running on port 8000
netstat -ano | findstr "8000"

# Verify health endpoint
curl http://localhost:8000/api/v1/health
```

### ngrok URL Not Updating
Manually update config/api.js:
```javascript
export const NGROK_URL = 'https://YOUR-NEW-URL.ngrok-free.dev/api/v1';
```

### Connection Timeout
- Check firewall settings
- Ensure backend `host="0.0.0.0"` (not localhost)
- Verify ngrok tunnel is active

### Mobile App Still Seeing Old URL
Clear Expo cache:
```bash
npx expo start --tunnel -c
```

The `-c` flag clears the cache and forces a fresh download of your app code.

## 📱 Testing on Different Devices

### Physical Phone (iOS/Android)
1. Scan Expo QR code with Expo Go app
2. App will use ngrok URL (works over internet)
3. Your phone doesn't need to be on same Wi-Fi

### Emulator/Simulator
1. App can use localhost (emulator/simulator access to host machine)
2. Or use ngrok URL (more realistic testing)
3. Configure in `InvestIQ-App/config/api.js`

### LAN Testing (Same Network)
1. Ensure phone and computer on same Wi-Fi
2. App falls back to `http://172.19.120.162:8000/api/v1` if ngrok unavailable
3. Fast and reliable for local testing

## 🔄 Automatic URL Updates

When ngrok restarts, it may assign a new URL. To avoid manual updates:

### Option 1: Batch Script (Windows)
```batch
.\start_ngrok_tunnel.bat
```

### Option 2: PowerShell Script
```powershell
.\update_ngrok_url.ps1
```

### Option 3: Bash Script
```bash
./get_ngrok_url.sh
```

These scripts automatically:
1. Query ngrok API at `http://127.0.0.1:4040/api/tunnels`
2. Extract current public HTTPS URL
3. Update `InvestIQ-App/config/api.js` with new URL
4. Display success confirmation

## 📚 API Endpoints

All endpoints require authentication (Bearer JWT token in Authorization header):

```bash
# Register / Login (no auth needed)
POST /api/v1/auth/register
POST /api/v1/auth/login

# Authenticated endpoints
GET  /api/v1/auth/me
POST /api/v1/predict
POST /api/v1/signals/batch
GET  /api/v1/tickers
POST /api/v1/sentiment/analyze
```

Example with ngrok URL:
```bash
curl -X POST https://larraine-supervigorous-shiplessly.ngrok-free.dev/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password123"}'
```

## 🎯 Best Practices

1. **Keep ngrok tunnel running** while testing mobile app
2. **Monitor ngrok dashboard** to see all traffic and diagnose issues
3. **Use `.env` files** for environment secrets (not in version control)
4. **Test on actual device** periodically (not just emulator)
5. **Keep backend logs open** to see server-side errors
6. **Use ngrok free plan first**, upgrade if you need features like IP whitelisting

## 📞 Support

For ngrok issues:
- Official docs: https://ngrok.com/docs
- Status page: https://status.ngrok.com

For InvestIQ issues:
- Check backend logs: `uvicorn` terminal
- Check frontend logs: Expo console
- Check ngrok traffic: `http://127.0.0.1:4040`

---

**Current Setup**:
- Backend: `uvicorn` on port 8000 ✅
- ngrok tunnel: `https://larraine-supervigorous-shiplessly.ngrok-free.dev` ✅
- Frontend: React Native + Expo ✅
- API config: `InvestIQ-App/config/api.js` ✅
