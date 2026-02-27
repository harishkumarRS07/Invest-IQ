# InvestIQ Mobile App 📱

> **AI-powered fintech mobile app** connecting to the InvestIQ FastAPI backend.
> Built with Expo (React Native + JavaScript), featuring professional fintech design, JWT authentication, live trading signals, interactive charts, and push notifications.

## ✨ Features

| Feature | Status |
|---|---|
| JWT Login / Register | ✅ |
| AI Signal Dashboard | ✅ |
| Stock Detail with 7-Day Chart | ✅ |
| Technical Indicators (RSI, MACD, BB) | ✅ |
| AI Explanation | ✅ |
| Portfolio Optimization (PieChart) | ✅ |
| Push Notifications | ✅ |
| Pull-to-Refresh | ✅ |
| Loading Skeletons | ✅ |
| Dark Mode | ✅ |
| Error Handling | ✅ |

---

## 📁 Project Structure

```
InvestIQ-App/
├── app/
│   ├── _layout.js              # Root layout (AuthGuard + Providers)
│   ├── (auth)/
│   │   ├── _layout.js
│   │   ├── login.js            # Login screen
│   │   └── register.js         # Register screen
│   ├── (tabs)/
│   │   ├── _layout.js          # Tab bar
│   │   ├── dashboard.js        # AI Signals feed
│   │   ├── portfolio.js        # Portfolio allocation
│   │   └── settings.js         # Settings + notifications
│   └── stock/
│       └── [symbol].js         # Stock detail + chart
├── src/
│   ├── components/
│   │   ├── ui.js               # Reusable components
│   │   └── StockSignalCard.js  # Dashboard card
│   ├── constants/
│   │   └── theme.js            # Design tokens
│   ├── context/
│   │   ├── AuthContext.js      # JWT auth state
│   │   └── ThemeContext.js     # Dark/Light mode
│   ├── hooks/
│   │   └── useStockData.js     # API hooks
│   └── services/
│       ├── api.js              # Axios + all API calls
│       └── notifications.js    # Push notification helpers
├── app.json
├── babel.config.js
└── package.json
```

---

## 🚀 Quick Start

### 1. Backend Setup

```bash
# From InvestIQ-main/
cd backend
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Mobile App Setup

```bash
cd InvestIQ-App
npm install
```

### 3. Configure Backend URL

Edit `src/services/api.js`:

```js
// Android Emulator
export const BASE_URL = 'http://10.0.2.2:8000/api/v1';

// iOS Simulator
// export const BASE_URL = 'http://localhost:8000/api/v1';

// Physical Device (replace with your PC's LAN IP)
// export const BASE_URL = 'http://192.168.1.XXX:8000/api/v1';
```

> **Find your LAN IP:** `ipconfig` (Windows) → IPv4 Address

### 4. Start the App

```bash
# Start Expo dev server
npm start

# Run on Android
npm run android

# Run on iOS
npm run ios
```

---

## 🔐 Security Best Practices

| Practice | Implementation |
|---|---|
| **JWT stored securely** | `expo-secure-store` (Keychain / Keystore) |
| **Tokens never in AsyncStorage** | Only `SecureStore` used for auth |
| **Bearer token on every request** | Axios request interceptor |
| **HMAC-SHA256 token signing** | Server-side, constant-time compare |
| **Password hashing** | SHA-256 (upgrade to bcrypt in production) |
| **HTTPS in production** | Configure via reverse proxy (Nginx) |
| **Token expiry** | 24 hours — auto logout on expiry |
| **Input validation** | Pydantic on backend, client-side checks |

---

## 🏭 Production Improvements

### Backend
1. **Replace JSON user store** with PostgreSQL / SQLite via SQLAlchemy
2. **Use bcrypt** instead of SHA-256 for password hashing
3. **Refresh tokens** — implement token rotation
4. **Rate limiting** — add slowapi or nginx rate limits
5. **Production CORS** — restrict `allow_origins` to your domain
6. **Indicators in prediction** — wire `_extract_indicators()` directly from the predictor

### Mobile App
1. **EAS Build** for production APK/IPA
2. **Sentry** for crash reporting
3. **React Query** for advanced server state caching
4. **Offline mode** — cache last signals with AsyncStorage
5. **Biometric auth** — add `expo-local-authentication`
6. **Stock watchlist** — let users pin favorite tickers

---

## 📦 Deployment (Expo EAS)

```bash
# Install EAS CLI
npm install -g eas-cli

# Login to Expo account
eas login

# Configure build
eas build:configure

# Build Android APK
eas build --platform android --profile preview

# Build for both (production)
eas build --platform all
```

---

## 🎨 Design System

| Token | Value |
|---|---|
| Background | `#0A0E1A` (deep navy) |
| Card | `#151E2E` |
| Brand Purple | `#7B61FF` |
| BUY Green | `#00D07C` |
| SELL Red | `#FF5353` |
| HOLD Amber | `#F5A623` |

---

## 📡 API Endpoints Used

| Endpoint | Purpose |
|---|---|
| `POST /auth/register` | Create account |
| `POST /auth/login` | Get JWT |
| `GET /auth/me` | Verify token |
| `GET /tickers` | List available stocks |
| `POST /signals/batch` | Dashboard signals |
| `POST /predict` | Full stock detail |
| `POST /portfolio/optimize` | Portfolio allocation |
| `GET /health` | Backend health check |

---

> Made with ❤️ for InvestIQ Hackathon — 2026
