# InvestIQ - Deployment Guide & Infrastructure Architecture

**Complete deployment procedures, infrastructure setup, and architecture diagrams**

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Local Development Setup](#local-development-setup)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [Monitoring & Maintenance](#monitoring--maintenance)
6. [Troubleshooting](#troubleshooting)

---

## High-Level Architecture

### System Components

```
┌──────────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACE LAYER                           │
├──────────────────────────────────────────────────────────────────────────┤
│  React Native Mobile App (iOS/Android via Expo)                           │
│  ├─ Stock Signal Cards (BUY/SELL/HOLD display)                          │
│  ├─ Portfolio Management UI                                              │
│  ├─ Trading History & Analytics                                          │
│  └─ User Authentication (JWT-based)                                      │
└────────────────┬─────────────────────────────────────────────────────────┘
                 │ HTTPS/REST
                 ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                         API GATEWAY & ROUTING LAYER                       │
├──────────────────────────────────────────────────────────────────────────┤
│  Load Balancer (Nginx / AWS ALB)                                         │
│  ├─ Route to /api/v1/* → Backend instances                              │
│  ├─ SSL/TLS termination                                                  │
│  ├─ Rate limiting                                                        │
│  └─ Request/Response caching                                             │
└────────────────┬─────────────────────────────────────────────────────────┘
                 │ Internal Network
                 ↓
┌──────────────────────────────────────────────────────────────────────────┐
│                        APPLICATION LAYER (FastAPI)                        │
├──────────────────────────────────────────────────────────────────────────┤
│  Backend API Servers (4 instances behind load balancer)                   │
│  ├─ /api/v1/predict ──→ Predictor.predict()                             │
│  ├─ /api/v1/train ────→ training.train_pipeline()                       │
│  ├─ /api/v1/portfolio → PortfolioOptimizer.optimize()                   │
│  ├─ /api/v1/risk/score → RiskCalculator.score()                         │
│  └─ /api/v1/batch/signals → batch_predict()                             │
│                                                                           │
│  All instances connected to:                                             │
│  ├─ Model Cache (Redis) - Fast feature store access                     │
│  ├─ Database (PostgreSQL) - User data, training configs                 │
│  └─ File Storage (S3/EBS) - Model artifacts, logs                       │
└────────────┬──────────────────────────────────────────────────────────────┘
             │
    ┌────────┴─────────┬──────────────┬─────────────┐
    ↓                  ↓              ↓             ↓
┌─────────────┐  ┌──────────┐  ┌─────────────┐  ┌──────────┐
│ Redis Cache │  │PostgreSQL│  │  File/S3    │  │  Logs    │
│ (Feature    │  │ Database │  │  Storage    │  │  (e.g.   │
│  Store)     │  │(Users,   │  │(Models,     │  │CloudWatch
│             │  │Training  │  │Indicators,  │  │or ELK)   │
│             │  │Configs)  │  │Backtests)   │  │          │
└─────────────┘  └──────────┘  └─────────────┘  └──────────┘
    ↓                              ↑
    └──────────────┬───────────────┘
                   │
    ┌──────────────┴────────────────┐
    ↓                               ↓
┌──────────────────────┐    ┌──────────────────────┐
│  ML INFERENCE LAYER  │    │  TRAINING LAYER      │
├──────────────────────┤    ├──────────────────────┤
│  LSTM Model          │    │  Training Pipeline   │
│  ├─ PyTorch tensor   │    │  ├─ Data loading     │
│  ├─ 64 hidden units  │    │  ├─ Feature eng.     │
│  ├─ 2 layers         │    │  ├─ LSTM training    │
│  └─ Output:prob(UP)  │    │  ├─ XGBoost training │
│                      │    │  ├─ Validation       │
│  XGBoost Model       │    │  ├─ Metrics calc.    │
│  ├─ 500 estimators   │    │  └─ Model saving     │
│  ├─ Max depth 7      │    │                      │
│  └─ Output:prob(UP)  │    │  (Background job)    │
│                      │    │  - Scheduled daily   │
│  Ensemble (50-50)    │    │  - Auto-retrain      │
│  └─ Weighted avg     │    │  - Logs stored       │
└──────────────────────┘    └──────────────────────┘
    ↑                               ↓
    └───────────────┬───────────────┘
                    │
          ┌─────────┴──────────┐
          ↓                    ↓
    ┌───────────────┐   ┌──────────────────┐
    │ Stock Data     │   │ Market Data      │
    │ (CSV/Database) │   │(NIFTY 50, Macro)│
    │  - HDFCBANK    │   │(Via yfinance API)│
    │  - INFY        │   │                  │
    │  - RELIANCE    │   │  Updated Daily   │
    │  - TCS         │   │  @ Market Close  │
    │  - ICICIBANK   │   │  (18:00 IST)     │
    └───────────────┘   └──────────────────┘
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ REAL-TIME PREDICTION REQUEST (User clicks "Get Signal")    │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────▼──────────────┐
        │  Frontend (React Native)  │
        │  POST /api/v1/predict     │
        │  Body: {symbol: "INFY"}   │
        └────────────┬──────────────┘
                     │ HTTP/REST (500 ms)
                     │
        ┌────────────▼──────────────────────────┐
        │  Backend FastAPI Server              │
        │  Route Handler: predict()             │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────▼──────────────────────────┐
        │  Data Loading (50 ms)                 │
        │  └─ Read INFY.csv from storage       │
        │     or cache from Redis              │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────▼──────────────────────────┐
        │  Feature Engineering (150 ms)         │
        │  └─ Create 50 features:              │
        │     - RSI, MACD, Bollinger Bands     │
        │     - OBV, Volume ratios             │
        │     - Lag returns (1-5 days)         │
        │     - Trend scores                   │
        │     - Market correlation            │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────▼──────────────────────────┐
        │  Normalization (20 ms)                │
        │  └─ StandardScaler                    │
        │     (loaded from pickle file)         │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────▼──────────────────────────┐
        │  Model Inference (100 ms)             │
        │  ├─ LSTM forward pass                │
        │  │  └─ Input: (1, 20, 50) tensor    │
        │  │  └─ Output: P(UP) = 0.68          │
        │  │                                   │
        │  └─ XGBoost prediction               │
        │     └─ Input: (1, 50) array         │
        │     └─ Output: P(UP) = 0.65          │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────▼──────────────────────────┐
        │  Ensemble (10 ms)                     │
        │  └─ final_score = 0.5*0.68 + 0.5*0.65│
        │  └─ final_score = 0.665              │
        │  └─ confidence = 66.5%               │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────▼──────────────────────────┐
        │  Signal Generation (10 ms)            │
        │  if 0.665 > 0.60:                    │
        │    signal = "BUY"                     │
        │    confidence = 0.665                │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────▼──────────────────────────┐
        │  Price Prediction (15 ms)             │
        │  └─ predicted_return = 3.5%           │
        │  └─ current_price = 500               │
        │  └─ predicted_price = 517.50          │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────▼──────────────────────────┐
        │  Risk Calculation (10 ms)             │
        │  └─ volatility = 1.2%                 │
        │  └─ risk_level = "Low"                │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────▼──────────────────────────┐
        │  API Response Formatting (10 ms)      │
        │  └─ JSON serialization                │
        │  └─ HTTP 200 OK                       │
        └────────────┬──────────────────────────┘
                     │ Response (350-400 ms total)
                     │
        ┌────────────▼──────────────────────────┐
        │  Frontend Display                     │
        │  ├─ BUY button (green)               │
        │  ├─ Confidence: 66.5%                 │
        │  ├─ Current: $500 → Predicted: $517.50
        │  ├─ Change: +3.5%                     │
        │  └─ Risk: Low                         │
        └────────────────────────────────────────┘
```

---

## Local Development Setup

### Windows (Recommended)

#### Prerequisites
- Python 3.10+
- Node.js 16+
- Git
- Visual Studio Code

#### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/InvestIQ-main.git
cd InvestIQ-main
```

#### Step 2: Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch (GPU or CPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# OR CPU-only version:
pip install torch torchvision torchaudio

# Verify installation
python -c "import torch; print(torch.__version__)"
```

#### Step 3: Download Stock Data

```bash
# Inside backend/
python data/download_historical_data.py

# This downloads 5 stocks:
# - HDFCBANK.csv
# - ICICIBANK.csv
# - INFY.csv
# - RELIANCE.csv
# - TCS.csv
# Into: backend/data/stock_data/
```

#### Step 4: Train Models (First Run)

```bash
# Train all stocks (takes ~10-15 minutes)
python training/train_improved_hybrid_models.py --verbose

# OR train single stock (faster for testing)
python training/train_improved_hybrid_models.py --ticker HDFCBANK

# Models saved to: backend/models/saved_models/
```

#### Step 5: Start Backend API

```bash
# Run FastAPI development server
uvicorn app.main:app --reload --port 8000

# Output:
# INFO:     Uvicorn running on http://127.0.0.1:8000
# INFO:     Application startup complete
# Visit http://127.0.0.1:8000/api/v1/docs for API documentation
```

#### Step 6: Frontend Setup

```bash
# Open new terminal, navigate to frontend
cd InvestIQ-App

# Install dependencies
npm install

# Install Expo CLI globally (if not installed)
npm install -g expo-cli

# Start Metro bundler
npm start

# Output:
# ┌─────────────────────────────────────────────────────────┐
# │  Metro waiting on exp://192.168.x.x:19000              │
# │  Press 'a' for Android (requires emulator)              │
# │  Press 'i' for iOS (Mac only)                           │
# │  Press 'w' for web                                       │
# │  Press 'c' to clear Metro bundler cache                 │
# └─────────────────────────────────────────────────────────┘

# Press 'w' for web preview, or 'a' for Android emulator
```

#### Step 7: Test Integration

```bash
# Backend should be running on: http://127.0.0.1:8000
# Frontend should be running on: http://localhost:19000 (web)

# Test API endpoint (in new terminal)
curl -X GET http://127.0.0.1:8000/api/v1/health

# Expected response:
# {"status":"ok","version":"2.0.0"}

# Test prediction
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"HDFCBANK"}'

# Expected response (example):
# {
#   "signal": "BUY",
#   "signal_confidence": 0.72,
#   "current_price": 1524.50,
#   "predicted_price": 1565.30,
#   ...
# }
```

### Directory Structure After Setup

```
InvestIQ-main/
├─ backend/
│  ├─ venv/                    # Virtual environment
│  ├─ app/                     # FastAPI routes
│  ├─ training/                # Training scripts
│  │  ├─ train_improved_hybrid_models.py
│  │  ├─ improved_hybrid_model.py
│  │  └─ evaluation_module.py
│  ├─ inference/               # Prediction scripts
│  │  └─ predict.py
│  ├─ models/
│  │  └─ saved_models/         # Trained models (created after training)
│  │     ├─ lstm_HDFCBANK.pth
│  │     ├─ xgboost_HDFCBANK.pkl
│  │     └─ scaler_HDFCBANK.pkl
│  ├─ data/
│  │  └─ stock_data/           # Stock CSV files (downloaded)
│  │     ├─ HDFCBANK.csv
│  │     ├─ INFY.csv
│  │     └─ ...
│  ├─ requirements.txt
│  └─ run_server.bat           # Start backend
│
├─ InvestIQ-App/
│  ├─ node_modules/            # Dependencies (after npm install)
│  ├─ src/
│  │  ├─ services/
│  │  │  └─ api.js             # Backend API calls
│  │  ├─ components/           # React components
│  │  └─ context/              # Global state
│  ├─ package.json
│  └─ run_expo.ps1             # Start frontend
│
├─ COMPLETE_SYSTEM_DOCUMENTATION.md        # Main documentation
├─ CALCULATIONS_AND_METRICS_QUICK_REFERENCE.md
└─ DEPLOYMENT_GUIDE.md                      # This file
```

---

## Docker Deployment

### Build Docker Image

```bash
# From project root directory
docker build -t investiq-backend:latest .

# Verify build
docker images | grep investiq

# Output:
# REPOSITORY          TAG       IMAGE ID      CREATED      SIZE
# investiq-backend    latest    abc123def456  2 mins ago   1.2GB
```

### Run Single Container

```bash
# Create data volume
docker volume create investiq-data

# Run container
docker run -d \
  --name investiq-api \
  -p 8000:8000 \
  -v investiq-data:/app/backend/data \
  -v investiq-models:/app/backend/models \
  -e ENVIRONMENT=development \
  -e LOG_LEVEL=INFO \
  investiq-backend:latest

# Check logs
docker logs investiq-api

# Stop container
docker stop investiq-api

# Remove container
docker rm investiq-api
```

### Docker Compose (Multi-Container Setup)

#### docker-compose.yml

```yaml
version: '3.8'

services:
  # Backend API
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: investiq-backend
    ports:
      - "8000:8000"
    environment:
      ENVIRONMENT: production
      DATABASE_URL: postgresql://investiq:secure_password@db:5432/investiq
      REDIS_URL: redis://redis:6379/0
      LOG_LEVEL: INFO
    depends_on:
      - db
      - redis
    volumes:
      - ./backend/models:/app/backend/models
      - ./backend/data:/app/backend/data
      - ./logs:/app/logs
    networks:
      - investiq-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # PostgreSQL Database
  db:
    image: postgres:14-alpine
    container_name: investiq-db
    environment:
      POSTGRES_USER: investiq
      POSTGRES_PASSWORD: secure_password
      POSTGRES_DB: investiq
    volumes:
      - db_volume:/var/lib/postgresql/data
    networks:
      - investiq-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U investiq"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: investiq-redis
    volumes:
      - redis_volume:/data
    networks:
      - investiq-network
    command: redis-server --appendonly yes
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Nginx Load Balancer (optional, for production)
  nginx:
    image: nginx:alpine
    container_name: investiq-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - backend
    networks:
      - investiq-network
    restart: unless-stopped

networks:
  investiq-network:
    driver: bridge

volumes:
  db_volume:
  redis_volume:
```

#### Start Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f db

# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## Production Deployment

### AWS Deployment (Recommended)

#### Architecture

```
Internet → CloudFront (CDN) → ALB → ECS Fargate Clusters →
  → RDS PostgreSQL
  → ElastiCache (Redis)
  → S3 (Models & Data)
```

#### Step 1: Prepare AWS Resources

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name investiq-backend --region us-east-1

# 2. Authenticate Docker with ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com

# 3. Tag Docker image
docker tag investiq-backend:latest <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/investiq-backend:latest

# 4. Push to ECR
docker push <aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/investiq-backend:latest
```

#### Step 2: Create RDS PostgreSQL Database

```bash
# Create database
aws rds create-db-instance \
  --db-instance-identifier investiq-db \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username investiq \
  --master-user-password secure_password_here \
  --allocated-storage 20 \
  --publicly-accessible false \
  --vpc-security-group-ids sg-xxxxxxxx
```

#### Step 3: Create ElastiCache Redis

```bash
# Create cache cluster
aws elasticache create-cache-cluster \
  --cache-cluster-id investiq-redis \
  --cache-node-type cache.t3.micro \
  --engine redis \
  --engine-version 7.0
```

#### Step 4: Create ECS Cluster & Task Definition

```bash
# Create cluster
aws ecs create-cluster --cluster-name investiq-cluster

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

#### task-definition.json

```json
{
  "family": "investiq-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "investiq",
      "image": "<aws_account_id>.dkr.ecr.us-east-1.amazonaws.com/investiq-backend:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "hostPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        },
        {
          "name": "DATABASE_URL",
          "value": "postgresql://investiq:password@investiq-db.xxxxx.rds.amazonaws.com:5432/investiq"
        },
        {
          "name": "REDIS_URL",
          "value": "redis://investiq-redis.xxxxx.cache.amazonaws.com:6379/0"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/investiq-backend",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Create ECS Service

```bash
aws ecs create-service \
  --cluster investiq-cluster \
  --service-name investiq-backend-service \
  --task-definition investiq-backend:1 \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxxxx,subnet-yyyyy],securityGroups=[sg-zzzzz]}"
```

### Environment Variables for Production

```bash
# .env.production

# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4
CORS_ORIGINS=https://investiq.app,https://www.investiq.app

# Database
DATABASE_URL=postgresql://investiq:${DB_PASSWORD}@investiq-db.xxxxx.rds.amazonaws.com:5432/investiq
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Cache
REDIS_URL=redis://investiq-redis.xxxxx.cache.amazonaws.com:6379/0
CACHE_TTL=3600

# Security
JWT_SECRET=${JWT_SECRET_KEY}
JWT_EXPIRY_HOURS=24
ALLOWED_HOSTS=investiq.app,*.investiq.app

# Models
MODEL_DIR=/app/backend/models/saved_models
DATA_DIR=/app/backend/data/stock_data
INFERENCE_MODE=hybrid
HYBRID_FALLBACK_TO_LEGACY=true

# Storage
AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY}
AWS_SECRET_ACCESS_KEY=${AWS_SECRET_KEY}
AWS_S3_BUCKET=investiq-models
AWS_REGION=us-east-1

# Monitoring
SENTRY_DSN=${SENTRY_DSN}
DATADOG_API_KEY=${DATADOG_API_KEY}
```

---

## Monitoring & Maintenance

### Health Check Endpoints

```bash
# Basic health check
curl http://localhost:8000/api/v1/health

# Response:
{
  "status": "ok",
  "version": "2.0.0"
}

# Check model availability
curl http://localhost:8000/api/v1/debug/predict/INFY \
  -H "Authorization: Bearer <token>"

# Response:
{
  "status": "success",
  "symbol": "INFY",
  "signal": "BUY",
  "signal_confidence": 0.72,
  "current_price": 1524.50,
  "predicted_price": 1565.30
}

# Check database connection
curl http://localhost:8000/api/v1/health/db

# Check cache connection
curl http://localhost:8000/api/v1/health/cache
```

### Logging

```
# Logs stored in: /app/logs/

# By level
/app/logs/info.log        # INFO level messages
/app/logs/error.log       # ERROR level messages
/app/logs/debug.log       # DEBUG level messages (dev only)

# View logs (Docker)
docker logs investiq-api | tail -100

# View logs (Production)
tail -f /app/logs/info.log
```

### Model Retraining

```bash
# Manual retrain (via API)
curl -X POST http://localhost:8000/api/v1/retrain/trigger \
  -H "Authorization: Bearer <admin_token>" \
  -H "Content-Type: application/json"

# Response:
{
  "status": "Retrain initiated",
  "job_id": "retrain_20260415_120000",
  "estimated_completion": "2026-04-15T12:30:00Z"
}

# Check retrain status
curl http://localhost:8000/api/v1/retrain/status \
  -H "Authorization: Bearer <admin_token>"

# Response:
{
  "status": "in_progress",
  "job_id": "retrain_20260415_120000",
  "progress": 65,
  "tickers_completed": ["HDFCBANK", "INFY"],
  "tickers_remaining": ["RELIANCE", "TCS", "ICICIBANK"]
}
```

---

## Troubleshooting

### Common Issues

#### 1. Models Not Found

```bash
# Symptom:
# ERROR: Failed to load LSTM model for HDFCBANK

# Solution:
# 1. Check models exist
ls -la backend/models/saved_models/

# 2. If missing, retrain
python backend/training/train_improved_hybrid_models.py --ticker HDFCBANK

# 3. Verify paths in config
cat backend/core/config.py | grep MODEL_DIR
```

#### 2. Out of Memory

```bash
# Symptom:
# MemoryError: Unable to allocate 2.5 GiB for an array

# Solution for training:
# 1. Reduce batch size
BATCH_SIZE=64  # from 128

# 2. Reduce sequence length
SEQ_LENGTH=15  # from 20

# 3. Use mixed precision (faster, less memory)
# Already enabled in PHASE 2 config

# Solution for inference:
# 1. Reduce concurrent requests
# 2. Increase memory allocation (if Docker/VM)
```

#### 3. Database Connection Error

```bash
# Symptom:
# psycopg2.OperationalError: could not connect to server

# Solution:
# 1. Check database running
docker ps | grep postgres
# or
pg_isready -h localhost -p 5432

# 2. Verify credentials
DATABASE_URL=postgresql://user:password@host:5432/db

# 3. Check firewall/security groups
# AWS: Modify RDS security group
# Local: Ensure PostgreSQL service running
```

#### 4. API Timeout (>10 seconds)

```bash
# Symptom:
# Request hangs for 10+ seconds

# Check what's slow:
# 1. Feature engineering too slow?
#    → Check if data loading cached in Redis
#    → Optimize indicator calculations

# 2. Model inference too slow?
#    →Check GPU usage
#    →Consider model quantization (Q8)

# 3. Database query slow?
#    → Check for missing indexes
#    → Monitor query performance

# Solution:
# Implement caching
redis-cli FLUSHALL  # Clear cache if corrupted

# Monitor performance
curl http://localhost:8000/api/v1/debug/timing
```

#### 5. Predictions Consistently Wrong

```bash
# Symptom:
# All predictions BUY (or wrong signal)

# Diagnosis:
python << 'EOF'
import pandas as pd
from backend.training.improved_hybrid_model import ProductionTrainingPipeline

pipeline = ProductionTrainingPipeline("HDFCBANK")
df = pipeline.load_and_preprocess("backend/data/stock_data/HDFCBANK.csv")

# Check features
print("Features shape:", df.shape)
print("NaN count:", df.isnull().sum())
print("Feature stats:")
print(df.describe())

# Check labels
from backend.training.improved_hybrid_model import SmartLabelEngineer
labels, returns = SmartLabelEngineer.create_binary_labels(df)
print("Label distribution:")
print(f"UP: {(labels==1).sum()}, DOWN: {(labels==0).sum()}")
EOF

# Solution:
# 1. Retrain models with fresh data
# 2. Check data quality
# 3. Review feature engineering
# 4. Check model checkpoints exist
```

---

## Performance Optimization

### For Development

```bash
# Use development settings (faster iteration)
ENVIRONMENT=development
EPOCHS=10  # Instead of 50
BATCH_SIZE=64

# Profile code performance
python -m cProfile -s cumulative backend/app/main.py | head -50

# Monitor memory
docker stats investiq-api --no-trunc
```

### For Production

```bash
# Use production settings (best accuracy)
ENVIRONMENT=production
EPOCHS=100
BATCH_SIZE=128
API_WORKERS=4  # GUnicorn workers

# Enable caching
CACHE_TTL=3600  # Cache predictions for 1 hour

# Use GPU if available
CUDA_VISIBLE_DEVICES=0

# Monitor metrics
docker stats --no-stream
```

---

**Deployment Guide Complete**  
**Updated**: April 15, 2026  
**Version**: 1.0
