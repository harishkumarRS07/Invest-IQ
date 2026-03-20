from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from backend.core.config import settings
from backend.core.logging import logger
from backend.app.routes import router
from backend.app.auth import get_current_user
from backend.training.auto_retrain import trigger_retrain, get_retrain_status

from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from backend.data.update_stock_data import refined_update
from typing import Optional
from fastapi import Header, HTTPException
from time import perf_counter

# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_auth_header(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization[len("Bearer "):]
    user = get_current_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return user

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = AsyncIOScheduler()

    # ① Daily stock data update – every day at 18:00 (after market close)
    scheduler.add_job(refined_update, 'cron', hour=18, minute=0,
                      id='daily_data_update', name='Daily Stock Data Update')

    # ② Weekly model retraining is disabled (Sunday schedule turned off).

    scheduler.start()
    print("Scheduler started.")
    print("  -> Daily data update: every day at 18:00")
    print("  -> Weekly retrain:    disabled")

    yield

    scheduler.shutdown()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix=settings.API_V1_STR)

@app.middleware("http")
async def log_requests(request, call_next):
    start = perf_counter()
    client = request.client.host if request.client else "unknown"
    method = request.method
    path = request.url.path
    try:
        response = await call_next(request)
        elapsed_ms = (perf_counter() - start) * 1000
        logger.info(f"HTTP {method} {path} from {client} -> {response.status_code} ({elapsed_ms:.1f} ms)")
        return response
    except Exception:
        elapsed_ms = (perf_counter() - start) * 1000
        logger.exception(f"HTTP {method} {path} from {client} -> 500 ({elapsed_ms:.1f} ms)")
        raise

# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "InvestIQ API v2. See /api/v1/docs for documentation."}

# ── Retrain Management Endpoints ──────────────────────────────────────────────

@app.get(f"{settings.API_V1_STR}/retrain/status")
def retrain_status(user: dict = Depends(_require_auth_header)):
    """Returns the current state of the auto-retrain system."""
    return get_retrain_status()

@app.post(f"{settings.API_V1_STR}/retrain/trigger")
def retrain_trigger(user: dict = Depends(_require_auth_header)):
    """
    Manually trigger an immediate model retrain in the background.
    Useful for testing or forcing a refresh outside of the Sunday schedule.
    """
    started = trigger_retrain(background=True)
    if not started:
        raise HTTPException(status_code=409, detail="Retrain already in progress.")
    return {"message": "Retrain started in background. Check /retrain/status for progress."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=5000, reload=True)
