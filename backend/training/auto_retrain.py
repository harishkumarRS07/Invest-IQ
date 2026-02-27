"""
Weekly Auto-Retrain Orchestrator - InvestIQ
Runs every Sunday at midnight to retrain all models on fresh accumulated data.
"""
import os
import glob
import threading
from backend.core.config import settings
from backend.core.logging import logger
from backend.data.update_stock_data import refined_update
from backend.training.train import train_pipeline


# ── Internal state ────────────────────────────────────────────────────────────
_retrain_status: dict = {
    "is_running": False,
    "last_run": None,       # ISO timestamp string
    "last_status": "never", # "success" | "failed" | "running" | "never"
    "log": [],              # list of log lines for the last run
}


def get_retrain_status() -> dict:
    """Return a copy of the current retraining status (used by the /retrain/status route)."""
    return dict(_retrain_status)


def _run_full_retrain():
    """
    Internal function executed in a background thread.
    Steps:
      1. Pull latest price data for all tickers.
      2. Retrain Transformer model for each ticker.
      3. Clear the prediction cache so fresh models are used.
    """
    global _retrain_status
    from datetime import datetime

    _retrain_status["is_running"] = True
    _retrain_status["last_status"] = "running"
    _retrain_status["log"] = []

    def _log(msg: str):
        logger.info(msg)
        _retrain_status["log"].append(msg)

    try:
        _log("═══ Weekly Auto-Retrain Started ═══")

        # Step 1 – Update stock data CSVs
        _log("Step 1/2: Updating stock data from Yahoo Finance…")
        try:
            refined_update()
            _log("  ✓ Stock data updated successfully.")
        except Exception as e:
            _log(f"  ⚠ Data update failed (continuing with existing data): {e}")

        # Step 2 – Retrain model for each CSV
        _log("Step 2/2: Retraining models for all tickers…")
        csv_files = sorted(glob.glob(os.path.join(settings.DATA_DIR, "*.csv")))
        # Exclude .NS files to avoid duplicate training (use the primary CSV)
        csv_files = [f for f in csv_files if ".NS" not in os.path.basename(f)]

        if not csv_files:
            _log("  ✗ No CSV files found – aborting retrain.")
            _retrain_status["last_status"] = "failed"
            return

        for file_path in csv_files:
            ticker = os.path.basename(file_path).replace(".csv", "")
            _log(f"  Training {ticker}…")
            try:
                train_pipeline(file_path)
                _log(f"  ✓ {ticker} retrained successfully.")
            except Exception as e:
                _log(f"  ✗ {ticker} training failed: {e}")

        # Step 3 – Bust prediction cache so new models are used immediately
        try:
            from backend.app.routes import _signals_cache
            _signals_cache.clear()
            _log("  ✓ Prediction cache cleared – new models active.")
        except Exception:
            pass  # cache may not exist yet

        _log("═══ Weekly Auto-Retrain Completed ═══")
        _retrain_status["last_status"] = "success"

    except Exception as e:
        _log(f"Retrain failed with unexpected error: {e}")
        _retrain_status["last_status"] = "failed"
    finally:
        from datetime import datetime
        _retrain_status["is_running"] = False
        _retrain_status["last_run"] = datetime.now().isoformat()


def trigger_retrain(background: bool = True):
    """
    Public entry point:
      - Called by APScheduler every Sunday at 00:00.
      - Can also be called manually via the /retrain/trigger API endpoint.
    """
    if _retrain_status["is_running"]:
        logger.warning("Retrain already in progress – skipping trigger.")
        return False

    if background:
        t = threading.Thread(target=_run_full_retrain, daemon=True, name="weekly-retrain")
        t.start()
        logger.info("Weekly retrain thread started.")
    else:
        _run_full_retrain()  # blocking (used in tests)

    return True
