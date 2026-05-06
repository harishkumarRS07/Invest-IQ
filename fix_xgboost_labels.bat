@echo off
REM ============================================================================
REM XGBoost Label Imbalance Fix - Complete Retraining Script
REM ============================================================================
REM
REM This batch file executes the complete fix for label imbalance:
REM 1. Verifies label distribution after threshold change
REM 2. Retrains all XGBoost models with 0.002 threshold (0.2%)
REM 3. Validates class distribution for each stock
REM 4. Generates predictions to verify diverse signals
REM
REM Expected Output:
REM - BUY signals are generated
REM - SELL signals are generated
REM - HOLD is not 100%
REM ============================================================================

echo.
echo ============================================================================
echo  XGBoost Label Fix - Complete Retraining
echo ============================================================================
echo.

REM Activate virtual environment
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo Error: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Set Python path
set PYTHONPATH=%cd%

REM Step 1: Show threshold impact
echo [Step 1/3] Showing threshold impact...
echo.
python verify_threshold_fix.py
if %errorlevel% neq 0 (
    echo Error: Threshold verification failed
    pause
    exit /b 1
)

echo.
echo [Step 2/3] Running complete fix script...
echo.
python fix_label_imbalance.py
if %errorlevel% neq 0 (
    echo Error: Label imbalance fix failed
    pause
    exit /b 1
)

echo.
echo [Step 3/3] Verifying models...
echo.

REM Quick test with demo
python backend/scripts/demo.py

echo.
echo ============================================================================
echo  All Steps Complete!
echo ============================================================================
echo.
echo Next Steps:
echo 1. Check output above for BUY/SELL/HOLD signal diversity
echo 2. If signals are still only HOLD, run: python verify_threshold_fix.py
echo 3. Start API: python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
echo.
pause
