@echo off
setlocal enabledelayedexpansion

echo.
echo ============================================================
echo    INVESTIQ MODEL TRAINING - ENHANCED VERSION
echo ============================================================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo [INFO] Virtual environment not found. Creating...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
)

REM Activate venv
echo [INFO] Activating Python environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Check if requirements are installed
pip show torch >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies ^(first time setup^)...
    pip install -r backend\requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies already installed
)

REM Set Python path
set PYTHONPATH=%cd%

REM Run enhanced training
echo.
echo [INFO] Starting training pipeline...
python train_enhanced.py

if errorlevel 1 (
    echo.
    echo [ERROR] Training failed. Check output above for details.
    pause
    exit /b 1
) else (
    echo.
    echo [OK] Training completed successfully!
    pause
    exit /b 0
)
