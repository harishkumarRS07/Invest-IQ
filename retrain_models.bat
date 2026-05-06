@echo off
REM Retrain models for evaluation
REM Run this to fix the feature mismatch issues

setlocal enabledelayedexpansion

echo.
echo ================================================================================
echo  RETRAINING MODELS FOR EVALUATION
echo ================================================================================
echo.
echo This will retrain LSTM and Transformer models with the current 21-feature pipeline
echo Duration: ~10-15 minutes depending on your system
echo.

REM Activate virtual environment
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Error: Virtual environment not found at venv\Scripts\activate.bat
    exit /b 1
)

REM Run retraining script
echo Starting retraining...
python backend/scripts/retrain_for_evaluation.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo  RETRAINING COMPLETE!
    echo ================================================================================
    echo.
    echo Next step: Run comprehensive evaluation
    echo   python backend/scripts/comprehensive_model_evaluation.py
    echo.
) else (
    echo.
    echo ERROR: Retraining failed!
    exit /b 1
)

pause
