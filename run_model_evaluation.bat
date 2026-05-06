@echo off
REM ============================================================================
REM InvestIQ Model Evaluation Suite - Main Runner
REM Comprehensive evaluation of all models and generation of publication graphs
REM ============================================================================

REM Activate Python environment
call backend\setup_env.bat

REM Run evaluation
echo.
echo ============================================================================
echo Starting InvestIQ Comprehensive Model Evaluation
echo ============================================================================
echo.

python backend/scripts/run_evaluation.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ============================================================================
    echo ERROR: Evaluation failed!
    echo ============================================================================
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo Evaluation Complete!
echo Results saved to: backend\models\saved_models\evaluation_results\
echo ============================================================================
echo.
echo Next steps:
echo 1. Open backend\models\saved_models\evaluation_results\ in File Explorer
echo 2. Copy PNG files to your paper
echo 3. Use CSV and report files for metrics
echo.
pause
