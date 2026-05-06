@echo off
REM ============================================================================
REM InvestIQ Model Evaluation - Generate Detailed Prediction Plots (Optional)
REM Creates prediction vs actual visualizations for each model and ticker
REM ============================================================================

REM Activate Python environment
call backend\setup_env.bat

REM Run prediction plot generation
echo.
echo ============================================================================
echo Generating Detailed Prediction Plots
echo ============================================================================
echo.

python backend/scripts/generate_prediction_plots.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo ============================================================================
    echo ERROR: Prediction plot generation failed!
    echo ============================================================================
    pause
    exit /b 1
)

echo.
echo ============================================================================
echo Prediction Plots Generated!
echo Results saved to: backend\models\saved_models\evaluation_results\prediction_visualizations\
echo ============================================================================
echo.
echo These additional plots show:
echo - LSTM predictions vs actual over time
echo - Transformer 7-day forecasts
echo - Residual analysis (errors and distributions)
echo.
pause
