@echo off
cd /d "%~dp0.."
call venv\Scripts\activate
echo Starting Stock Predictor API...
uvicorn backend.app.main:app --host 0.0.0.0 --port 5000 --reload
pause
