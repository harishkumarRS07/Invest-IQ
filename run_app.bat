@echo off
echo Starting Stock Predictor API...
call venv\Scripts\activate
uvicorn backend.app.main:app --host 0.0.0.0 --port 5000 --reload
pause
