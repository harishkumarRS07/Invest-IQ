@echo off
echo Starting InvestIQ AI Model Training...
echo This will fetch data and train deep learning models for HDFCBANK, RELIANCE, TCS, INFY, ICICIBANK.
echo Please wait...
call venv\Scripts\activate
set PYTHONPATH=%cd%
python backend\scripts\train_all.py
pause
