@echo off
echo Running InvestIQ AI Financial Platform Demo...
call venv\Scripts\activate
set PYTHONPATH=%cd%
python backend\scripts\demo.py
pause
