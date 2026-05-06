@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set LT_PORT=8000

echo ============================================
echo Starting backend tunnel for InvestIQ API
echo Forwarding to local port: %LT_PORT%
echo ============================================
echo.

:restart_tunnel
echo [%date% %time%] Connecting tunnel...
echo Keep this terminal OPEN while testing!
echo.
npx localtunnel --port %LT_PORT% || (
    echo.
    echo [%date% %time%] Connection failed. Retrying in 5 seconds...
    timeout /t 5 /nobreak >nul
)

goto restart_tunnel
