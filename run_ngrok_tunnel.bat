@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set NGROK_AUTH_TOKEN=371G5qsM4syzZVEFLK4Hl3vyDDi_3dzgfP6gk2Feqx2ujqvtk
set BACKEND_PORT=8000

echo ============================================
echo Starting ngrok tunnel for InvestIQ Backend
echo ============================================
echo.
echo Authenticating with ngrok...
call npx ngrok config add-authtoken %NGROK_AUTH_TOKEN%

echo.
echo [%date% %time%] Connecting ngrok tunnel...
echo Exposing: http://localhost:%BACKEND_PORT%
echo.
echo Keep this terminal OPEN while testing!
echo The public URL will appear below and update on reconnect.
echo.

:restart_tunnel
npx ngrok http %BACKEND_PORT% --log=stdout

echo.
echo [%date% %time%] Tunnel disconnected. Reconnecting in 5 seconds...
timeout /t 5 /nobreak >nul

goto restart_tunnel
