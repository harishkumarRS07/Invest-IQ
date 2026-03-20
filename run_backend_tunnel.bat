@echo off
cd /d "%~dp0"
echo Starting backend tunnel for InvestIQ API...
echo URL: https://curly-friends-strive.loca.lt
npx localtunnel --port 5000 --host https://localtunnel.me --subdomain curly-friends-strive
