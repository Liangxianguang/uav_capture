@echo off
setlocal
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_central_v5_contract_recovery.ps1 %*
if errorlevel 1 exit /b %errorlevel%
endlocal
