@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_central_v5_baseline_evaluation.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
