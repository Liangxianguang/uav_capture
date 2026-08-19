@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0evaluate_capture_radius_mappo.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
