@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_capture_radius_recurrent_behavior_cloning.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
