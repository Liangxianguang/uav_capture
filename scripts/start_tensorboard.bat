@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_tensorboard.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
