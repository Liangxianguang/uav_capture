@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_bc_pybullet_gpu.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
