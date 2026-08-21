@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_central_v5_wallcoverage_pilot.ps1" -Device cuda
exit /b %ERRORLEVEL%
