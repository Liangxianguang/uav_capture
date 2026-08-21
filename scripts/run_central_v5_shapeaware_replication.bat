@echo off
setlocal

set "ROOT=%~dp0.."
pushd "%ROOT%"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_central_v5_shapeaware_replication.ps1 -Device cuda -Replica %*
set "EXITCODE=%ERRORLEVEL%"
popd
exit /b %EXITCODE%
