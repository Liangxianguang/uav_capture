[CmdletBinding()]
param(
    [string]$LogDir = "results",
    [int]$Port = 6006
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
& conda run --no-capture-output --name uav-encirclement-gpu tensorboard --logdir $LogDir --port $Port
