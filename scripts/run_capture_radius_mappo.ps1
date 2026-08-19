[CmdletBinding()]
param(
    [string]$Config = "configs/capture_radius_mappo_dev.yaml",
    [string]$Output = "",
    [int]$Seed = 521001,
    [int]$TotalSteps = 0,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$InitializeFrom = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
    $Output = Join-Path $ProjectRoot "results\capture_radius_mappo_$($RunId)_seed$($Seed)"
}
$Arguments = @(
    "--no-capture-output", "--name", "uav-encirclement-gpu", "python",
    "scripts/train_capture_radius_mappo.py", "--config", $Config,
    "--output", $Output, "--seed", $Seed, "--device", $Device
)
if ($TotalSteps -gt 0) {
    $Arguments += @("--total-steps", $TotalSteps)
}
if (-not [string]::IsNullOrWhiteSpace($InitializeFrom)) {
    $Arguments += @("--initialize-from", $InitializeFrom)
}
& conda run @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Capture-radius MAPPO training failed with exit code $LASTEXITCODE."
}
