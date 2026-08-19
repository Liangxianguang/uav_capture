[CmdletBinding()]
param(
    [string]$Config = "configs/capture_radius_behavior_cloning_dev.yaml",
    [string]$Output = "",
    [int]$Seed = 521001,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$PredictionCheckpoint = "",
    [int]$PredictionHistoryLength = 8,
    [int]$PredictionHorizonIndex = 2
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
    $Output = Join-Path $ProjectRoot "results\capture_radius_behavior_cloning_$($RunId)_seed$($Seed)"
}
$Arguments = @(
    "--no-capture-output", "--name", "uav-encirclement-gpu", "python",
    "scripts/train_capture_radius_behavior_cloning.py", "--config", $Config,
    "--output", $Output, "--seed", $Seed, "--device", $Device
)
if (-not [string]::IsNullOrWhiteSpace($PredictionCheckpoint)) {
    $Arguments += @(
        "--prediction-checkpoint", $PredictionCheckpoint,
        "--prediction-history-length", $PredictionHistoryLength,
        "--prediction-horizon-index", $PredictionHorizonIndex
    )
}
& conda run @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Capture-radius behavior cloning failed with exit code $LASTEXITCODE."
}
