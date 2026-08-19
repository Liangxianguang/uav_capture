[CmdletBinding()]
param(
    [string]$Config = "configs/capture_radius_recurrent_mappo_gru_prediction_pilot.yaml",
    [string]$Output = "",
    [int]$Seed = 521001,
    [int]$TotalSteps = 0,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$InitializeFrom = "",
    [string]$PredictionCheckpoint = "",
    [int]$SequenceLength = 32,
    [int]$PredictionHistoryLength = 8,
    [int]$PredictionHorizonIndex = 2
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
    $Output = Join-Path $ProjectRoot "results\capture_radius_recurrent_mappo_$($RunId)_seed$($Seed)"
}
$Arguments = @(
    "--no-capture-output", "--name", "uav-encirclement-gpu", "python",
    "scripts/train_capture_radius_recurrent_mappo.py", "--config", $Config,
    "--output", $Output, "--seed", $Seed, "--device", $Device,
    "--sequence-length", $SequenceLength
)
if ($TotalSteps -gt 0) { $Arguments += @("--total-steps", $TotalSteps) }
if (-not [string]::IsNullOrWhiteSpace($InitializeFrom)) { $Arguments += @("--initialize-from", $InitializeFrom) }
if (-not [string]::IsNullOrWhiteSpace($PredictionCheckpoint)) {
    $Arguments += @(
        "--prediction-checkpoint", $PredictionCheckpoint,
        "--prediction-history-length", $PredictionHistoryLength,
        "--prediction-horizon-index", $PredictionHorizonIndex
    )
}
& conda run @Arguments
if ($LASTEXITCODE -ne 0) { throw "Recurrent MAPPO training failed with exit code $LASTEXITCODE." }
