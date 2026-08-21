[CmdletBinding()]
param(
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$Python = "C:\Users\liangxianguang\.conda\envs\uav-encirclement-gpu\python.exe"
$FixedOutput = "results\central_v5\fixed_shapeaware_seed661601"
$RetainedOutput = "results\central_v5\shapeaware_retained_seed661602"

function Invoke-RetainedTraining {
    param([string]$Config, [string]$Output)
    if (Test-Path -LiteralPath $Output) {
        throw "Refusing to overwrite existing output: $Output"
    }
    & $Python "scripts\train_capture_radius_recurrent_behavior_cloning.py" `
        --config $Config --output $Output --device $Device
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $Config with exit code $LASTEXITCODE."
    }
}

Invoke-RetainedTraining `
    "configs\capture_radius_recurrent_behavior_cloning_central_v5_fixed_shapeaware.yaml" `
    $FixedOutput

$FixedCheckpoint = Join-Path $FixedOutput "checkpoint.pt"
$FixedArchive = Join-Path $FixedOutput "expert_sequence_dataset.npz"
if (-not (Test-Path -LiteralPath $FixedCheckpoint -PathType Leaf) -or -not (Test-Path -LiteralPath $FixedArchive -PathType Leaf)) {
    throw "Fixed shape-aware stage did not produce a checkpoint and archive."
}

Invoke-RetainedTraining `
    "configs\capture_radius_recurrent_behavior_cloning_central_v5_shapeaware_retained.yaml" `
    $RetainedOutput

Write-Output "Shape-aware fixed stage and warm-start retained stage completed."
Write-Output "Fixed stage: $FixedOutput"
Write-Output "Retained stage: $RetainedOutput"
