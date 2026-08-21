[CmdletBinding()]
param(
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$Python = "C:\Users\liangxianguang\.conda\envs\uav-encirclement-gpu\python.exe"

$FixedConfig = "configs\capture_radius_recurrent_behavior_cloning_central_v5_fixed_wallcoverage_seed661701.yaml"
$RetainedConfig = "configs\capture_radius_recurrent_behavior_cloning_central_v5_wallcoverage_retained_seed661702.yaml"
$FixedOutput = "results\central_v5\fixed_wallcoverage_seed661701"
$RetainedOutput = "results\central_v5\wallcoverage_retained_seed661702"

function Invoke-WallCoverageTraining {
    param(
        [string]$Config,
        [string]$Output
    )
    if (Test-Path -LiteralPath $Output) {
        throw "Refusing to overwrite existing P3-A output: $Output"
    }
    & $Python "scripts\train_capture_radius_recurrent_behavior_cloning.py" --config $Config --output $Output --device $Device
    if ($LASTEXITCODE -ne 0) {
        throw "P3-A training failed for $Config with exit code $LASTEXITCODE."
    }
}

Invoke-WallCoverageTraining -Config $FixedConfig -Output $FixedOutput
if (-not (Test-Path -LiteralPath (Join-Path $FixedOutput "checkpoint.pt") -PathType Leaf)) {
    throw "The P3-A fixed stage did not write checkpoint.pt."
}
if (-not (Test-Path -LiteralPath (Join-Path $FixedOutput "expert_sequence_dataset.npz") -PathType Leaf)) {
    throw "The P3-A fixed stage did not write its expert archive."
}
Invoke-WallCoverageTraining -Config $RetainedConfig -Output $RetainedOutput

Write-Output "V5 P3-A wall-coverage pilot training completed."
Write-Output "Fixed stage: $FixedOutput"
Write-Output "Retained stage: $RetainedOutput"
