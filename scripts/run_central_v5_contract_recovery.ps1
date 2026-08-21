[CmdletBinding()]
param(
    [string]$FixedOutput = "results\central_v5\fixed_contract_archive_seed661501",
    [string]$RecoveryOutput = "results\central_v5\contract_recovery_seed661502",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$Python = "C:\Users\liangxianguang\.conda\envs\uav-encirclement-gpu\python.exe"

function Invoke-Training {
    param([string]$Config, [string]$Output)
    if (Test-Path -LiteralPath $Output) {
        throw "Refusing to overwrite existing recovery output: $Output"
    }
    & $Python "scripts\train_capture_radius_recurrent_behavior_cloning.py" `
        --config $Config --output $Output --device $Device
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $Config with exit code $LASTEXITCODE."
    }
}

Invoke-Training `
    "configs\capture_radius_recurrent_behavior_cloning_central_v5_fixed_contract_collection.yaml" `
    $FixedOutput

if (-not (Test-Path -LiteralPath (Join-Path $FixedOutput "expert_sequence_dataset.npz") -PathType Leaf)) {
    throw "Fixed contract archive was not produced: $FixedOutput"
}

Invoke-Training `
    "configs\capture_radius_recurrent_behavior_cloning_central_v5_contract_recovery.yaml" `
    $RecoveryOutput

Write-Output "Fixed archive and contract-recovery checkpoint completed."
Write-Output "Fixed archive: $FixedOutput"
Write-Output "Recovery checkpoint: $RecoveryOutput"
