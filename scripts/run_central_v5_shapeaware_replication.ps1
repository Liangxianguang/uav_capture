[CmdletBinding()]
param(
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda",

    [ValidateSet("seed661604", "seed661606")]
    [string]$Replica = "seed661604"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$Python = "C:\Users\liangxianguang\.conda\envs\uav-encirclement-gpu\python.exe"

$Runs = @{
    "seed661604" = @{
        FixedConfig = "configs\capture_radius_recurrent_behavior_cloning_central_v5_fixed_shapeaware_seed661603.yaml"
        RetainedConfig = "configs\capture_radius_recurrent_behavior_cloning_central_v5_shapeaware_retained_seed661604.yaml"
        FixedOutput = "results\central_v5\fixed_shapeaware_seed661603"
        RetainedOutput = "results\central_v5\shapeaware_retained_seed661604"
    }
    "seed661606" = @{
        FixedConfig = "configs\capture_radius_recurrent_behavior_cloning_central_v5_fixed_shapeaware_seed661605.yaml"
        RetainedConfig = "configs\capture_radius_recurrent_behavior_cloning_central_v5_shapeaware_retained_seed661606.yaml"
        FixedOutput = "results\central_v5\fixed_shapeaware_seed661605"
        RetainedOutput = "results\central_v5\shapeaware_retained_seed661606"
    }
}

$Run = $Runs[$Replica]

function Invoke-ReplicaTraining {
    param(
        [string]$Config,
        [string]$Output
    )
    if (Test-Path -LiteralPath $Output) {
        throw "Refusing to overwrite existing output: $Output"
    }
    & $Python "scripts\train_capture_radius_recurrent_behavior_cloning.py" --config $Config --output $Output --device $Device
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed for $Config with exit code $LASTEXITCODE."
    }
}

Invoke-ReplicaTraining -Config $Run.FixedConfig -Output $Run.FixedOutput

$FixedCheckpoint = Join-Path $Run.FixedOutput "checkpoint.pt"
$FixedArchive = Join-Path $Run.FixedOutput "expert_sequence_dataset.npz"
if (-not (Test-Path -LiteralPath $FixedCheckpoint -PathType Leaf) -or -not (Test-Path -LiteralPath $FixedArchive -PathType Leaf)) {
    throw "Fixed stage did not produce a checkpoint and expert archive."
}

Invoke-ReplicaTraining -Config $Run.RetainedConfig -Output $Run.RetainedOutput

Write-Output "Frozen V5 shape-aware replication completed: $Replica"
Write-Output "Fixed stage: $($Run.FixedOutput)"
Write-Output "Retained stage: $($Run.RetainedOutput)"
