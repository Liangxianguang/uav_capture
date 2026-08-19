[CmdletBinding()]
param(
    [string]$Config = "configs/pybullet_dagger_eval_target03_independent.yaml",
    [string]$Output = "results/pybullet_dagger_target03_independent_gpu_seed240001",
    [int]$Seed = 240001,
    [int]$Episodes = 30,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
& conda run --no-capture-output --name uav-encirclement-gpu python scripts/evaluate_pybullet.py --config $Config --output $Output --seed $Seed --episodes $Episodes --device $Device
if ($LASTEXITCODE -ne 0) {
    throw "PyBullet DAgger evaluation failed with exit code $LASTEXITCODE."
}
