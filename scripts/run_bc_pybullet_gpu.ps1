[CmdletBinding()]
param(
    [string]$Config = "configs/bc_pybullet_target03.yaml",
    [string]$Output = "",
    [int]$Seed = 404,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
    $Output = Join-Path $ProjectRoot "results\bc_pybullet\${RunId}_seed${Seed}"
}
& conda run --no-capture-output --name uav-encirclement-gpu python scripts/train_behavior_cloning.py --config $Config --output $Output --seed $Seed --device $Device
if ($LASTEXITCODE -ne 0) {
    throw "PyBullet behavior-cloning training failed with exit code $LASTEXITCODE."
}
