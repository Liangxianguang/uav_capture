[CmdletBinding()]
param(
    [string]$Config = "configs/dagger_pybullet_moderate_target03_full.yaml",
    [string]$Output = "",
    [int]$Seed = 801,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
    $Output = Join-Path $ProjectRoot "results\dagger_pybullet\${RunId}_seed${Seed}"
}
& conda run --no-capture-output --name uav-encirclement-gpu python scripts/train_dagger_pybullet.py --config $Config --output $Output --seed $Seed --device $Device
if ($LASTEXITCODE -ne 0) {
    throw "PyBullet DAgger training failed with exit code $LASTEXITCODE."
}
