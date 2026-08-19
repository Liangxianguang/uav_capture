[CmdletBinding()]
param(
    [string]$Config = "configs/pybullet_bc_eval.yaml",
    [string]$Output = "",
    [int]$Seed = 210001,
    [int]$Episodes = 0,
    [double]$TargetSpeedScale = 0.0,
    [string]$TraceCsv = "",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
    $Output = Join-Path $ProjectRoot "results\pybullet_bc\${RunId}_seed${Seed}"
}
$Arguments = @("scripts/evaluate_pybullet.py", "--config", $Config, "--output", $Output, "--seed", $Seed, "--device", $Device)
if ($Episodes -gt 0) {
    $Arguments += @("--episodes", $Episodes)
}
if ($TargetSpeedScale -gt 0.0) {
    $Arguments += @("--target-speed-scale", $TargetSpeedScale)
}
if (-not [string]::IsNullOrWhiteSpace($TraceCsv)) {
    $Arguments += @("--trace-csv", $TraceCsv)
}
& conda run --no-capture-output --name uav-encirclement-gpu python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "PyBullet policy evaluation failed with exit code $LASTEXITCODE."
}
