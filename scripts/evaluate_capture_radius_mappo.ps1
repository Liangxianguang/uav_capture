[CmdletBinding()]
param(
    [string]$Config = "configs/capture_radius_pursuit_dev.yaml",
    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,
    [string]$Output = "",
    [int]$Seed = 610001,
    [int]$Episodes = 100,
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cpu",
    [switch]$UseCbf
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
    $Output = Join-Path $ProjectRoot "results\capture_radius_mappo_eval_$($RunId)_seed$($Seed)"
}
$Arguments = @(
    "--no-capture-output", "--name", "uav-encirclement-gpu", "python",
    "scripts/evaluate_capture_radius_mappo.py", "--config", $Config,
    "--checkpoint", $Checkpoint, "--output", $Output, "--seed", $Seed,
    "--episodes", $Episodes, "--device", $Device
)
if ($UseCbf) {
    $Arguments += "--use-cbf"
}
& conda run @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Capture-radius MAPPO evaluation failed with exit code $LASTEXITCODE."
}
