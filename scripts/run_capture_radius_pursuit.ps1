[CmdletBinding()]
param(
    [string]$Config = "configs/capture_radius_pursuit_dev.yaml",
    [string]$Output = "",
    [ValidateSet("pure", "prediction", "encirclement", "pure_cbf", "prediction_cbf", "encirclement_cbf")]
    [string]$Controller = "encirclement_cbf",
    [int]$Episodes = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
    $Output = Join-Path $ProjectRoot "results\capture_radius_$($Controller)_$($RunId)"
}
if (Test-Path $Output) {
    $Existing = Get-ChildItem -LiteralPath $Output -Force -ErrorAction SilentlyContinue
    if ($Existing) {
        throw "Refusing to overwrite a non-empty output directory: $Output"
    }
}

$Arguments = @(
    "--no-capture-output", "--name", "uav-encirclement-gpu", "python",
    "scripts/run_capture_radius_pursuit.py", "--config", $Config,
    "--output", $Output, "--controller", $Controller
)
if ($Episodes -gt 0) {
    $Arguments += @("--episodes", $Episodes)
}
& conda run @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Capture-radius baseline run failed with exit code $LASTEXITCODE."
}
