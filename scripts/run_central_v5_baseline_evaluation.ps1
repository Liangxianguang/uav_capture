[CmdletBinding()]
param(
    [string]$Checkpoint = "results\central_v5\bc_baseline_seed661401\checkpoint.pt",
    [string]$RunDir = "results\central_v5\bc_baseline_seed661401",
    [string]$EvaluationRoot = "results\central_v5",
    [string]$RunId = "bc_baseline_seed661401",
    [ValidateSet("auto", "cuda", "cpu")]
    [string]$Device = "cuda",
    [string]$OutputJson = "CENTRAL_V5_BASELINE_VALIDATION_SUMMARY.json",
    [string]$OutputMarkdown = "CENTRAL_V5_BASELINE_VALIDATION_REPORT.md"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
$Python = "C:\Users\liangxianguang\.conda\envs\uav-encirclement-gpu\python.exe"

function Invoke-V5Python {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "V5 evaluation command failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "Checkpoint does not exist: $Checkpoint"
}
if (-not (Test-Path -LiteralPath $RunDir -PathType Container)) {
    throw "Training run directory does not exist: $RunDir"
}

$FixedScenes = @(
    @{ Name = "s1_cylinder"; Scenario = "s1"; Layout = "cylinder" },
    @{ Name = "s1_box"; Scenario = "s1"; Layout = "box" },
    @{ Name = "s1_wall"; Scenario = "s1"; Layout = "wall" },
    @{ Name = "s2"; Scenario = "v4_s2"; Layout = "mixed" }
)

foreach ($Scene in $FixedScenes) {
    foreach ($Mode in @("raw", "cbf")) {
        $Output = Join-Path $EvaluationRoot "$RunId`_$($Scene.Name)_$Mode`_20"
        if (Test-Path -LiteralPath $Output) {
            throw "Refusing to overwrite existing fixed-regression artifact: $Output"
        }
        $Arguments = @(
            "scripts\evaluate_mixed_obstacle_showcase.py",
            "--method", "f2",
            "--checkpoint", $Checkpoint,
            "--output-dir", $Output,
            "--seed", "660501",
            "--episodes", "20",
            "--scenario", $Scene.Scenario,
            "--layout", $Scene.Layout,
            "--protocol-config", "configs\central_bidirectional_v4.yaml",
            "--device", $Device
        )
        if ($Mode -eq "cbf") {
            $Arguments += "--use-cbf"
        }
        Invoke-V5Python $Arguments
    }
}

foreach ($Mode in @("raw", "cbf")) {
    $Output = Join-Path $EvaluationRoot "$RunId`_s3_validation_$Mode`_60"
    if (Test-Path -LiteralPath $Output) {
        throw "Refusing to overwrite existing S3 validation artifact: $Output"
    }
    $Arguments = @(
        "scripts\evaluate_random_central_mixed_obstacles.py",
        "--method", "f2",
        "--checkpoint", $Checkpoint,
        "--protocol", "configs\central_random_mixed_obstacle_s3_v5_protocol.yaml",
        "--environment-config", "configs\capture_radius_pursuit_central_v4_flee.yaml",
        "--split", "validation",
        "--episodes", "60",
        "--output-dir", $Output,
        "--device", $Device
    )
    if ($Mode -eq "cbf") {
        $Arguments += "--use-cbf"
    }
    Invoke-V5Python $Arguments
    Invoke-V5Python @(
        "scripts\build_s3_failure_index.py",
        "--episodes-csv", (Join-Path $Output "episodes.csv"),
        "--output-json", (Join-Path $Output "failure_index.json"),
        "--output-md", (Join-Path $Output "failure_analysis.md")
    )
}

Invoke-V5Python @(
    "scripts\aggregate_central_v5_baseline.py",
    "--run-dir", $RunDir,
    "--evaluation-root", $EvaluationRoot,
    "--run-id", $RunId,
    "--output-json", $OutputJson,
    "--output-md", $OutputMarkdown
)
