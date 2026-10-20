param(
    [string]$Data = "data/processed/ashare_etf.csv",
    [string]$RawData = "data/processed/ashare_etf_unadjusted.csv",
    [string]$Suffix = "v2",
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$ProjectPath = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectPath
$PythonPath = Join-Path $ProjectPath ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $PythonPath)) { throw "Run scripts/bootstrap.ps1 first." }
$TrainArguments = @("--seeds", "7", "17", "29", "--algorithms", "td3", "ddpg", "sac", "ppo", "a2c", "--risk-levels", "2")
if ($Resume) { $TrainArguments += "--resume" }
& $PythonPath -m rlasset.cli train --data $Data --output "runs/comparison_$Suffix" --steps 10000 @TrainArguments
if ($LASTEXITCODE -ne 0) { throw "Main experiment failed." }
& $PythonPath -m rlasset.cli walk-forward --data $Data --output "runs/walkforward_$Suffix" --steps 5000 --folds 3 @TrainArguments
if ($LASTEXITCODE -ne 0) { throw "Walk-forward experiment failed." }
foreach ($RiskValue in @(0, 8)) {
    $RiskArguments = @()
    if ($Resume) { $RiskArguments += "--resume" }
    & $PythonPath -m rlasset.cli train --data $Data --output "runs/risk$($RiskValue)_$Suffix" --steps 10000 --seeds 7 17 29 --algorithms td3 --risk-levels $RiskValue @RiskArguments
    if ($LASTEXITCODE -ne 0) { throw "Risk ablation failed." }
}
foreach ($SeedValue in @(7, 17, 29)) {
    foreach ($CostValue in @(10, 20)) {
        & $PythonPath -m rlasset.cli evaluate --run "runs/comparison_$Suffix" --seed $SeedValue --cost-bps $CostValue
        if ($LASTEXITCODE -ne 0) { throw "Cost stress failed." }
    }
}
if (-not (Test-Path -LiteralPath $RawData)) {
    & $PythonPath -m rlasset.execution_data --output $RawData
    if ($LASTEXITCODE -ne 0) { throw "Raw-price download failed." }
}
& $PythonPath -m rlasset.paper --run "runs/comparison_$Suffix" --raw-data $RawData --output "runs/paper_$Suffix" --seeds 7 17 29
if ($LASTEXITCODE -ne 0) { throw "Paper replay failed." }
& $PythonPath -m rlasset.research_report --run "runs/comparison_$Suffix" --walk-forward "runs/walkforward_$Suffix" --output "runs/research_$Suffix"
if ($LASTEXITCODE -ne 0) { throw "Research report failed." }
& $PythonPath -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "Verification failed." }
