[CmdletBinding()]
param(
    [string]$CaseId = "BRATS_224",
    [switch]$IncludeExpensiveNeqrAer
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$required = @(
    "artifacts\msd_rf_240.joblib",
    "outputs\reports\msd_patient_split.json",
    "data\raw\Task01_BrainTumour\dataset.json",
    "scripts\run_highres_quantum.py"
)
foreach ($item in $required) {
    if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot $item))) {
        throw "Required file is missing: $item. Apply the V3.3 update and preserve your calibrated model, split and dataset."
    }
}

$split = Get-Content (Join-Path $ProjectRoot "outputs\reports\msd_patient_split.json") -Raw | ConvertFrom-Json
if ($CaseId -notin $split.test) {
    throw "$CaseId is not in the saved held-out test partition. Choose a patient from outputs\reports\msd_patient_split.json -> test."
}

$logDir = Join-Path $ProjectRoot "outputs\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir ("highres_demo_suite_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Project virtual environment not found at $PythonExe. Activate/create .venv and install requirements first."
}
"PROJECT QUANTUM V3.3 HIGH-RESOLUTION DEMO SUITE" | Set-Content -LiteralPath $log -Encoding UTF8
"Case: $CaseId" | Add-Content -LiteralPath $log -Encoding UTF8
"Project: $ProjectRoot" | Add-Content -LiteralPath $log -Encoding UTF8

$commands = @()
foreach ($resolution in @(64, 128, 240)) {
    foreach ($representation in @("FRQI", "NEQR")) {
        $commands += [pscustomobject]@{
            Label = "$resolution x $resolution $representation exact patchwise statevector + native RF"
            Args = @("scripts/run_highres_quantum.py", "--case-id", $CaseId, "--resolution", "$resolution", "--representation", $representation, "--mode", "statevector")
        }
    }
}
$commands += [pscustomobject]@{
    Label = "Native 240 x 240 FRQI full-grid Aer ideal (64 sequential tile circuits)"
    Args = @("scripts/run_highres_quantum.py", "--case-id", $CaseId, "--resolution", "240", "--representation", "FRQI", "--mode", "aer_ideal_full")
}
$commands += [pscustomobject]@{
    Label = "Native 240 x 240 FRQI noisy Aer one-tile spot-check"
    Args = @("scripts/run_highres_quantum.py", "--case-id", $CaseId, "--resolution", "240", "--representation", "FRQI", "--mode", "aer_noisy_spotcheck", "--max-aer-tiles", "1", "--shots", "50000")
}
if ($IncludeExpensiveNeqrAer) {
    $commands += [pscustomobject]@{
        Label = "Native 240 x 240 NEQR ideal Aer one-tile spot-check (18-qubit opt-in)"
        Args = @("scripts/run_highres_quantum.py", "--case-id", $CaseId, "--resolution", "240", "--representation", "NEQR", "--mode", "aer_ideal_spotcheck", "--max-aer-tiles", "1", "--allow-expensive-18-qubit")
    }
}

$index = 0
foreach ($command in $commands) {
    $index++
    Write-Host "[$index/$($commands.Count)] $($command.Label)" -ForegroundColor Cyan
    "`n===== [$index/$($commands.Count)] $($command.Label) =====" | Add-Content -LiteralPath $log -Encoding UTF8
    $cmdArgs = $command.Args
    & $PythonExe @cmdArgs 2>&1 | Tee-Object -FilePath $log -Append
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "High-resolution experiment failed with exit code $exitCode. See full log: $log"
    }
}

Write-Host ""
Write-Host "HIGH-RESOLUTION DEMO SUITE COMPLETED" -ForegroundColor Green
Write-Host "JSON/NPZ outputs: outputs\reports\quantum\highres"
Write-Host "Run log: $log"
Write-Host "Figures can be exported with: python scripts/export_research_figures.py --resolutions all"
