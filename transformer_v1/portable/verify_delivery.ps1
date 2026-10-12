$ErrorActionPreference = "Stop"

$portableDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $portableDir
$modelPath = Join-Path $portableDir "artifacts\coop_tracker_v7.onnx"
$caseDir = Join-Path $portableDir "artifacts\golden_case"
$cVerifier = Join-Path $portableDir "c\build\Release\verify_golden_case.exe"
$matlabDir = Join-Path $portableDir "matlab"

if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "Missing model: $modelPath"
}
if (-not (Test-Path -LiteralPath $cVerifier)) {
    throw "Missing C verifier: $cVerifier"
}

Write-Host "[1/2] Verifying C entry point"
& $cVerifier $modelPath $caseDir
if ($LASTEXITCODE -ne 0) {
    throw "C golden-case verification failed with exit code $LASTEXITCODE"
}

$matlab = Get-Command matlab.exe -ErrorAction SilentlyContinue
if ($null -eq $matlab) {
    throw "matlab.exe was not found on PATH"
}

Write-Host "[2/2] Verifying MATLAB entry point"
$matlabCommand = "cd('$($projectDir.Replace("'", "''"))'); " +
    "addpath('$($matlabDir.Replace("'", "''"))'); " +
    "verify_golden_case('$($modelPath.Replace("'", "''"))'," +
    "'$($caseDir.Replace("'", "''"))');"
& $matlab.Source -batch $matlabCommand
if ($LASTEXITCODE -ne 0) {
    throw "MATLAB golden-case verification failed with exit code $LASTEXITCODE"
}

Write-Host "C and MATLAB delivery verification passed."
