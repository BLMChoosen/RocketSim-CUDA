# scripts/build_and_test.ps1 - Automated compile, differential harness and pytest execution
$ErrorActionPreference = "Stop"

$vcvars = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

Write-Host "=== [1/3] Building C++/CUDA, Harness and Python extension ===" -ForegroundColor Cyan
cmd.exe /c "call `"$vcvars`" && cmake --build build --config Release -j"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`n=== [2/3] Running Differential Harness (--scenario all --check) ===" -ForegroundColor Cyan
& ".\build\differential_harness.exe" --scenario all --check docs/parity_thresholds.json
if ($LASTEXITCODE -ne 0) {
    Write-Error "Differential harness check failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`n=== [3/3] Running Python Pytest Suite ===" -ForegroundColor Cyan
pytest tests/python -q
if ($LASTEXITCODE -ne 0) {
    Write-Error "Pytest suite failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

Write-Host "`n=== ALL CHECKS PASSED (Build, Harness --check, Pytest) ===" -ForegroundColor Green
exit 0
