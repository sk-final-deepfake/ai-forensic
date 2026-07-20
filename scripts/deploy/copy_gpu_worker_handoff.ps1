#Requires -Version 5.1
<#
.SYNOPSIS
  GPU worker 작업자용 배포 안내를 생성하고 클립보드에 복사합니다.
.EXAMPLE
  .\scripts\deploy\copy_gpu_worker_handoff.ps1
.EXAMPLE
  .\scripts\deploy\copy_gpu_worker_handoff.ps1 -Branch fix/dynamic-weighted-risk-lane
#>
param(
    [string]$Branch = "",
    [ValidateSet("dynamic-risk", "fusion-v4c")]
    [string]$Topic = "dynamic-risk"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $repoRoot

$args = @("scripts/deploy/gpu_worker_handoff.py", "--copy", "--topic", $Topic)
if ($Branch) {
    $args += @("--branch", $Branch)
}

python @args
if ($LASTEXITCODE -ne 0) {
    throw "gpu_worker_handoff.py failed with exit code $LASTEXITCODE"
}

Write-Host ""
Write-Host "카톡에 Ctrl+V 로 붙여넣으면 됩니다." -ForegroundColor Green
