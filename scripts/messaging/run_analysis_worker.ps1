# Start AI RabbitMQ worker (Option A)
# Usage: powershell -ExecutionPolicy Bypass -File .\scripts\messaging\run_analysis_worker.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent)

if (Test-Path ".\.env") {
    Get-Content ".\.env" | ForEach-Object {
        if ($_ -match '^(?!#)(.+?)=(.*)$') {
            Set-Item -Path "env:$($matches[1])" -Value $matches[2]
        }
    }
}

$python = "..\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    $python = "python"
}

Write-Host "Starting AI analysis worker (RabbitMQ) ..."
& $python -m app.workers.run_analysis_worker
