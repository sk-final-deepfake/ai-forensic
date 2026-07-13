$ErrorActionPreference = "Stop"
$Remote = "sk4team@58.151.205.220"
$LocalRoot = "c:\FINAL\ai-forensic\data\pull\evidence"
New-Item -ItemType Directory -Force -Path $LocalRoot | Out-Null

foreach ($name in @("mvtamperbench-200-s3", "csvted-200-balanced")) {
    $localTar = Join-Path $env:TEMP "$name.tar.gz"
    Write-Host "`n=== scp $name ===" -ForegroundColor Cyan
    if (Test-Path $localTar) { Remove-Item $localTar -Force }
    scp "${Remote}:/tmp/$name.tar.gz" $localTar
    if ($LASTEXITCODE -ne 0) { throw "scp failed: $name" }

    $dest = Join-Path $LocalRoot $name
    if (Test-Path $dest) { Remove-Item $dest -Recurse -Force }
    Write-Host "extract -> $LocalRoot"
    tar -xzf $localTar -C $LocalRoot
    if ($LASTEXITCODE -ne 0) { throw "extract failed: $name" }
    Remove-Item $localTar -Force -ErrorAction SilentlyContinue
    Write-Host "OK: $dest" -ForegroundColor Green
}

Write-Host "`n탐색기:"; Write-Host "  explorer `"$LocalRoot`""
Get-ChildItem $LocalRoot -Directory | ForEach-Object {
    $n = (Get-ChildItem $_.FullName -Recurse -File -Include *.mp4,*.webm,*.mov,*.avi,*.mkv -EA SilentlyContinue).Count
    Write-Host ("{0}: {1} videos" -f $_.Name, $n)
}
