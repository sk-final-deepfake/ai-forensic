# TruFor 고정 테스트셋 400건 — GPU 서버 → 로컬 PC
#
# 대상:
#   mvtamperbench-200-s3   (real 100 + fake 100)
#   csvted-200-balanced    (real 100 + fake 100)
#
# 사용 (PowerShell, 비밀번호 입력 가능한 터미널):
#   cd c:\FINAL\ai-forensic\scripts\download\data
#   .\download_trufor_benchmark_evidence.ps1
#
# 옵션:
#   .\download_trufor_benchmark_evidence.ps1 -Mode tar    # 권장: 서버에서 tar 후 1회 scp (빠름)
#   .\download_trufor_benchmark_evidence.ps1 -Mode scp    # 폴더 단위 recursive scp
#   .\download_trufor_benchmark_evidence.ps1 -Dataset mvtb   # mvtb만
#   .\download_trufor_benchmark_evidence.ps1 -Dataset csvted # csvted만

param(
    [ValidateSet("tar", "scp")]
    [string]$Mode = "tar",
    [ValidateSet("all", "mvtb", "csvted")]
    [string]$Dataset = "all",
    [string]$RemoteHost = "sk4team@58.151.205.220",
    [string]$RemoteEvidenceRoot = "~/forenShield-ai/forgery/data/pull/evidence",
    [string]$LocalRoot = "c:\FINAL\ai-forensic\data\pull\evidence"
)

$ErrorActionPreference = "Stop"

$mvtbName = "mvtamperbench-200-s3"
$csvtedName = "csvted-200-balanced"

function Ensure-Dir([string]$Path) {
    if (-not (Test-Path $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

function Invoke-Remote([string]$Command) {
    # PowerShell이 && 를 로컬에서 해석하지 않도록 원격 명령을 반드시 한 덩어리로 전달
    Write-Host ">> ssh $RemoteHost `"$Command`"" -ForegroundColor DarkGray
    ssh -t $RemoteHost "bash -lc $(ConvertTo-Json $Command)"
    if ($LASTEXITCODE -ne 0) { throw "remote command failed: $Command" }
}

function Download-TarBundle([string]$FolderName) {
    $remoteTar = "/tmp/${FolderName}.tar.gz"
    $localTar = Join-Path $env:TEMP "${FolderName}.tar.gz"
    $localDest = Join-Path $LocalRoot $FolderName

    Write-Host "`n=== $FolderName (tar mode) ===" -ForegroundColor Cyan
    Invoke-Remote "cd $RemoteEvidenceRoot && test -d '$FolderName' && tar czf '$remoteTar' '$FolderName' && du -sh '$remoteTar'"

    if (Test-Path $localTar) { Remove-Item $localTar -Force }
    Write-Host ">> scp $RemoteHost`:$remoteTar -> $localTar"
    scp "${RemoteHost}:${remoteTar}" $localTar
    if ($LASTEXITCODE -ne 0) { throw "scp failed for $FolderName" }

    if (Test-Path $localDest) {
        Write-Host "기존 폴더 삭제: $localDest"
        Remove-Item $localDest -Recurse -Force
    }
    Ensure-Dir $LocalRoot

    Write-Host "압축 해제 중..."
    tar -xzf $localTar -C $LocalRoot
    if ($LASTEXITCODE -ne 0) { throw "tar extract failed for $FolderName" }

    Invoke-Remote "rm -f '$remoteTar'"
    Remove-Item $localTar -Force -ErrorAction SilentlyContinue
    Write-Host "완료: $localDest" -ForegroundColor Green
}

function Download-ScpFolder([string]$FolderName) {
    $localDest = Join-Path $LocalRoot $FolderName
    Write-Host "`n=== $FolderName (scp mode) ===" -ForegroundColor Cyan
    Ensure-Dir $LocalRoot
    if (Test-Path $localDest) {
        Write-Host "기존 폴더 삭제: $localDest"
        Remove-Item $localDest -Recurse -Force
    }
    Write-Host ">> scp -r ${RemoteHost}:${RemoteEvidenceRoot}/${FolderName} $LocalRoot\"
    scp -r "${RemoteHost}:${RemoteEvidenceRoot}/${FolderName}" $LocalRoot
    if ($LASTEXITCODE -ne 0) { throw "scp failed for $FolderName" }
    Write-Host "완료: $localDest" -ForegroundColor Green
}

function Show-Summary {
    param([string[]]$Folders)
    Write-Host "`n--- 로컬 요약 ---" -ForegroundColor Yellow
    foreach ($f in $Folders) {
        $p = Join-Path $LocalRoot $f
        if (-not (Test-Path $p)) {
            Write-Host "$f : (없음)"
            continue
        }
        $videos = Get-ChildItem $p -Recurse -File -Include *.mp4,*.webm,*.avi,*.mov,*.mkv -ErrorAction SilentlyContinue
        $real = ($videos | Where-Object { $_.FullName -match '\\original\\' }).Count
        $fake = ($videos | Where-Object { $_.FullName -match '\\tampered\\' }).Count
        $sizeMb = [math]::Round((($videos | Measure-Object Length -Sum).Sum / 1MB), 1)
        Write-Host "$f : $($videos.Count) videos (original~$real, tampered~$fake) ~${sizeMb} MB"
        Write-Host "  경로: $p"
    }
    Write-Host "`n탐색기로 열기:"
    Write-Host "  explorer `"$LocalRoot`""
}

Ensure-Dir $LocalRoot

$targets = @()
if ($Dataset -eq "all" -or $Dataset -eq "mvtb") { $targets += $mvtbName }
if ($Dataset -eq "all" -or $Dataset -eq "csvted") { $targets += $csvtedName }

Write-Host "Remote: $RemoteHost"
Write-Host "Local : $LocalRoot"
Write-Host "Mode  : $Mode"
Write-Host "Sets  : $($targets -join ', ')"

foreach ($name in $targets) {
    if ($Mode -eq "tar") {
        Download-TarBundle $name
    } else {
        Download-ScpFolder $name
    }
}

Show-Summary -Folders $targets
