# TruFor train scripts -> GPU server
# Usage (PowerShell): .\deploy_to_gpu.ps1
# Optional: $env:GPU_HOST = "sk4team@58.127.241.84"
#
# Batched scp (3 password prompts on Windows; SSH key = zero prompts).
# Note: Windows OpenSSH ControlMaster is unreliable — not used here.

$ErrorActionPreference = "Stop"

# Stale ControlMaster socket from a previous failed deploy (Windows OpenSSH bug)
Get-ChildItem (Join-Path $env:USERPROFILE ".ssh") -Filter "deploy-gpu-*" -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

$Remote = if ($env:GPU_HOST) { $env:GPU_HOST } else { "sk4team@58.127.241.84" }
$LocalDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RemoteTrain = "~/forenShield-ai/forgery/scripts/train"

$files = @(
    "prepare_trufor_video_frames.py",
    "train_trufor_video_forgery.py",
    "trufor_video_common.py",
    "merge_trufor_infer_checkpoint.py",
    "run_trufor_forgery_train.sh",
    "run_trufor_forgery_train_v2.sh"
)

$patches = @(
    "dataset_ForenShieldVideo.py",
    "trufor_forgery_video.yaml",
    "trufor_forgery_video_v2.yaml"
)

function Resolve-LocalPaths([string[]]$Names, [string]$SubDir) {
    $paths = @()
    foreach ($name in $Names) {
        $src = if ($SubDir) { Join-Path $LocalDir (Join-Path $SubDir $name) } else { Join-Path $LocalDir $name }
        if (-not (Test-Path $src)) {
            throw "Missing local file: $src"
        }
        $paths += $src
    }
    return $paths
}

Write-Host "Deploy TruFor train scripts to $Remote"
Write-Host "Password: up to 3 times (scripts batch, patches batch, verify). Use SSH key to skip."
Write-Host ""

$mainPaths = Resolve-LocalPaths $files ""

# vendor_patches/*.py|yaml existence check
foreach ($patch in $patches) {
    $null = Resolve-LocalPaths @($patch) "vendor_patches"
}

Write-Host "[1/3] scp train scripts ($($mainPaths.Count) files) ..."
scp @mainPaths "${Remote}:${RemoteTrain}/"

Write-Host "[2/3] scp vendor_patches/ ..."
$patchDir = Join-Path $LocalDir "vendor_patches"
scp -r $patchDir "${Remote}:${RemoteTrain}/"

Write-Host "[3/3] verify on server ..."
ssh $Remote @"
ls -la ${RemoteTrain}/run_trufor_forgery_train_v2.sh ${RemoteTrain}/prepare_trufor_video_frames.py ${RemoteTrain}/vendor_patches/trufor_forgery_video_v2.yaml
python3 -m py_compile ${RemoteTrain}/prepare_trufor_video_frames.py && echo 'prepare_trufor_video_frames.py: syntax OK'
"@

Write-Host ""
Write-Host "Done. On GPU:"
Write-Host "  cd ~/forenShield-ai/forgery && source ../.venv/bin/activate"
Write-Host "  sed -i 's/\r$//' scripts/train/run_trufor_forgery_train_v2.sh"
Write-Host "  bash scripts/train/run_trufor_forgery_train_v2.sh"
