# TruFor train scripts -> GPU server
# Usage (PowerShell): .\deploy_to_gpu.ps1
# Optional: $env:GPU_HOST = "sk4team@58.127.241.84"

$ErrorActionPreference = "Stop"

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

Write-Host "Deploy TruFor train scripts to $Remote"

foreach ($f in $files) {
    scp (Join-Path $LocalDir $f) "${Remote}:${RemoteTrain}/"
}

ssh $Remote "mkdir -p ${RemoteTrain}/vendor_patches"
scp (Join-Path $LocalDir "vendor_patches\dataset_ForenShieldVideo.py") "${Remote}:${RemoteTrain}/vendor_patches/"
scp (Join-Path $LocalDir "vendor_patches\trufor_forgery_video.yaml") "${Remote}:${RemoteTrain}/vendor_patches/"
scp (Join-Path $LocalDir "vendor_patches\trufor_forgery_video_v2.yaml") "${Remote}:${RemoteTrain}/vendor_patches/"

Write-Host "Done. SSH and run Phase 1:"
Write-Host "  cd ~/forenShield-ai/forgery && source ../.venv/bin/activate"
Write-Host "  sed -i 's/\r$//' scripts/train/run_trufor_forgery_train_v2.sh"
Write-Host "  bash scripts/train/run_trufor_forgery_train_v2.sh"
