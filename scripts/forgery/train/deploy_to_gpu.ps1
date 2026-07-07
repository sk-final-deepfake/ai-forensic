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
    "patch_builder_np_conf_load.py",
    "run_trufor_forgery_train.sh",
    "run_trufor_forgery_train_v2.sh",
    "run_trufor_forgery_train_v4.sh",
    "run_trufor_forgery_train_r1.sh",
    "run_trufor_forgery_train_r2.sh",
    "run_trufor_forgery_train_r3.sh",
    "run_trufor_forgery_train_r4.sh",
    "run_trufor_forgery_train_r5.sh",
    "run_trufor_forgery_train_r5_calibrated.sh",
    "run_trufor_forgery_train_s1_calibrated.sh",
    "spatial_benchmark_calibrate_from_predictions.py",
    "discover_mvtb_video_pools.py",
    "evaluate_mvtb_holdout_predictions.py",
    "prepare_mvtb_holdout_benchmark.py",
    "run_mvtb500_holdout_eval.sh",
    "record_r3_mvtb_dev_adoption.sh",
    "record_r5_mvtb_dev_adoption.sh",
    "trufor_deepfake_benchmark_infer.py"
)

$configFiles = @(
    "trufor_r3_mvtb_dev_calibration.json",
    "trufor_r5_mvtb_dev_calibration.json"
)

$patches = @(
    "dataset_ForenShieldVideo.py",
    "trufor_forgery_video.yaml",
    "trufor_forgery_video_v2.yaml",
    "trufor_forgery_video_v3.yaml",
    "trufor_forgery_video_v4.yaml",
    "trufor_forgery_video_r1.yaml",
    "trufor_forgery_video_r2.yaml",
    "trufor_forgery_video_r0.yaml",
    "trufor_forgery_video_r4.yaml",
    "trufor_forgery_video_r5.yaml",
    "trufor_forgery_video_s1.yaml"
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
Write-Host "Password: up to 4 times (scripts, config, patches, verify). Use SSH key to skip."
Write-Host ""

$mainPaths = Resolve-LocalPaths $files ""

# vendor_patches/*.py|yaml existence check
foreach ($patch in $patches) {
    $null = Resolve-LocalPaths @($patch) "vendor_patches"
}

Write-Host "[1/4] scp train scripts ($($mainPaths.Count) files) ..."
scp @mainPaths "${Remote}:${RemoteTrain}/"

Write-Host "[2/4] scp config/forgery/ ..."
$aiForensicRoot = (Resolve-Path (Join-Path $LocalDir "..\..\..")).Path
$configDir = Join-Path $aiForensicRoot "config\forgery"
$configPaths = @()
foreach ($name in $configFiles) {
    $src = Join-Path $configDir $name
    if (-not (Test-Path $src)) { throw "Missing local file: $src" }
    $configPaths += $src
}
ssh $Remote "mkdir -p ~/forenShield-ai/forgery/config/forgery"
scp @configPaths "${Remote}:~/forenShield-ai/forgery/config/forgery/"

Write-Host "[3/4] scp vendor_patches/ ..."
$patchDir = Join-Path $LocalDir "vendor_patches"
scp -r $patchDir "${Remote}:${RemoteTrain}/"

Write-Host "[4/4] verify on server ..."
ssh $Remote @"
ls -la ${RemoteTrain}/run_trufor_forgery_train_r5.sh ${RemoteTrain}/vendor_patches/trufor_forgery_video_r5.yaml ${RemoteTrain}/trufor_deepfake_benchmark_infer.py
python3 -m py_compile ${RemoteTrain}/prepare_trufor_video_frames.py && echo 'prepare_trufor_video_frames.py: syntax OK'
python3 -m py_compile ${RemoteTrain}/trufor_deepfake_benchmark_infer.py && echo 'trufor_deepfake_benchmark_infer.py: syntax OK'
"@

Write-Host ""
Write-Host "Done. On GPU:"
Write-Host "  cd ~/forenShield-ai/forgery && source ../.venv/bin/activate"
Write-Host "  sed -i 's/\r$//' scripts/train/record_r5_mvtb_dev_adoption.sh"
Write-Host "  bash scripts/train/run_trufor_forgery_train_s1_calibrated.sh   # S1 line (isolated from R-line)"
