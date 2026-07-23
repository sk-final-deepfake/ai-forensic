# TruFor / forgery scripts -> GPU server
# Usage (PowerShell): .\deploy_to_gpu.ps1
# Optional: $env:GPU_HOST = "sk4team@58.151.205.220"
#
# Batched scp (password prompts on Windows; SSH key = zero prompts).
# Note: Windows OpenSSH ControlMaster is unreliable — not used here.

$ErrorActionPreference = "Stop"

# Stale ControlMaster socket from a previous failed deploy (Windows OpenSSH bug)
Get-ChildItem (Join-Path $env:USERPROFILE ".ssh") -Filter "deploy-gpu-*" -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

$Remote = if ($env:GPU_HOST) { $env:GPU_HOST } else { "sk4team@58.151.205.220" }
$LocalTrain = Split-Path -Parent $MyInvocation.MyCommand.Path
$LocalForgery = Split-Path -Parent $LocalTrain
$LocalInfer = Join-Path $LocalForgery "infer"
$LocalData = Join-Path $LocalForgery "data"
$RemoteTrain = "~/forenShield-ai/forgery/scripts/train"
$RemoteInfer = "~/forenShield-ai/forgery/scripts/infer"
$RemoteData = "~/forenShield-ai/forgery/scripts/data"

$trainFiles = @(
    "prepare_trufor_video_frames.py",
    "train_trufor_video_forgery.py",
    "trufor_video_common.py",
    "patch_builder_np_conf_load.py",
    "run_trufor_forgery_train_r5_calibrated.sh",
    "run_trufor_forgery_train_s1_calibrated.sh",
    "run_trufor_forgery_train_s2_calibrated.sh",
    "run_trufor_forgery_train_f16_calibrated.sh",
    "discover_mvtb_video_pools.py",
    "evaluate_mvtb_holdout_predictions.py",
    "prepare_mvtb_holdout_benchmark.py",
    "run_mvtb500_holdout_eval.sh",
    "mine_timesformer_forgery_contrastive_pairs.py",
    "train_timesformer_forgery_window_mil.py",
    "train_timesformer_forgery_contrastive_mil.py",
    "run_timesformer_forgery_v1.9_hardneg.sh",
    "run_timesformer_forgery_v1.8_csvted_boost.sh",
    "run_timesformer_forgery_v1.7_contrastive.sh",
    "run_timesformer_forgery_v1.6_rank.sh",
    "run_timesformer_forgery_v1.4_prepare_temporal_train.sh"
)

$inferFiles = @(
    "spatial_benchmark_calibrate_from_predictions.py",
    "spatial_benchmark_calibration_validate.py",
    "trufor_deepfake_benchmark_infer.py",
    "spatial_mvtamperbench_benchmark.py",
    "timesformer_forgery_features.py",
    "timesformer_forgery_benchmark.py",
    "tamper_segment_labels.py",
    "video_decode_robust.py",
    "run_timesformer_forgery_benchmark.sh",
    "sweep_spatial_benchmark_threshold.py",
    "sweep_timesformer_forgery_threshold.py"
)

$dataFiles = @(
    "prepare_gmflow_temporal_dataset.py"
)

$configFiles = @(
    "trufor_r5_mvtb_dev_calibration.json",
    "trufor_videocof_v2_dev_adoption.json"
)

$patches = @(
    "dataset_ForenShieldVideo.py",
    "trufor_forgery_video_r5.yaml",
    "trufor_forgery_video_s1.yaml",
    "trufor_forgery_video_s2.yaml"
)

function Resolve-LocalPaths([string[]]$Names, [string]$BaseDir, [string]$SubDir) {
    $paths = @()
    foreach ($name in $Names) {
        $src = if ($SubDir) { Join-Path $BaseDir (Join-Path $SubDir $name) } else { Join-Path $BaseDir $name }
        if (-not (Test-Path $src)) {
            throw "Missing local file: $src"
        }
        $paths += $src
    }
    return $paths
}

Write-Host "Deploy forgery scripts to $Remote"
Write-Host "Password: up to 6 times (train, infer, data, config, patches, verify). Use SSH key to skip."
Write-Host ""

$trainPaths = Resolve-LocalPaths $trainFiles $LocalTrain ""
$inferPaths = Resolve-LocalPaths $inferFiles $LocalInfer ""
$dataPaths = Resolve-LocalPaths $dataFiles $LocalData ""

foreach ($patch in $patches) {
    $null = Resolve-LocalPaths @($patch) $LocalTrain "vendor_patches"
}

Write-Host "[1/6] scp train scripts ($($trainPaths.Count) files) ..."
scp @trainPaths "${Remote}:${RemoteTrain}/"

Write-Host "[2/6] scp infer scripts ($($inferPaths.Count) files) ..."
ssh $Remote "mkdir -p $RemoteInfer"
scp @inferPaths "${Remote}:${RemoteInfer}/"

Write-Host "[3/6] scp data scripts ($($dataPaths.Count) files) ..."
ssh $Remote "mkdir -p $RemoteData"
scp @dataPaths "${Remote}:${RemoteData}/"

Write-Host "[4/6] scp config/forgery/ ..."
$aiForensicRoot = (Resolve-Path (Join-Path $LocalTrain "..\..\..")).Path
$configDir = Join-Path $aiForensicRoot "config\forgery"
$configPaths = @()
foreach ($name in $configFiles) {
    $src = Join-Path $configDir $name
    if (-not (Test-Path $src)) { throw "Missing local file: $src" }
    $configPaths += $src
}
ssh $Remote "mkdir -p ~/forenShield-ai/forgery/config/forgery"
scp @configPaths "${Remote}:~/forenShield-ai/forgery/config/forgery/"

Write-Host "[5/6] scp vendor_patches/ ..."
$patchDir = Join-Path $LocalTrain "vendor_patches"
scp -r $patchDir "${Remote}:${RemoteTrain}/"

Write-Host "[6/6] verify on server ..."
ssh $Remote @"
ls -la ${RemoteTrain}/run_trufor_forgery_train_r5.sh ${RemoteTrain}/mine_timesformer_forgery_contrastive_pairs.py
ls -la ${RemoteInfer}/spatial_mvtamperbench_benchmark.py ${RemoteInfer}/sweep_spatial_benchmark_threshold.py ${RemoteInfer}/sweep_timesformer_forgery_threshold.py
ls -la ${RemoteData}/prepare_gmflow_temporal_dataset.py
python3 -m py_compile ${RemoteTrain}/prepare_trufor_video_frames.py && echo 'prepare_trufor_video_frames.py: syntax OK'
python3 -m py_compile ${RemoteInfer}/spatial_mvtamperbench_benchmark.py && echo 'spatial_mvtamperbench_benchmark.py: syntax OK'
python3 -m py_compile ${RemoteData}/prepare_gmflow_temporal_dataset.py && echo 'prepare_gmflow_temporal_dataset.py: syntax OK'
"@

Write-Host ""
Write-Host "Done. On GPU:"
Write-Host "  cd ~/forenShield-ai/forgery && source ../.venv/bin/activate"
Write-Host "  bash scripts/train/run_trufor_forgery_train_s1_calibrated.sh"
