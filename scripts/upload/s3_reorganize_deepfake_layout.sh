#!/usr/bin/env bash
# S3 딥페이크 prefix 정리 — 복사만, 삭제 없음.
#
# Usage:
#   source ~/forenShield-ai/config/env.local && unset AWS_PROFILE
#   bash scripts/upload/s3_reorganize_deepfake_layout.sh              # dry-run all
#   APPLY=1 PHASE=evidence bash scripts/upload/s3_reorganize_deepfake_layout.sh
#   APPLY=1 PHASE=models bash scripts/upload/s3_reorganize_deepfake_layout.sh
#
# Doc: docs/S3_DEEPFAKE_FOLDER_LAYOUT.md
set -euo pipefail

EVIDENCE_BUCKET="${S3_EVIDENCE_BUCKET:-forenshield-evidence-877044078824}"
MODELS_BUCKET="${S3_MODELS_BUCKET:-forenshield-models-877044078824}"
APPLY="${APPLY:-0}"
PHASE="${PHASE:-all}"

if [[ "${APPLY}" == "1" ]]; then
  DRYRUN=()
  echo "==> APPLY=1: copying to new deepfake/ prefixes (no deletes)"
else
  DRYRUN=(--dryrun)
  echo "==> dry-run only (set APPLY=1 to copy)"
fi

sync_pair() {
  local src_bucket="$1"
  local src_prefix="$2"
  local dst_bucket="$3"
  local dst_prefix="$4"
  local tag="$5"

  src_prefix="${src_prefix%/}"
  dst_prefix="${dst_prefix%/}"
  echo ""
  echo "--- [${tag}] s3://${src_bucket}/${src_prefix}/ -> s3://${dst_bucket}/${dst_prefix}/"
  if ! aws s3 ls "s3://${src_bucket}/${src_prefix}/" >/dev/null 2>&1; then
    echo "    (skip: source missing)"
    return 0
  fi
  aws s3 sync "s3://${src_bucket}/${src_prefix}/" "s3://${dst_bucket}/${dst_prefix}/" "${DRYRUN[@]}"
}

upload_readme() {
  local bucket="$1"
  local key="$2"
  local tmp
  tmp="$(mktemp)"
  cat >"${tmp}" <<EOF
# ForenShield deepfake S3 layout

See ai-forensic/docs/S3_DEEPFAKE_FOLDER_LAYOUT.md

- datasets/train   : training manifests & clips
- datasets/golden  : regression golden set
- datasets/bench   : benchmark input mp4 (celebdf, ffpp_vox)
- datasets/field   : youtube / ad-hoc field tests
- results/perf     : metrics & benchmark reports
- results/infer    : per-video infer JSON
- artifacts/analysis : heatmap / overlay from production
- archive/legacy   : deprecated benchmark prefixes (copies)
EOF
  if [[ "${APPLY}" == "1" ]]; then
    aws s3 cp "${tmp}" "s3://${bucket}/${key}" --content-type "text/markdown"
    echo "uploaded s3://${bucket}/${key}"
  else
    echo "(dry-run) would upload s3://${bucket}/${key}"
  fi
  rm -f "${tmp}"
}

run_evidence() {
  echo "========== EVIDENCE: ${EVIDENCE_BUCKET} =========="

  # Bench datasets (shared mp4)
  sync_pair "${EVIDENCE_BUCKET}" "cases/test/video-benchmark-datasets/celebdf" \
    "${EVIDENCE_BUCKET}" "deepfake/datasets/bench/celebdf" "bench-dataset"
  sync_pair "${EVIDENCE_BUCKET}" "cases/test/video-benchmark-datasets/ffpp_vox" \
    "${EVIDENCE_BUCKET}" "deepfake/datasets/bench/ffpp_vox" "bench-dataset"

  # Train manifests
  sync_pair "${EVIDENCE_BUCKET}" "cases/train/video/xception" \
    "${EVIDENCE_BUCKET}" "deepfake/datasets/train/video/xception" "train"

  # Field / ad-hoc
  sync_pair "${EVIDENCE_BUCKET}" "cases/test/youtube-shorts-adhoc" \
    "${EVIDENCE_BUCKET}" "deepfake/datasets/field/youtube-shorts" "field"

  # Infer results from unified benchmark tree (model slugs)
  local models=(xception timesformer videomae video-swin convnext raft gmflow)
  local model
  for model in "${models[@]}"; do
    sync_pair "${EVIDENCE_BUCKET}" "cases/test/video-benchmark-datasets/${model}" \
      "${EVIDENCE_BUCKET}" "deepfake/results/infer/${model}" "infer"
  done
  sync_pair "${EVIDENCE_BUCKET}" "cases/test/video-benchmark-datasets/PWC-Net" \
    "${EVIDENCE_BUCKET}" "deepfake/archive/legacy-benchmarks/pwcnet" "archive"

  # Legacy per-model benchmark trees
  local legacy_prefixes=(
    video-xception-benchmark
    video-videomae-benchmark
    video-videomae-celebdf-benchmark
    video-timesformer-celebdf-benchmark
    video-swin-celebdf-benchmark
    video-convnext-celebdf-benchmark
    video-optical-flow-benchmark
    video-raft-ffpp-vox-benchmark
  )
  local legacy
  for legacy in "${legacy_prefixes[@]}"; do
    sync_pair "${EVIDENCE_BUCKET}" "cases/test/${legacy}" \
      "${EVIDENCE_BUCKET}" "deepfake/archive/legacy-benchmarks/${legacy}" "legacy"
  done

  # Unused experiments
  sync_pair "${EVIDENCE_BUCKET}" "cases/test/test-sine" \
    "${EVIDENCE_BUCKET}" "deepfake/archive/legacy-experiments/test-sine" "archive"

  upload_readme "${EVIDENCE_BUCKET}" "deepfake/README.md"
}

run_models() {
  echo "========== MODELS: ${MODELS_BUCKET} =========="

  # Operational deploy weights (adjust subpaths if your tree differs)
  local deploy_models=(xception timesformer)
  local m
  for m in "${deploy_models[@]}"; do
    sync_pair "${MODELS_BUCKET}" "video/${m}" \
      "${MODELS_BUCKET}" "deepfake/deploy/video/${m}" "deploy"
  done
  sync_pair "${MODELS_BUCKET}" "video/gmflow" \
    "${MODELS_BUCKET}" "deepfake/deploy/video/optical/gmflow" "deploy"

  # Bench-only models
  local bench_models=(convnext videomae video-swin)
  for m in "${bench_models[@]}"; do
    sync_pair "${MODELS_BUCKET}" "video/${m}" \
      "${MODELS_BUCKET}" "deepfake/bench/video/${m}" "bench"
  done

  # Root legacy snapshots
  sync_pair "${MODELS_BUCKET}" "v1.0" "${MODELS_BUCKET}" "deepfake/archive/root-v1.0" "archive"
  sync_pair "${MODELS_BUCKET}" "v1.1" "${MODELS_BUCKET}" "deepfake/archive/root-v1.1" "archive"
  sync_pair "${MODELS_BUCKET}" "test" "${MODELS_BUCKET}" "deepfake/archive/root-test" "archive"
  sync_pair "${MODELS_BUCKET}" "test-sets" "${MODELS_BUCKET}" "deepfake/archive/root-test-sets" "archive"

  upload_readme "${MODELS_BUCKET}" "deepfake/README.md"
}

case "${PHASE}" in
  evidence) run_evidence ;;
  models) run_models ;;
  all)
    run_evidence
    run_models
    ;;
  *)
    echo "Unknown PHASE=${PHASE} (use evidence|models|all)" >&2
    exit 1
    ;;
esac

echo ""
echo "Done. Verify with:"
echo "  aws s3 ls s3://${EVIDENCE_BUCKET}/deepfake/"
echo "  aws s3 ls s3://${MODELS_BUCKET}/deepfake/"
