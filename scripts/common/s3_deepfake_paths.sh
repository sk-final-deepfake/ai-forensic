# Shared S3 deepfake prefix defaults (override via env).
# Usage: source "$(dirname "$0")/../common/s3_deepfake_paths.sh"
S3_DEEPFAKE_ROOT="${S3_DEEPFAKE_ROOT:-deepfake}"
S3_DEEPFAKE_DATASETS_BENCH="${S3_DEEPFAKE_DATASETS_BENCH:-${S3_DEEPFAKE_ROOT}/datasets/bench}"
S3_DEEPFAKE_DATASETS_TRAIN="${S3_DEEPFAKE_DATASETS_TRAIN:-${S3_DEEPFAKE_ROOT}/datasets/train}"
S3_DEEPFAKE_DATASETS_FIELD="${S3_DEEPFAKE_DATASETS_FIELD:-${S3_DEEPFAKE_ROOT}/datasets/field}"
S3_DEEPFAKE_RESULTS_INFER="${S3_DEEPFAKE_RESULTS_INFER:-${S3_DEEPFAKE_ROOT}/results/infer}"
S3_DEEPFAKE_RESULTS_PERF="${S3_DEEPFAKE_RESULTS_PERF:-${S3_DEEPFAKE_ROOT}/results/perf}"
S3_DEEPFAKE_ARCHIVE_LEGACY="${S3_DEEPFAKE_ARCHIVE_LEGACY:-${S3_DEEPFAKE_ROOT}/archive/legacy-benchmarks}"
S3_DEEPFAKE_ARTIFACTS_ANALYSIS="${S3_DEEPFAKE_ARTIFACTS_ANALYSIS:-${S3_DEEPFAKE_ROOT}/artifacts/analysis}"

s3_bench_profile() {
  echo "${S3_DEEPFAKE_DATASETS_BENCH}/${1}"
}

s3_infer_model() {
  echo "${S3_DEEPFAKE_RESULTS_INFER}/${1}/${2}"
}

s3_legacy_reports() {
  echo "${S3_DEEPFAKE_ARCHIVE_LEGACY}/${1}/reports"
}
