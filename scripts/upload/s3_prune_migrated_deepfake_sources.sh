#!/usr/bin/env bash
# Delete migrated deepfake S3 sources only when destination copy exists (same size).
#
# Usage:
#   source ~/forenShield-ai/config/env.local && unset AWS_PROFILE
#   bash scripts/upload/s3_prune_migrated_deepfake_sources.sh
#   APPLY=1 PHASE=evidence bash scripts/upload/s3_prune_migrated_deepfake_sources.sh
#
# Doc: docs/ops/S3_DEEPFAKE_FOLDER_LAYOUT.md (Phase 2)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "${ROOT}/scripts/upload/s3_prune_migrated_deepfake_sources.py"
