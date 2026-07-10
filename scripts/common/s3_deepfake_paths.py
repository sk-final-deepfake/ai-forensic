"""S3 prefix constants for the deepfake evidence bucket layout.

Override any default via env (see S3_DEEPFAKE_FOLDER_LAYOUT.md).
"""
from __future__ import annotations

import os

_ROOT = os.getenv("S3_DEEPFAKE_ROOT", "deepfake")

DATASETS_BENCH = os.getenv("S3_DEEPFAKE_DATASETS_BENCH", f"{_ROOT}/datasets/bench")
DATASETS_TRAIN = os.getenv("S3_DEEPFAKE_DATASETS_TRAIN", f"{_ROOT}/datasets/train")
DATASETS_FIELD = os.getenv("S3_DEEPFAKE_DATASETS_FIELD", f"{_ROOT}/datasets/field")
RESULTS_INFER = os.getenv("S3_DEEPFAKE_RESULTS_INFER", f"{_ROOT}/results/infer")
RESULTS_PERF = os.getenv("S3_DEEPFAKE_RESULTS_PERF", f"{_ROOT}/results/perf")
ARCHIVE_LEGACY = os.getenv("S3_DEEPFAKE_ARCHIVE_LEGACY", f"{_ROOT}/archive/legacy-benchmarks")
ARTIFACTS_ANALYSIS = os.getenv("S3_DEEPFAKE_ARTIFACTS_ANALYSIS", f"{_ROOT}/artifacts/analysis")


def bench_profile(profile: str) -> str:
    return f"{DATASETS_BENCH}/{profile}"


def infer_model(model_slug: str, profile: str | None = None) -> str:
    base = f"{RESULTS_INFER}/{model_slug}"
    return f"{base}/{profile}" if profile else base


def legacy_reports(benchmark_name: str) -> str:
    return f"{ARCHIVE_LEGACY}/{benchmark_name}/reports"


LEGACY_XCEPTION = legacy_reports("video-xception-benchmark")
LEGACY_EFFICIENTNETB4 = legacy_reports("video-efficientnetb4-benchmark")
LEGACY_CONVNEXT_CELEBDF = legacy_reports("video-convnext-celebdf-benchmark")
LEGACY_VIDEOMAE = legacy_reports("video-videomae-benchmark")
LEGACY_VIDEOMAE_CELEBDF = legacy_reports("video-videomae-celebdf-benchmark")
LEGACY_VIDEOMAE_DFDC = legacy_reports("video-videomae-dfdc-benchmark")
LEGACY_TIMESFORMER_CELEBDF = legacy_reports("video-timesformer-celebdf-benchmark")
LEGACY_VIDEO_SWIN_CELEBDF = legacy_reports("video-swin-celebdf-benchmark")
LEGACY_OPTICAL_FLOW = legacy_reports("video-optical-flow-benchmark")
