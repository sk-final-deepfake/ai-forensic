"""Shared GMFlow golden-200 eval helpers."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from gmflow_feature_extract import normalize_report, profile_from_filename

S3_BASE = (
    "s3://forenshield-evidence-877044078824/"
    "cases/test/video-benchmark-datasets/gmflow"
)
DEFAULT_RUNS = {
    "ffpp_vox": f"{S3_BASE}/ffpp_vox/gmflow-ffpp-vox-benchmark-20260622-0544",
    "celebdf": f"{S3_BASE}/celebdf/gmflow-celebdf-benchmark-20260622-0142",
}
PROFILES = ("ffpp_vox", "celebdf")

# Legacy domain hold-out (ffpp train / celeb test within golden 200).
DEFAULT_TRAIN_PROFILE = "ffpp_vox"
DEFAULT_TEST_PROFILE = "celebdf"

# Pull-train infer cache (ff1k + celeb1k manifest train/val JSON).
DEFAULT_TRAIN_CACHE = "docs/notebooks/output/.gmflow_train_cache"
DEFAULT_TEST_CACHE = "docs/notebooks/output/.gmflow_cache"
PULL_TRAIN_STAGES = ("ff1k", "celeb1k")
PULL_TRAIN_SPLITS = ("train", "val")


def report_profile(report: dict[str, Any]) -> str | None:
    return profile_from_filename(str(report.get("file") or ""))


def split_holdout(
    reports: list[dict[str, Any]],
    *,
    train_profile: str = DEFAULT_TRAIN_PROFILE,
    test_profile: str = DEFAULT_TEST_PROFILE,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = [r for r in reports if report_profile(r) == train_profile]
    test = [r for r in reports if report_profile(r) == test_profile]
    return train, test


def scores_vector(
    reports: list[dict[str, Any]],
    scores_by_file: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    labels, files = labels_and_files(reports)
    if not files:
        return labels, np.array([], dtype=np.float64), files
    kept_labels: list[int] = []
    kept_files: list[str] = []
    kept_scores: list[float] = []
    for lab, fn in zip(labels, files, strict=True):
        score = scores_by_file.get(fn)
        if score is None:
            continue
        kept_labels.append(int(lab))
        kept_files.append(fn)
        kept_scores.append(float(score))
    return (
        np.array(kept_labels, dtype=np.int64),
        np.array(kept_scores, dtype=np.float64),
        kept_files,
    )


def pick_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    *,
    threshold_mode: str = "max_fake_rec_50fpr",
) -> float:
    if threshold_mode == "max_fake_rec_50fpr":
        return best_threshold_max_fake_recall(labels, scores, max_fpr=0.5)
    return best_threshold_youden(labels, scores)


def eval_holdout(
    train_reports: list[dict[str, Any]],
    test_reports: list[dict[str, Any]],
    scores_by_file: dict[str, float],
    *,
    train_profile: str = DEFAULT_TRAIN_PROFILE,
    test_profile: str = DEFAULT_TEST_PROFILE,
    threshold_mode: str = "max_fake_rec_50fpr",
    fixed_threshold: float | None = None,
) -> dict[str, Any]:
    """Tune threshold on train profile only; report metrics on test profile only."""
    train_labels, train_scores, _ = scores_vector(train_reports, scores_by_file)
    test_labels, test_scores, _ = scores_vector(test_reports, scores_by_file)

    if fixed_threshold is not None:
        t = float(fixed_threshold)
    elif train_labels.size and len(np.unique(train_labels)) > 1:
        t = pick_threshold(train_labels, train_scores, threshold_mode=threshold_mode)
    else:
        t = 0.5

    train_metrics = metrics_at_threshold(train_labels, train_scores, t) if train_labels.size else None
    test_metrics = metrics_at_threshold(test_labels, test_scores, t) if test_labels.size else None
    test_auc = float(roc_auc_score(test_labels, test_scores)) if test_labels.size and len(np.unique(test_labels)) > 1 else None

    return {
        "split": "holdout",
        "train_profile": train_profile,
        "test_profile": test_profile,
        "threshold_mode": threshold_mode,
        "threshold": t,
        "threshold_display": round(t, 6),
        "train_n": int(train_labels.size),
        "test_n": int(test_labels.size),
        "train_in_sample": train_metrics,
        "test_holdout": test_metrics,
        "test_auc": round(test_auc, 4) if test_auc is not None else None,
    }


def aws_cp(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", src, str(dst)], check=True)


def aws_sync(prefix: str, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", prefix.rstrip("/") + "/", str(dst)], check=True)


def load_profile_reports(cache_dir: Path, profile: str, *, download: bool) -> list[dict[str, Any]]:
    run_prefix = DEFAULT_RUNS[profile]
    local_run = cache_dir / profile
    json_dir = local_run / "json"
    summary_path = local_run / "infer_summary.json"

    if download or not any(json_dir.glob("*.json")):
        if not download and not summary_path.is_file():
            raise FileNotFoundError(f"missing cache: {json_dir} (use --download)")
        if download:
            aws_cp(f"{run_prefix}/infer_summary.json", summary_path)
            aws_sync(f"{run_prefix}/json/", json_dir)

    if json_dir.is_dir() and any(json_dir.glob("*.json")):
        reports = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(json_dir.glob("*.json"))]
    elif summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        reports = []
        for item in summary.get("items", []):
            reports.append(
                {
                    "file": item.get("file"),
                    "ground_truth_label": item.get("ground_truth_label"),
                    "status": item.get("status", "ok"),
                    "flow_mean": item.get("flow_mean"),
                    "motion_anomaly_score": item.get("motion_anomaly_score") or item.get("fake_score"),
                    "pred_label": item.get("pred_label"),
                    "score_breakdown": item.get("score_breakdown"),
                }
            )
    else:
        raise FileNotFoundError(f"GMFlow cache missing for {profile}: {local_run}")

    for report in reports:
        normalize_report(report)
    return reports


def load_all_reports(cache_dir: Path, *, download: bool) -> tuple[dict[str, list[dict]], list[dict]]:
    by_profile = {p: load_profile_reports(cache_dir, p, download=download) for p in PROFILES}
    combined = by_profile["ffpp_vox"] + by_profile["celebdf"]
    return by_profile, combined


def load_json_dir_reports(json_dir: Path) -> list[dict[str, Any]]:
    if not json_dir.is_dir():
        return []
    reports: list[dict[str, Any]] = []
    for path in sorted(json_dir.glob("*.json")):
        if path.name == "infer_summary.json":
            continue
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    for report in reports:
        normalize_report(report)
    return reports


def load_pull_split_reports(
    train_cache_dir: Path,
    stage: str,
    split: str,
) -> list[dict[str, Any]]:
    json_dir = train_cache_dir / stage / split / "json"
    return load_json_dir_reports(json_dir)


def load_pull_train_val_reports(
    train_cache_dir: Path,
    *,
    stages: tuple[str, ...] = PULL_TRAIN_STAGES,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load ff1k/celeb1k manifest train+val infer JSON (excludes golden 200)."""
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for stage in stages:
        for split, bucket in (("train", train_rows), ("val", val_rows)):
            rows = load_pull_split_reports(train_cache_dir, stage, split)
            if not rows:
                missing.append(f"{stage}/{split}")
            bucket.extend(rows)
    if missing:
        raise FileNotFoundError(
            "GMFlow pull-train infer cache incomplete. Missing JSON under "
            f"{train_cache_dir}/{{stage}}/{{train|val}}/json for: {', '.join(missing)}. "
            "Run scripts/eval/run_gmflow_pull_train_infer.py on GPU first."
        )
    return train_rows, val_rows


def load_golden_test_reports(
    test_cache_dir: Path,
    *,
    download: bool,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Golden 200 benchmark infer JSON — evaluation only."""
    return load_all_reports(test_cache_dir, download=download)


def eval_golden_test(
    val_reports: list[dict[str, Any]],
    test_reports: list[dict[str, Any]],
    scores_by_file: dict[str, float],
    *,
    threshold_mode: str = "max_fake_rec_50fpr",
    fixed_threshold: float | None = None,
) -> dict[str, Any]:
    """Tune threshold on pull val; report metrics on golden 200 (combined + profiles)."""
    val_labels, val_scores, _ = scores_vector(val_reports, scores_by_file)
    test_labels, test_scores, _ = scores_vector(test_reports, scores_by_file)

    if fixed_threshold is not None:
        t = float(fixed_threshold)
    elif val_labels.size and len(np.unique(val_labels)) > 1:
        t = pick_threshold(val_labels, val_scores, threshold_mode=threshold_mode)
    else:
        t = 0.5

    val_metrics = metrics_at_threshold(val_labels, val_scores, t) if val_labels.size else None
    test_combined = metrics_at_threshold(test_labels, test_scores, t) if test_labels.size else None
    test_auc = (
        float(roc_auc_score(test_labels, test_scores))
        if test_labels.size and len(np.unique(test_labels)) > 1
        else None
    )
    test_profiles: dict[str, Any] = {}
    for prof in PROFILES:
        prof_labels, prof_scores, _ = scores_vector(
            [r for r in test_reports if report_profile(r) == prof],
            scores_by_file,
        )
        if prof_labels.size:
            test_profiles[prof] = metrics_at_threshold(prof_labels, prof_scores, t)

    return {
        "split": "golden200_test",
        "threshold_mode": threshold_mode,
        "threshold": t,
        "threshold_display": round(t, 6),
        "val_n": int(val_labels.size),
        "val_tune": val_metrics,
        "test_n": int(test_labels.size),
        "test_golden_200": test_combined,
        "test_profiles": test_profiles,
        "test_auc": round(test_auc, 4) if test_auc is not None else None,
        # aliases for downstream readers
        "test_holdout": test_combined,
        "train_in_sample": val_metrics,
    }


def labels_and_files(reports: list[dict[str, Any]], *, profile: str | None = None) -> tuple[np.ndarray, list[str]]:
    ok = [
        r
        for r in reports
        if r.get("status") == "ok"
        and r.get("ground_truth_label") in ("fake", "real")
    ]
    if profile:
        ok = [r for r in ok if profile_from_filename(str(r.get("file") or "")) == profile]
    labels = np.array([1 if r["ground_truth_label"] == "fake" else 0 for r in ok], dtype=np.int64)
    files = [str(r.get("file") or "") for r in ok]
    return labels, files


def best_threshold_youden(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    youden = tpr - fpr
    return float(thresholds[int(np.argmax(youden))])


def best_threshold_max_fake_recall(labels: np.ndarray, scores: np.ndarray, *, max_fpr: float = 0.5) -> float:
    """Pick threshold that maximizes fake recall subject to FPR <= max_fpr."""
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(labels, scores)
    best_t = float(thresholds[0]) if thresholds.size else 0.5
    best_rec = -1.0
    for f, t, th in zip(fpr, tpr, thresholds):
        if f <= max_fpr and t > best_rec:
            best_rec = float(t)
            best_t = float(th)
    if best_rec < 0:
        return best_threshold_youden(labels, scores)
    return best_t


def metrics_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (scores >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else None
    return {
        "n": int(labels.size),
        "threshold": round(threshold, 6),
        "auc": round(auc, 4) if auc is not None else None,
        "accuracy": round(float((preds == labels).mean()), 4) if labels.size else None,
        "fake_recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "real_recall": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else None,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
    }


def eval_scores(
    reports: list[dict[str, Any]],
    scores_by_file: dict[str, float],
    *,
    profile: str | None = None,
    threshold_mode: str = "youden",
    fixed_threshold: float | None = None,
) -> dict[str, Any]:
    labels, files = labels_and_files(reports, profile=profile)
    scores = np.array([scores_by_file[f] for f in files], dtype=np.float64)
    if fixed_threshold is not None:
        t = fixed_threshold
    elif threshold_mode == "max_fake_rec_50fpr":
        t = best_threshold_max_fake_recall(labels, scores, max_fpr=0.5)
    else:
        t = best_threshold_youden(labels, scores)
    return {
        "profile": profile or "combined_200",
        "threshold_mode": threshold_mode,
        "best_threshold": round(t, 6),
        "default_T_0.5": metrics_at_threshold(labels, scores, 0.5),
        "optimized_T": metrics_at_threshold(labels, scores, t),
    }


def route_scores_by_profile(
    reports: list[dict[str, Any]],
    profile_scores: dict[str, dict[str, float]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for report in reports:
        if report.get("status") != "ok":
            continue
        fn = str(report.get("file") or "")
        prof = profile_from_filename(fn) or "ffpp_vox"
        bucket = profile_scores.get(prof) or profile_scores.get("ffpp_vox") or {}
        if fn in bucket:
            out[fn] = bucket[fn]
    return out
