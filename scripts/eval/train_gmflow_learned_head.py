#!/usr/bin/env python3
"""Train sklearn head on GMFlow flow features.

Strategies:
  pooled   — ffpp_vox train → celebdf val (cross-domain generalization probe)
  profile  — separate LR head per profile (deploy: route by profile)
  combined — train on all 200 with strong regularization
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gmflow_feature_extract import (
    build_dataset,
    feature_names,
    load_json_reports,
    profile_from_filename,
)

S3_BASE = (
    "s3://forenshield-evidence-877044078824/"
    "cases/test/video-benchmark-datasets/gmflow"
)
DEFAULT_RUNS = {
    "ffpp_vox": f"{S3_BASE}/ffpp_vox/gmflow-ffpp-vox-benchmark-20260622-0544",
    "celebdf": f"{S3_BASE}/celebdf/gmflow-celebdf-benchmark-20260622-0142",
}
PROFILES = ("ffpp_vox", "celebdf")


def aws_cp(src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "cp", src, str(dst)], check=True)


def aws_sync(prefix: str, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    subprocess.run(["aws", "s3", "sync", prefix.rstrip("/") + "/", str(dst)], check=True)


def ensure_cache(cache_dir: Path, *, download: bool) -> None:
    for profile, run_prefix in DEFAULT_RUNS.items():
        local = cache_dir / profile
        json_dir = local / "json"
        if any(json_dir.glob("*.json")):
            continue
        if not download:
            raise FileNotFoundError(f"missing cache: {json_dir} (use --download)")
        aws_cp(f"{run_prefix}/infer_summary.json", local / "infer_summary.json")
        aws_sync(f"{run_prefix}/json/", json_dir)


def best_threshold_youden(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(labels, scores)
    if thresholds.size == 0:
        return 0.5
    youden = tpr - fpr
    return float(thresholds[int(np.argmax(youden))])


def metrics_block(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (scores >= threshold).astype(np.int64)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) > 1 else None
    return {
        "n": int(labels.size),
        "threshold": round(threshold, 4),
        "auc": round(auc, 4) if auc is not None else None,
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "fake_recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "real_recall": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
    }


def predict_proba_fake(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    proba = pipeline.predict_proba(X)
    classes = list(pipeline.named_steps["clf"].classes_)
    fake_idx = classes.index(1)
    return proba[:, fake_idx]


def make_pipeline(C: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced")),
        ]
    )


def eval_profile_block(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    scores = predict_proba_fake(pipeline, X)
    t = threshold if threshold is not None else best_threshold_youden(y, scores)
    return {
        "default_T_0.5": metrics_block(y, scores, 0.5),
        "youden_T": metrics_block(y, scores, t),
        "best_threshold": round(t, 4),
    }


def train_pooled(
    ffpp_reports: list[dict],
    celeb_reports: list[dict],
    all_reports: list[dict],
    C: float,
) -> tuple[Pipeline, dict[str, Any]]:
    X_train, y_train, _ = build_dataset(ffpp_reports, profile="ffpp_vox")
    X_val, y_val, _ = build_dataset(celeb_reports, profile="celebdf")
    X_all, y_all, _ = build_dataset(all_reports)

    pipeline = make_pipeline(C)
    pipeline.fit(X_train, y_train)

    train_scores = predict_proba_fake(pipeline, X_train)
    t_youden = best_threshold_youden(y_train, train_scores)
    val_scores = predict_proba_fake(pipeline, X_val)
    all_scores = predict_proba_fake(pipeline, X_all)

    report = {
        "strategy": "pooled",
        "train_profile": "ffpp_vox",
        "val_profile": "celebdf",
        "train": {
            "default_T_0.5": metrics_block(y_train, train_scores, 0.5),
            "youden_T": metrics_block(y_train, train_scores, t_youden),
            "best_threshold": round(t_youden, 4),
        },
        "val": {
            "default_T_0.5": metrics_block(y_val, val_scores, 0.5),
            "youden_T": metrics_block(y_val, val_scores, t_youden),
        },
        "combined_200": {
            "default_T_0.5": metrics_block(y_all, all_scores, 0.5),
            "youden_T": metrics_block(y_all, all_scores, t_youden),
        },
    }
    return pipeline, report


def train_profile_heads(
    reports_by_profile: dict[str, list[dict]],
    all_reports: list[dict],
    C: float,
) -> tuple[dict[str, Pipeline], dict[str, Any]]:
    pipelines: dict[str, Pipeline] = {}
    per_profile: dict[str, Any] = {}
    thresholds: dict[str, float] = {}

    for profile in PROFILES:
        reports = reports_by_profile[profile]
        X, y, _ = build_dataset(reports, profile=profile)
        pipe = make_pipeline(C)
        pipe.fit(X, y)
        pipelines[profile] = pipe
        block = eval_profile_block(pipe, X, y)
        thresholds[profile] = float(block["best_threshold"])
        per_profile[profile] = block

    X_all, y_all, paths_all = build_dataset(all_reports)
    routed_scores = np.zeros(len(y_all), dtype=np.float64)
    for i, fn in enumerate(paths_all):
        prof = profile_from_filename(fn)
        if prof not in pipelines:
            prof = "ffpp_vox"
        routed_scores[i] = predict_proba_fake(pipelines[prof], X_all[i : i + 1])[0]

    combined_t = best_threshold_youden(y_all, routed_scores)
    report = {
        "strategy": "profile",
        "per_profile": per_profile,
        "thresholds": {k: round(v, 4) for k, v in thresholds.items()},
        "combined_200_routed": {
            "default_T_0.5": metrics_block(y_all, routed_scores, 0.5),
            "youden_T": metrics_block(y_all, routed_scores, combined_t),
            "best_threshold": round(combined_t, 4),
        },
        "cross_domain_note": (
            "ffpp head on celebdf / celebdf head on ffpp not evaluated here; "
            "deploy routes by filename profile."
        ),
    }
    return pipelines, report


def train_combined(all_reports: list[dict], C: float) -> tuple[Pipeline, dict[str, Any]]:
    X, y, _ = build_dataset(all_reports)
    pipeline = make_pipeline(C)
    pipeline.fit(X, y)
    scores = predict_proba_fake(pipeline, X)
    t = best_threshold_youden(y, scores)
    report = {
        "strategy": "combined",
        "train_n": int(y.size),
        "in_sample": {
            "default_T_0.5": metrics_block(y, scores, 0.5),
            "youden_T": metrics_block(y, scores, t),
            "best_threshold": round(t, 4),
        },
    }
    return pipeline, report


def main() -> None:
    parser = argparse.ArgumentParser(description="GMFlow learned head (LR on flow features)")
    parser.add_argument("--root", default=".", help="forenShield-ai root")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--C", type=float, default=0.1, help="LR inverse regularization (lower = stronger)")
    parser.add_argument(
        "--strategy",
        choices=("profile", "pooled", "combined"),
        default="profile",
        help="profile=per-domain heads (recommended for celebdf); pooled=ffpp→celeb val",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else root / "docs" / "notebooks" / "output" / ".gmflow_cache"
    )
    ensure_cache(cache_dir, download=args.download)

    reports_by_profile = {
        p: load_json_reports(cache_dir / p / "json") for p in PROFILES
    }
    all_reports = reports_by_profile["ffpp_vox"] + reports_by_profile["celebdf"]

    base_meta = {
        "model": "gmflow_learned_head_lr",
        "feature_dim": len(feature_names()),
        "feature_names": feature_names(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_C": args.C,
    }

    model_dir = root / "models" / "test" / "video" / "optical-flow" / "gmflow" / "v1.0.0"
    model_dir.mkdir(parents=True, exist_ok=True)
    eval_path = root / "results" / "eval" / "gmflow_learned_head_eval.json"

    if args.strategy == "profile":
        pipelines, strat_report = train_profile_heads(reports_by_profile, all_reports, args.C)
        for profile, pipe in pipelines.items():
            joblib.dump(pipe, model_dir / f"gmflow_learned_head_{profile}.joblib")
        joblib.dump(pipelines, model_dir / "gmflow_learned_head_profile_bundle.joblib")
        eval_report = {**base_meta, **strat_report}
        primary_path = model_dir / "gmflow_learned_head_profile_bundle.joblib"
    elif args.strategy == "pooled":
        pipeline, strat_report = train_pooled(
            reports_by_profile["ffpp_vox"],
            reports_by_profile["celebdf"],
            all_reports,
            args.C,
        )
        joblib.dump(pipeline, model_dir / "gmflow_learned_head.joblib")
        eval_report = {**base_meta, **strat_report}
        primary_path = model_dir / "gmflow_learned_head.joblib"
    else:
        pipeline, strat_report = train_combined(all_reports, args.C)
        joblib.dump(pipeline, model_dir / "gmflow_learned_head_combined.joblib")
        eval_report = {**base_meta, **strat_report}
        primary_path = model_dir / "gmflow_learned_head_combined.joblib"

    meta_path = model_dir / "gmflow_learned_head.meta.json"
    meta_path.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(eval_report, indent=2), encoding="utf-8")

    print(f"=== GMFlow learned head ({args.strategy}, C={args.C}) ===")
    if args.strategy == "profile":
        for p in PROFILES:
            m = eval_report["per_profile"][p]["youden_T"]
            print(f"  {p}: AUC={m['auc']} fake_rec={m['fake_recall']} T={eval_report['thresholds'][p]}")
        c = eval_report["combined_200_routed"]["youden_T"]
        print(f"  combined (routed): AUC={c['auc']} fake_rec={c['fake_recall']}")
    elif args.strategy == "pooled":
        print(f"  train AUC={eval_report['train']['youden_T']['auc']}")
        print(f"  val   AUC={eval_report['val']['youden_T']['auc']}")
        print(f"  combined AUC={eval_report['combined_200']['youden_T']['auc']}")
    else:
        m = eval_report["in_sample"]["youden_T"]
        print(f"  in-sample AUC={m['auc']} fake_rec={m['fake_recall']}")
    print(f"model: {primary_path}")
    print(f"eval:  {eval_path}")


if __name__ == "__main__":
    main()
