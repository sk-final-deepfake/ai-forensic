"""Score GMFlow JSON with best sweep model (learned head or heuristic)."""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib

EVAL_DIR = Path(__file__).resolve().parent
INFER_DIR = EVAL_DIR.parent / "infer"
for p in (EVAL_DIR, INFER_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gmflow_feature_extract import features_from_report, normalize_report, profile_from_filename
from gmflow_scoring import enrich_motion_scores

DEFAULT_MODEL = Path("models/test/video/optical-flow/gmflow/v1.0.0/gmflow_learned_head.joblib")
DEFAULT_BUNDLE = Path("models/test/video/optical-flow/gmflow/v1.0.0/gmflow_learned_head_profile_bundle.joblib")
DEFAULT_META = Path("models/test/video/optical-flow/gmflow/v1.0.0/gmflow_best.meta.json")
LEGACY_META = Path("models/test/video/optical-flow/gmflow/v1.0.0/gmflow_learned_head.meta.json")


def load_scoring_config(root: Path) -> tuple[Any, dict]:
    for meta_path in (root / DEFAULT_META, root / LEGACY_META):
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        winner = str(meta.get("winner_name") or "")
        if meta.get("heuristic") or winner.startswith("heuristic"):
            return "heuristic", meta
        if meta.get("evaluation_protocol") == "holdout":
            # Never unpickle for hold-out — caller retrains on ffpp_vox train split.
            return "holdout_retrain", meta
        if winner.startswith(("lr_profile", "hgb_profile", "rf_profile", "ensemble")) or meta.get("strategy") == "profile":
            bundle = root / DEFAULT_BUNDLE
            if bundle.is_file():
                return joblib.load(bundle), meta
        for rel in (DEFAULT_MODEL, Path("models/test/video/optical-flow/gmflow/v1.0.0/gmflow_learned_head_combined.joblib")):
            model_path = root / rel
            if model_path.is_file():
                return joblib.load(model_path), meta
    raise FileNotFoundError(f"GMFlow scoring meta not found under {root / DEFAULT_META.parent}")


def load_learned_head(root: Path) -> tuple[Any, dict]:
    return load_scoring_config(root)


def resolve_threshold(meta: dict, profile: str | None) -> float:
    thresholds = meta.get("thresholds") or {}
    if profile and profile in thresholds:
        return float(thresholds[profile])
    if "combined_200" in thresholds:
        return float(thresholds["combined_200"])
    if meta.get("strategy") == "combined":
        return float(meta["in_sample"]["best_threshold"])
    if meta.get("train"):
        return float(meta["train"]["best_threshold"])
    return 0.5


def pipeline_for_report(pipeline_or_bundle: Any, report: dict[str, Any]) -> Any:
    if not isinstance(pipeline_or_bundle, dict):
        return pipeline_or_bundle
    fn = str(report.get("file") or report.get("video_id") or report.get("filename") or "")
    profile = profile_from_filename(fn)
    if profile and profile in pipeline_or_bundle:
        return pipeline_or_bundle[profile]
    return pipeline_or_bundle.get("celebdf") or next(iter(pipeline_or_bundle.values()))


def fake_score_from_report(report: dict[str, Any], scorer: Any, meta: dict) -> float | None:
    if scorer == "heuristic":
        heur = meta.get("heuristic") or {}
        row = deepcopy(report)
        normalize_report(row)
        enrich_motion_scores(
            [row],
            threshold=0.5,
            per_profile_cohort=bool(heur.get("per_profile_cohort")),
            signal_weights=heur.get("signal_weights"),
        )
        val = row.get("motion_anomaly_score")
        return float(val) if val is not None else None

    feat = features_from_report(report)
    if feat is None:
        return None
    pipeline = pipeline_for_report(scorer, report)
    proba = pipeline.predict_proba(feat.reshape(1, -1))
    classes = list(pipeline.named_steps["clf"].classes_)
    return float(proba[0, classes.index(1)])


def score_report(report: dict[str, Any], scorer: Any, meta: dict, threshold: float) -> dict[str, Any]:
    fn = str(report.get("file") or "")
    profile = profile_from_filename(fn)
    t = resolve_threshold(meta, profile) if meta.get("thresholds") else threshold
    score = fake_score_from_report(report, scorer, meta)
    out = dict(report)
    if score is None:
        out["fake_score"] = None
        out["pred_label"] = None
        out["score_source"] = meta.get("score_source", "gmflow_learned_head")
        return out
    out["fake_score"] = round(score, 6)
    out["motion_anomaly_score"] = out["fake_score"]
    out["pred_label"] = "fake" if score >= t else "real"
    out["score_source"] = meta.get("score_source", "gmflow_learned_head")
    return out
