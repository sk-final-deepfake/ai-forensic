#!/usr/bin/env python3
"""Offline field eval: YouTube/Kakao videos → module scores → late fusion.

Runs Xception / TimeSformer / GMFlow then compares:
  - legacy_v4 (dual-high cap 0.60 + unconditional GMF veto)
  - current ops config (fusion_v4_ts_gated.json after agreement-first edits)

Usage (from ai/):
  set KMP_DUPLICATE_LIB_OK=TRUE
  ..\\.venv\\Scripts\\python.exe scripts/eval/eval_field_late_fusion.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
if str(AI_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_ROOT))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("AI_ROOT", str(AI_ROOT))
os.environ.setdefault(
    "XCEPTION_WEIGHTS",
    str(AI_ROOT / "models/test/video/xception/v1.0.0/xception_finetuned_celeb1k.pth"),
)
os.environ.setdefault(
    "TIMESFORMER_WEIGHTS",
    str(AI_ROOT / "models/test/video/timesformer/v1.0.0/timesformer_finetuned_celeb1k.pth"),
)
os.environ.setdefault("FUSION_CONFIG_PATH", str(AI_ROOT / "config/fusion_v4_ts_gated.json"))

from app.core.model_settings import load_model_settings
from app.services.infer_bridge import InferRuntime
from app.services.late_fusion import FusionConfig, fuse_scores_gated, load_fusion_config

VIDEO_ROOTS = [
    AI_ROOT / "data/test/video/youtube-fresh",
    AI_ROOT / "data/test/video/youtube-pilot",
    AI_ROOT / "data/test/video/kakao_youtube-fresh",
]
OUT_DIR = AI_ROOT / "results" / "eval" / "field_late_fusion_v4b"
CACHE_DIR = OUT_DIR / "module_cache"


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def discover_videos() -> list[tuple[Path, str]]:
    """Return (path, gt_label) where gt_label in {fake, real, unknown}."""
    items: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for root in VIDEO_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in {".mp4", ".webm", ".mov", ".avi", ".mkv"}:
                continue
            key = str(path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            parent = path.parent.name.lower()
            if parent in {"fake", "real"}:
                gt = parent
            elif path.name.lower().startswith("ai_") or path.name.lower().startswith("fake"):
                gt = "fake"
            elif path.name.lower().startswith("real_"):
                gt = "real"
            else:
                gt = "unknown"
            items.append((path, gt))
    return items


def legacy_config(current: FusionConfig) -> FusionConfig:
    """Replay the previous aggressive dual-high-cap / unconditional GMF veto policy."""
    payload = {
        "fusion_version": "fusion-v4-ts-gated-legacy-cap",
        "method": "gated",
        "weights": dict(current.weights),
        "threshold": current.threshold,
        "module_thresholds": dict(current.module_thresholds),
        "risk_levels": dict(current.risk_levels),
        "suspicious_segment": dict(current.suspicious_segment),
        "model_versions": dict(current.model_versions),
        "gating": {
            "cnn_ambiguous_lo": 0.40,
            "cnn_ambiguous_hi": 0.78,
            "ts_rescue_min": 0.60,
            "ts_rescue_margin": 0.15,
            "ts_rescue_cnn_weight": 0.20,
            "ts_rescue_temporal_weight": 0.80,
            "ts_rescue_strong_min": 0.75,
            "ts_rescue_strong_cnn_weight": 0.10,
            "ts_rescue_strong_temporal_weight": 0.90,
            "ts_base_weight": 0.06,
            "ts_base_min": 0.45,
            "ts_base_requires_ambiguous_cnn": True,
            "gmflow_veto_max": 0.30,
            "gmflow_veto_max_ts": 1.01,  # always allow hard veto (legacy)
            "cnn_discount_when_gmf_low": 0.18,
            "gmflow_soft_veto_cnn_min": 0.62,
            "gmflow_soft_veto_max": 0.15,
            "cnn_soft_discount": 0.10,
            "ambiguous_boost": 0.04,
            "ambiguous_boost_cnn_max": 0.68,
            "ambiguous_cnn_floor_min": 0.58,
            "ambiguous_gmf_max": 0.30,
            "ambiguous_cnn_floor": False,
            "dual_high_cnn_min": 0.85,
            "dual_high_ts_min": 0.85,
            "dual_high_gmf_max": 0.48,
            "dual_high_fusion_cap": 0.60,
            "dual_high_agree_boost": 0.0,
            "dual_module_rescue": True,
            "dual_module_ts_min": 0.60,
            "dual_module_gmf_min": 0.50,
            "dual_module_ts_weight": 0.70,
            "dual_module_gmf_weight": 0.30,
        },
    }
    return FusionConfig.from_dict(payload)


def cache_path(video: Path) -> Path:
    return CACHE_DIR / f"{video.stem}.modules.json"


def load_or_infer(runtime: InferRuntime, video: Path) -> dict:
    cpath = cache_path(video)
    if cpath.is_file():
        return json.loads(cpath.read_text(encoding="utf-8"))

    t0 = time.time()
    modules = runtime.analyze_modules(video)
    elapsed = time.time() - t0
    by = {m.module: m for m in modules}
    payload = {
        "file": video.name,
        "path": str(video),
        "elapsed_sec": round(elapsed, 2),
        "cnn": {
            "status": by["cnn"].status if "cnn" in by else None,
            "fake_score": by["cnn"].fake_score if "cnn" in by else None,
        },
        "temporal": {
            "status": by["temporal"].status if "temporal" in by else None,
            "fake_score": by["temporal"].fake_score if "temporal" in by else None,
        },
        "optical": {
            "status": by["optical"].status if "optical" in by else None,
            "fake_score": by["optical"].fake_score if "optical" in by else None,
        },
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def score_row(gt: str, modules: dict, cfg_new: FusionConfig, cfg_old: FusionConfig) -> dict | None:
    s_cnn = modules["cnn"]["fake_score"]
    s_ts = modules["temporal"]["fake_score"]
    s_gmf = modules["optical"]["fake_score"]
    if s_cnn is None:
        return {
            "gt": gt,
            "skipped": True,
            "reason": f"cnn_status={modules['cnn']['status']}",
            "cnn": None,
            "temporal": s_ts,
            "optical": s_gmf,
        }
    s_ts_v = float(s_ts or 0.0)
    s_gmf_v = float(s_gmf or 0.0)
    new_score, new_meta = fuse_scores_gated(
        s_cnn=float(s_cnn), s_temporal=s_ts_v, s_optical=s_gmf_v, config=cfg_new
    )
    old_score, old_meta = fuse_scores_gated(
        s_cnn=float(s_cnn), s_temporal=s_ts_v, s_optical=s_gmf_v, config=cfg_old
    )
    thr = cfg_new.threshold
    return {
        "gt": gt,
        "skipped": False,
        "cnn": round(float(s_cnn), 4),
        "temporal": round(s_ts_v, 4),
        "optical": round(s_gmf_v, 4),
        "fusion_new": new_score,
        "fusion_old": old_score,
        "pred_new": "fake" if new_score >= thr else "real",
        "pred_old": "fake" if old_score >= thr else "real",
        "threshold": thr,
        "meta_new": {k: v for k, v in new_meta.items() if v not in (False, None)},
        "meta_old": {k: v for k, v in old_meta.items() if v not in (False, None)},
    }


def summarize(rows: list[dict], key_pred: str) -> dict:
    labeled = [r for r in rows if not r.get("skipped") and r.get("gt") in {"fake", "real"}]
    tp = sum(1 for r in labeled if r["gt"] == "fake" and r[key_pred] == "fake")
    tn = sum(1 for r in labeled if r["gt"] == "real" and r[key_pred] == "real")
    fp = sum(1 for r in labeled if r["gt"] == "real" and r[key_pred] == "fake")
    fn = sum(1 for r in labeled if r["gt"] == "fake" and r[key_pred] == "real")
    n_fake = sum(1 for r in labeled if r["gt"] == "fake")
    n_real = sum(1 for r in labeled if r["gt"] == "real")
    return {
        "n": len(labeled),
        "n_fake": n_fake,
        "n_real": n_real,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fake_recall": round(tp / n_fake, 4) if n_fake else None,
        "real_recall": round(tn / n_real, 4) if n_real else None,
        "accuracy": round((tp + tn) / len(labeled), 4) if labeled else None,
        "precision_fake": round(tp / (tp + fp), 4) if (tp + fp) else None,
    }


def main() -> int:
    videos = discover_videos()
    if not videos:
        print("No videos found under youtube-fresh / youtube-pilot / kakao_youtube-fresh")
        return 1

    settings = load_model_settings()
    print(f"device={settings.infer_device}")
    print(f"xception={settings.xception_weights} exists={settings.xception_weights.is_file()}")
    print(f"timesformer={settings.timesformer_weights} exists={settings.timesformer_weights.is_file()}")
    print(f"videos={len(videos)}")

    cfg_new = load_fusion_config(settings.fusion_config_path)
    cfg_old = legacy_config(cfg_new)
    runtime = InferRuntime(settings)

    rows: list[dict] = []
    failures: list[dict] = []
    for idx, (video, gt) in enumerate(videos, 1):
        print(f"[{idx}/{len(videos)}] {gt} {video.name}", flush=True)
        try:
            modules = load_or_infer(runtime, video)
            row = score_row(gt, modules, cfg_new, cfg_old)
            if row is None:
                continue
            row["file"] = video.name
            row["rel"] = str(video.relative_to(AI_ROOT))
            row["elapsed_sec"] = modules.get("elapsed_sec")
            rows.append(row)
            if row.get("skipped"):
                print(f"  skip: {row.get('reason')}", flush=True)
            else:
                print(
                    f"  cnn={row['cnn']} ts={row['temporal']} gmf={row['optical']} "
                    f"old={row['fusion_old']}({row['pred_old']}) "
                    f"new={row['fusion_new']}({row['pred_new']})",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001
            failures.append({"file": video.name, "gt": gt, "error": str(exc)})
            print(f"  FAIL: {exc}", flush=True)
            traceback.print_exc()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": _utc(),
        "fusion_config": str(settings.fusion_config_path),
        "device": settings.infer_device,
        "threshold": cfg_new.threshold,
        "summary_new": summarize(rows, "pred_new"),
        "summary_old": summarize(rows, "pred_old"),
        "gt_counts": dict(Counter(gt for _, gt in videos)),
        "rows": rows,
        "failures": failures,
    }
    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_new": report["summary_new"], "summary_old": report["summary_old"]}, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
