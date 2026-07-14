#!/usr/bin/env python3
"""Offline grid search of fusion gates on field_late_fusion_v4b report rows."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AI_ROOT))

from app.services.late_fusion import FusionConfig, GatingConfig, fuse_scores_gated, load_fusion_config

REPORT = AI_ROOT / "results/eval/field_late_fusion_v4b/report.json"
CONFIG = AI_ROOT / "config/fusion_v4_ts_gated.json"


def eval_cfg(cfg: FusionConfig, rows: list[dict]) -> dict:
    tp = tn = fp = fn = 0
    misses: list[tuple] = []
    for r in rows:
        s, _meta = fuse_scores_gated(
            s_cnn=float(r["cnn"]),
            s_temporal=float(r["temporal"]),
            s_optical=float(r["optical"]),
            config=cfg,
        )
        pred = "fake" if s >= cfg.threshold else "real"
        gt = r["gt"]
        if gt == "fake" and pred == "fake":
            tp += 1
        elif gt == "real" and pred == "real":
            tn += 1
        elif gt == "real" and pred == "fake":
            fp += 1
            misses.append((r["file"], gt, pred, round(s, 4), r["cnn"], r["temporal"], r["optical"]))
        else:
            fn += 1
            misses.append((r["file"], gt, pred, round(s, 4), r["cnn"], r["temporal"], r["optical"]))
    n = tp + tn + fp + fn
    return {
        "acc": (tp + tn) / n if n else 0.0,
        "f1": (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fake_rec": tp / (tp + fn) if tp + fn else 0.0,
        "real_rec": tn / (tn + fp) if tn + fp else 0.0,
        "prec": tp / (tp + fp) if tp + fp else 0.0,
        "thr": cfg.threshold,
        "misses": misses,
    }


def with_gating(base: FusionConfig, *, threshold: float | None = None, **gating_kwargs) -> FusionConfig:
    g0 = base.gating or GatingConfig()
    g = replace(g0, **gating_kwargs)
    return FusionConfig(
        fusion_version=base.fusion_version,
        method=base.method,
        weights=dict(base.weights),
        threshold=float(threshold if threshold is not None else base.threshold),
        module_thresholds=dict(base.module_thresholds),
        risk_levels=dict(base.risk_levels),
        suspicious_segment=dict(base.suspicious_segment),
        model_versions=dict(base.model_versions),
        gating=g,
    )


def main() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = [r for r in report["rows"] if not r.get("skipped") and r.get("gt") in {"fake", "real"}]
    base = load_fusion_config(CONFIG)

    print("BASE", {k: v for k, v in eval_cfg(base, rows).items() if k != "misses"})

    candidates: list[tuple] = []
    for thr in [0.55, 0.58, 0.60, 0.6051, 0.62, 0.635, 0.65, 0.68, 0.70]:
        for agree in [0.0, 0.04]:
            for veto_max in [0.30, 0.36, 0.42, 0.50]:
                for veto_d in [0.18, 0.28, 0.38, 0.48]:
                    for veto_ts in [0.50, 0.20, 1.01]:
                        for soft_d in [0.05, 0.10, 0.20]:
                            for amb_boost in [0.0, 0.04]:
                                for soft_veto_max in [0.15, 0.25]:
                                    cfg = with_gating(
                                        base,
                                        threshold=thr,
                                        dual_high_agree_boost=agree,
                                        gmflow_veto_max=veto_max,
                                        cnn_discount_when_gmf_low=veto_d,
                                        gmflow_veto_max_ts=veto_ts,
                                        cnn_soft_discount=soft_d,
                                        ambiguous_boost=amb_boost,
                                        gmflow_soft_veto_max=soft_veto_max,
                                    )
                                    m = eval_cfg(cfg, rows)
                                    candidates.append(
                                        (
                                            m["acc"],
                                            m["f1"],
                                            -m["fp"],
                                            -m["fn"],
                                            m,
                                            {
                                                "thr": thr,
                                                "agree": agree,
                                                "veto_max": veto_max,
                                                "veto_d": veto_d,
                                                "veto_ts": veto_ts,
                                                "soft_d": soft_d,
                                                "amb_boost": amb_boost,
                                                "soft_veto_max": soft_veto_max,
                                            },
                                        )
                                    )

    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    print(f"searched={len(candidates)}")
    print("TOP8 current-knobs:")
    seen = set()
    shown = 0
    for acc, f1, nfp, nfn, m, params in candidates:
        key = (m["tp"], m["tn"], m["fp"], m["fn"], params["thr"])
        if key in seen:
            continue
        seen.add(key)
        print(
            f"  acc={acc:.4f} f1={f1:.4f} tp={m['tp']} tn={m['tn']} fp={m['fp']} fn={m['fn']} "
            f"params={params}"
        )
        for miss in m["misses"][:8]:
            print(f"    miss {miss}")
        shown += 1
        if shown >= 8:
            break

    # Also try ts-disagree via local monkey-patch simulation:
    # discount CNN when cnn high & ts weak, optionally requiring gmf not-high.
    print("\n--- synthetic ts_disagree search (manual formula) ---")
    synth_best = []
    for thr in [0.58, 0.6051, 0.62, 0.65]:
        for cnn_min in [0.78, 0.85]:
            for ts_max in [0.05, 0.12, 0.20]:
                for disc in [0.20, 0.30, 0.40, 0.50]:
                    for require_gmf_max in [None, 0.40, 0.55, 1.01]:
                        for agree in [0.0]:
                            tp = tn = fp = fn = 0
                            misses = []
                            for r in rows:
                                cnn, ts, gmf = float(r["cnn"]), float(r["temporal"]), float(r["optical"])
                                # start from current ops gating roughly: hard veto then ts_disagree
                                cnn_eff = cnn
                                if cnn >= 0.78 and gmf < 0.36 and ts < 0.50:
                                    cnn_eff = cnn_eff * (1.0 - 0.28)
                                if (
                                    cnn >= cnn_min
                                    and ts < ts_max
                                    and (require_gmf_max is None or gmf < require_gmf_max)
                                ):
                                    cnn_eff = cnn_eff * (1.0 - disc)
                                # soft veto-ish
                                if 0.62 <= cnn < 0.78 and gmf < 0.15 and ts < 0.12:
                                    cnn_eff = cnn_eff * 0.90
                                score = 0.90 * cnn_eff + 0.10 * gmf
                                if agree and cnn >= 0.78 and ts >= 0.60:
                                    score = min(1.0, score + agree)
                                pred = "fake" if score >= thr else "real"
                                gt = r["gt"]
                                if gt == "fake" and pred == "fake":
                                    tp += 1
                                elif gt == "real" and pred == "real":
                                    tn += 1
                                elif gt == "real" and pred == "fake":
                                    fp += 1
                                    misses.append((r["file"], round(score, 4), cnn, ts, gmf))
                                else:
                                    fn += 1
                                    misses.append((r["file"], round(score, 4), cnn, ts, gmf))
                            n = tp + tn + fp + fn
                            acc = (tp + tn) / n
                            f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
                            synth_best.append(
                                (
                                    acc,
                                    f1,
                                    -fp,
                                    -fn,
                                    {
                                        "thr": thr,
                                        "cnn_min": cnn_min,
                                        "ts_max": ts_max,
                                        "disc": disc,
                                        "gmf_max": require_gmf_max,
                                        "tp": tp,
                                        "tn": tn,
                                        "fp": fp,
                                        "fn": fn,
                                    },
                                    misses,
                                )
                            )
    synth_best.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
    seen = set()
    shown = 0
    for acc, f1, _, _, p, misses in synth_best:
        key = (p["tp"], p["tn"], p["fp"], p["fn"], p["thr"], p["disc"], p["ts_max"], p["gmf_max"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  acc={acc:.4f} f1={f1:.4f} {p}")
        for m in misses[:10]:
            print(f"    miss {m}")
        shown += 1
        if shown >= 10:
            break


if __name__ == "__main__":
    main()
