#!/usr/bin/env python3
"""Recompute field_late_fusion_v4b report rows with current fusion config (cache-only)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(AI_ROOT))

from app.services.late_fusion import fuse_scores_gated, load_fusion_config
from scripts.eval.eval_field_late_fusion import legacy_config, score_row, summarize

REPORT = AI_ROOT / "results/eval/field_late_fusion_v4b/report.json"
CONFIG = AI_ROOT / "config/fusion_v4_ts_gated.json"


def main() -> None:
    prev = json.loads(REPORT.read_text(encoding="utf-8"))
    cfg_new = load_fusion_config(CONFIG)
    cfg_old = legacy_config(cfg_new)

    rows_out = []
    for row in prev["rows"]:
        if row.get("skipped"):
            rows_out.append(row)
            continue
        modules = {
            "cnn": {"status": "ok", "fake_score": row["cnn"]},
            "temporal": {"status": "ok", "fake_score": row["temporal"]},
            "optical": {"status": "ok", "fake_score": row["optical"]},
        }
        scored = score_row(row["gt"], modules, cfg_new, cfg_old)
        scored["file"] = row.get("file")
        scored["rel"] = row.get("rel")
        scored["elapsed_sec"] = row.get("elapsed_sec")
        rows_out.append(scored)

    report = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "fusion_config": str(CONFIG),
        "fusion_version": cfg_new.fusion_version,
        "threshold": cfg_new.threshold,
        "summary_new": summarize(rows_out, "pred_new"),
        "summary_old": summarize(rows_out, "pred_old"),
        "gt_counts": prev.get("gt_counts"),
        "rows": rows_out,
        "failures": prev.get("failures", []),
        "note": "Recomputed from previous module scores; models were not re-inferred.",
    }
    out = REPORT.with_name("report_v4c.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary_new": report["summary_new"], "summary_old": report["summary_old"]}, indent=2))
    print(f"wrote {REPORT}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
