#!/usr/bin/env python3
"""Smoke-test current late fusion on a 5fake+5real slice of Golden-200 from S3.

Requires working AWS credentials (AWS_PROFILE or default).

Usage (from ai/):
  set AWS_PROFILE=team4
  set KMP_DUPLICATE_LIB_OK=TRUE
  ..\\.venv\\Scripts\\python.exe scripts/eval/smoke_golden200_fusion_v4c.py
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[2]
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
from app.services.late_fusion import fuse_scores_gated, load_fusion_config

BUCKET = os.getenv("EVIDENCE_BUCKET", "forenshield-evidence-877044078824")
AWS_PROFILE = os.getenv("AWS_PROFILE", "team4")
SEED = int(os.getenv("SMOKE_SEED", "42"))
N_PER_CLASS = int(os.getenv("SMOKE_N", "5"))
OUT_DIR = AI_ROOT / "results" / "eval" / "golden200_smoke_v4c"
LOCAL_DIR = AI_ROOT / "data" / "pull" / "golden200_smoke"
PROFILES = ("celebdf", "ffpp_vox")


def _utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _aws_base() -> list[str]:
    cmd = ["aws"]
    if AWS_PROFILE:
        cmd += ["--profile", AWS_PROFILE]
    return cmd


def s3_list(prefix: str) -> list[str]:
    cmd = _aws_base() + [
        "s3",
        "ls",
        f"s3://{BUCKET}/{prefix.rstrip('/')}/",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"aws s3 ls failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
        )
    keys: list[str] = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[-1]
        if name.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
            keys.append(f"{prefix.rstrip('/')}/{name}")
    return keys


def s3_cp(key: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return
    cmd = _aws_base() + ["s3", "cp", f"s3://{BUCKET}/{key}", str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"aws s3 cp failed for {key}: {proc.stderr.strip()}")


def pick_samples() -> list[tuple[str, str, str]]:
    """Return list of (label, profile, s3_key)."""
    rng = random.Random(SEED)
    picked: list[tuple[str, str, str]] = []
    for label in ("fake", "real"):
        pool: list[tuple[str, str]] = []
        for profile in PROFILES:
            prefix = f"deepfake/datasets/bench/{profile}/{label}"
            for key in s3_list(prefix):
                pool.append((profile, key))
        if len(pool) < N_PER_CLASS:
            raise RuntimeError(f"Need ≥{N_PER_CLASS} {label} videos, found {len(pool)}")
        chosen = rng.sample(pool, N_PER_CLASS)
        for profile, key in chosen:
            picked.append((label, profile, key))
    return picked


def main() -> None:
    print(f"profile={AWS_PROFILE} bucket={BUCKET} n_per_class={N_PER_CLASS} seed={SEED}")
    print(f"fusion={os.environ['FUSION_CONFIG_PATH']}")

    samples = pick_samples()
    local_items: list[tuple[str, str, Path]] = []
    for label, profile, key in samples:
        name = Path(key).name
        dest = LOCAL_DIR / profile / label / name
        print(f"download {key} -> {dest}")
        s3_cp(key, dest)
        local_items.append((label, profile, dest))

    settings = load_model_settings()
    runtime = InferRuntime(settings)
    fusion = load_fusion_config(Path(os.environ["FUSION_CONFIG_PATH"]))
    thr = fusion.threshold

    rows = []
    tp = tn = fp = fn = 0
    for label, profile, path in local_items:
        print(f"\n[{label}/{profile}] {path.name}")
        t0 = time.time()
        modules = runtime.analyze_modules(path)
        elapsed = round(time.time() - t0, 2)
        by = {m.module: m for m in modules}
        cnn = by.get("cnn")
        ts = by.get("temporal")
        gmf = by.get("optical")
        s_cnn = cnn.fake_score if cnn else None
        s_ts = ts.fake_score if ts else None
        s_gmf = gmf.fake_score if gmf else None

        row: dict = {
            "file": path.name,
            "profile": profile,
            "gt": label,
            "elapsed_sec": elapsed,
            "cnn_status": getattr(cnn, "status", None),
            "temporal_status": getattr(ts, "status", None),
            "optical_status": getattr(gmf, "status", None),
            "cnn": None if s_cnn is None else round(float(s_cnn), 4),
            "temporal": None if s_ts is None else round(float(s_ts), 4),
            "optical": None if s_gmf is None else round(float(s_gmf), 4),
        }
        if s_cnn is None:
            row["skipped"] = True
            row["reason"] = f"cnn_status={row['cnn_status']}"
            print(f"  skip: {row['reason']}")
            rows.append(row)
            continue

        score, meta = fuse_scores_gated(
            s_cnn=float(s_cnn),
            s_temporal=float(s_ts or 0.0),
            s_optical=float(s_gmf or 0.0),
            config=fusion,
        )
        pred = "fake" if score >= thr else "real"
        flags = [k for k, v in meta.items() if v is True]
        row.update(
            {
                "skipped": False,
                "fusion": score,
                "pred": pred,
                "threshold": thr,
                "ok": pred == label,
                "gates": flags,
            }
        )
        print(
            f"  cnn={row['cnn']} ts={row['temporal']} gmf={row['optical']} "
            f"fusion={score:.4f}({pred}) gates={flags} {'OK' if row['ok'] else 'MISS'}"
        )
        if label == "fake" and pred == "fake":
            tp += 1
        elif label == "real" and pred == "real":
            tn += 1
        elif label == "real":
            fp += 1
        else:
            fn += 1
        rows.append(row)

    labeled = [r for r in rows if not r.get("skipped")]
    n = len(labeled)
    summary = {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round((tp + tn) / n, 4) if n else None,
        "fake_recall": round(tp / (tp + fn), 4) if (tp + fn) else None,
        "real_recall": round(tn / (tn + fp), 4) if (tn + fp) else None,
        "threshold": thr,
        "fusion_version": fusion.fusion_version,
    }
    report = {
        "generated_at": _utc(),
        "seed": SEED,
        "n_per_class": N_PER_CLASS,
        "bucket": BUCKET,
        "samples": [{"gt": a, "profile": b, "key": c} for a, b, c in samples],
        "summary": summary,
        "rows": rows,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
