#!/usr/bin/env python3
"""Bundle benchmark infer results into one JSON report with a unified item schema."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.1"


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest_rows(dataset_dir: Path) -> list[dict]:
    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.is_file():
        raw = load_json(manifest_path)
        if isinstance(raw, list):
            return [dict(row) for row in raw]
    rows: list[dict] = []
    for mp4 in sorted(dataset_dir.glob("*.mp4")):
        rows.append({"file": mp4.name})
    return rows


def normalize_dataset_entry(
    row: dict,
    *,
    ground_truth_label: str,
    dataset_source: str,
    local_dir: Path,
    profile: str = "ffpp_vox",
) -> dict:
    """Map fake/real manifest rows into one comparable schema."""
    filename = row["file"]
    local_path = str((local_dir / filename).resolve())

    if profile == "dfdc":
        origin_path = row.get("source") or row.get("hf_path") or row.get("source_path")
        origin = {
            "type": "dfdc",
            "manipulation": None,
            "video_id": None,
            "full_duration_sec": row.get("duration_sec"),
            "clip_start_sec": None,
            "clip_duration_sec": row.get("duration_sec"),
        }
        duration_sec = row.get("duration_sec")
    elif profile == "celebdf":
        origin_path = row.get("source") or row.get("source_path")
        origin = {
            "type": "celeb_df_v2",
            "manipulation": None,
            "video_id": None,
            "full_duration_sec": row.get("duration_sec"),
            "clip_start_sec": None,
            "clip_duration_sec": row.get("duration_sec"),
        }
        duration_sec = row.get("duration_sec")
    elif ground_truth_label == "fake":
        duration_sec = row.get("duration_sec")
        origin_path = row.get("source_path")
        origin = {
            "type": "faceforensics_deepfake",
            "manipulation": row.get("manipulation"),
            "video_id": None,
            "full_duration_sec": duration_sec,
            "clip_start_sec": None,
            "clip_duration_sec": duration_sec,
        }
    else:
        clip_duration = row.get("clip_duration_sec")
        duration_sec = clip_duration
        video_id = row.get("video_id")
        origin_path = f"https://www.youtube.com/watch?v={video_id}" if video_id else None
        origin = {
            "type": "voxceleb_youtube",
            "manipulation": None,
            "video_id": video_id,
            "full_duration_sec": row.get("full_duration_sec"),
            "clip_start_sec": row.get("clip_start_sec"),
            "clip_duration_sec": clip_duration,
        }

    return {
        "file": filename,
        "ground_truth_label": ground_truth_label,
        "dataset_source": row.get("dataset_source", row.get("dataset", dataset_source)),
        "duration_sec": duration_sec,
        "local_path": local_path,
        "origin_path": origin_path,
        "origin": origin,
    }


def normalize_inference(infer_item: dict) -> dict:
    return {
        "status": infer_item.get("status"),
        "fake_score": infer_item.get("fake_score"),
        "pred_label": infer_item.get("pred_label"),
        "correct": infer_item.get("correct"),
        "frames_used": infer_item.get("frames_used"),
        "analyzed_at": infer_item.get("analyzed_at"),
        "score_breakdown": infer_item.get("score_breakdown"),
    }


def merge_item(dataset_entry: dict, infer_item: dict | None) -> dict:
    merged = dict(dataset_entry)
    if infer_item is None:
        merged["inference"] = {
            "status": "missing",
            "fake_score": None,
            "pred_label": None,
            "correct": None,
            "frames_used": 0,
            "analyzed_at": None,
            "score_breakdown": None,
        }
        return merged
    merged["inference"] = normalize_inference(infer_item)
    return merged


PROFILE_META = {
    "ffpp_vox": {
        "fake_source": "FaceForensics++",
        "real_source": "VoxCeleb",
        "fake_name": "FaceForensics++ DeepFakeDetection (c40, >=60s)",
        "real_name": "VoxCeleb (trimmed long clips)",
    },
    "dfdc": {
        "fake_source": "DFDC",
        "real_source": "DFDC",
        "fake_name": "DFDC subset (fake)",
        "real_name": "DFDC subset (real)",
    },
    "celebdf": {
        "fake_source": "Celeb-DF v2",
        "real_source": "Celeb-DF v2",
        "fake_name": "Celeb-DF v2 Celeb-synthesis (sample)",
        "real_name": "Celeb-DF v2 Celeb-real + YouTube-real (sample)",
    },
}


def build_items(
    fake_dir: Path,
    real_dir: Path,
    infer_items: list[dict],
    *,
    profile: str = "ffpp_vox",
) -> list[dict]:
    infer_by_file = {item["file"]: item for item in infer_items}
    meta = PROFILE_META.get(profile, PROFILE_META["ffpp_vox"])

    parent_manifest = fake_dir.parent / "manifest.json"
    parent_rows: dict[str, dict] = {}
    if parent_manifest.is_file():
        raw = load_json(parent_manifest)
        if isinstance(raw, list):
            parent_rows = {str(row.get("file")): row for row in raw if row.get("file")}

    fake_rows = load_manifest_rows(fake_dir)
    real_rows = load_manifest_rows(real_dir)
    for rows in (fake_rows, real_rows):
        for row in rows:
            parent = parent_rows.get(row["file"])
            if parent:
                row.setdefault("dataset", parent.get("dataset"))
                row.setdefault("source", parent.get("source"))
                row.setdefault("label", parent.get("label"))

    fake_entries = [
        normalize_dataset_entry(
            row,
            ground_truth_label="fake",
            dataset_source=meta["fake_source"],
            local_dir=fake_dir,
            profile=profile,
        )
        for row in fake_rows
    ]
    real_entries = [
        normalize_dataset_entry(
            row,
            ground_truth_label="real",
            dataset_source=meta["real_source"],
            local_dir=real_dir,
            profile=profile,
        )
        for row in real_rows
    ]

    items = [
        merge_item(entry, infer_by_file.get(entry["file"]))
        for entry in fake_entries + real_entries
    ]
    items.sort(key=lambda x: (x["ground_truth_label"], x["file"]))
    return items


def build_inference_summary(
    run_id: str,
    items: list[dict],
    generated_at: str,
    *,
    threshold: float | None = None,
) -> dict:
    summary_items = []
    for item in items:
        inf = item.get("inference") or {}
        breakdown = inf.get("score_breakdown") or {}
        aggregate = breakdown.get("aggregate") or {}
        summary_items.append(
            {
                "file": item["file"],
                "ground_truth_label": item["ground_truth_label"],
                "fake_score": inf.get("fake_score"),
                "pred_label": inf.get("pred_label"),
                "logit_real": aggregate.get("logit_real"),
                "logit_fake": aggregate.get("logit_fake"),
                "prob_real": aggregate.get("prob_real"),
                "prob_fake": aggregate.get("prob_fake"),
                "margin": aggregate.get("margin"),
                "entropy": aggregate.get("entropy"),
                "confidence": aggregate.get("confidence"),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "threshold": threshold,
        "count": len(summary_items),
        "items": summary_items,
    }


def write_inference_summary(infer_dir: Path, summary: dict) -> Path:
    out_dir = infer_dir / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "infer_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def write_normalized_manifests(
    infer_dir: Path,
    fake_items: list[dict],
    real_items: list[dict],
    generated_at: str,
    *,
    profile: str = "ffpp_vox",
) -> tuple[Path, Path]:
    """Write per-split manifests with dataset + inference (same item schema as benchmark_report)."""
    out_dir = infer_dir / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = PROFILE_META.get(profile, PROFILE_META["ffpp_vox"])

    fake_manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "ground_truth_label": "fake",
        "dataset_source": meta["fake_source"],
        "count": len(fake_items),
        "items": fake_items,
    }
    real_manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "ground_truth_label": "real",
        "dataset_source": meta["real_source"],
        "count": len(real_items),
        "items": real_items,
    }

    fake_path = out_dir / "fake_manifest.json"
    real_path = out_dir / "real_manifest.json"
    fake_path.write_text(json.dumps(fake_manifest, indent=2), encoding="utf-8")
    real_path.write_text(json.dumps(real_manifest, indent=2), encoding="utf-8")
    return fake_path, real_path


def build_report(
    root: Path,
    run_id: str,
    *,
    profile: str = "ffpp_vox",
) -> tuple[dict, Path, Path, Path]:
    infer_dir = root / "results/infer" / run_id
    eval_dir = root / "results/eval" / run_id
    pred_path = infer_dir / "predictions.json"
    metrics_path = eval_dir / "metrics.json"

    if not pred_path.is_file():
        raise SystemExit(f"missing predictions: {pred_path}")

    predictions = load_json(pred_path)
    metrics = load_json(metrics_path) if metrics_path.is_file() else {}

    fake_dir = Path(predictions["fake_dir"])
    real_dir = Path(predictions["real_dir"])
    items = build_items(
        fake_dir,
        real_dir,
        predictions.get("items", []),
        profile=profile,
    )

    fake_items = [x for x in items if x["ground_truth_label"] == "fake"]
    real_items = [x for x in items if x["ground_truth_label"] == "real"]
    generated_at = datetime.now(timezone.utc).isoformat()
    meta = PROFILE_META.get(profile, PROFILE_META["ffpp_vox"])

    fake_manifest_path, real_manifest_path = write_normalized_manifests(
        infer_dir,
        fake_items,
        real_items,
        generated_at,
        profile=profile,
    )
    infer_summary_path = write_inference_summary(
        infer_dir,
        build_inference_summary(
            run_id,
            items,
            generated_at,
            threshold=predictions.get("threshold", metrics.get("threshold")),
        ),
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": generated_at,
        "model": predictions.get("model"),
        "threshold": predictions.get("threshold", metrics.get("threshold", 0.5)),
        "weights": predictions.get("weights"),
        "device": predictions.get("device"),
        "metrics": metrics,
        "datasets_meta": {
            "fake": {
                "name": meta["fake_name"],
                "dir": str(fake_dir),
                "count": len(fake_items),
            },
            "real": {
                "name": meta["real_name"],
                "dir": str(real_dir),
                "count": len(real_items),
            },
        },
        "profile": profile,
        "summary": {
            "total": len(items),
            "fake_count": len(fake_items),
            "real_count": len(real_items),
            "inference_ok": sum(1 for x in items if x["inference"]["status"] == "ok"),
        },
        "items": items,
    }
    return report, fake_manifest_path, real_manifest_path, infer_summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle Xception benchmark into one JSON")
    parser.add_argument("run_id", help="e.g. xception-benchmark-20260618-0411")
    parser.add_argument("--root", default=".", help="forenShield-ai root")
    parser.add_argument(
        "--output",
        default=None,
        help="output path (default: results/infer/<run_id>/benchmark_report.json)",
    )
    parser.add_argument(
        "--profile",
        choices=["ffpp_vox", "dfdc", "celebdf"],
        default="ffpp_vox",
        help="dataset metadata profile (default: ffpp_vox)",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = Path(args.output) if args.output else root / "results/infer" / args.run_id / "benchmark_report.json"
    if not out.is_absolute():
        out = (root / out).resolve()

    report, fake_manifest_path, real_manifest_path, infer_summary_path = build_report(
        root,
        args.run_id,
        profile=args.profile,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("run_id:", args.run_id)
    print("schema_version:", report["schema_version"])
    print("items:", len(report["items"]))
    print("fake:", report["summary"]["fake_count"])
    print("real:", report["summary"]["real_count"])
    print("written:", out)
    print("fake_manifest:", fake_manifest_path)
    print("real_manifest:", real_manifest_path)
    print("infer_summary:", infer_summary_path)


if __name__ == "__main__":
    main()
