#!/usr/bin/env python3
"""Regenerate summary.json / benchmark_report.html from an existing infer run dir."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.infer.optical_flow_common import (  # noqa: E402
    DEFAULT_ANOMALY_THRESHOLD,
    build_benchmark_report_document,
    enrich_reports_with_scores,
    generate_benchmark_html,
    save_report_json,
)
from scripts.common import s3_deepfake_paths as s3p

S3_BUCKET = "s3://forenshield-evidence-877044078824"

_META_JSON_NAMES = {
    "benchmark_report.json",
    "summary.json",
    "predictions.json",
    "checkpoint.json",
    "infer_summary.json",
    "metrics.json",
}


def _infer_model_from_run_id(run_id: str) -> str:
    lowered = run_id.lower()
    if "raft" in lowered:
        return "raft"
    if "gmflow" in lowered:
        return "gmflow"
    if "pwcnet" in lowered or "pwc" in lowered:
        return "pwcnet"
    return "raft"


def _infer_profile(run_id: str, run_dir: Path, meta_profile: str | None) -> str | None:
    if meta_profile:
        return meta_profile
    lowered = run_id.lower()
    if "ffpp" in lowered:
        return "ffpp_vox"
    if "celeb" in lowered:
        return "celebdf"
    checkpoint = run_dir / "checkpoint.json"
    if checkpoint.is_file():
        data = json.loads(checkpoint.read_text(encoding="utf-8"))
        if data.get("profile"):
            return data["profile"]
    return None


def _is_per_video_report(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("status") not in ("ok", "error", "pending"):
        return False
    if data.get("ground_truth_label") in ("fake", "real"):
        return True
    if data.get("label") in ("fake", "real"):
        return True
    if data.get("pair_stats") or data.get("score_breakdown"):
        return True
    if data.get("flow_mean") is not None and data.get("model"):
        return True
    return False


def _collect_report_json_paths(run_dir: Path, model: str | None = None) -> list[Path]:
    paths: list[Path] = []

    if model:
        model_json = run_dir / model / "json"
        if model_json.is_dir():
            return sorted(model_json.glob("*.json"))
        json_dir = run_dir / "json"
        if json_dir.is_dir():
            return sorted(json_dir.glob("*.json"))

    predictions_path = run_dir / "predictions.json"
    json_dir = run_dir / "json"
    model_json_dirs = [run_dir / model / "json" for model in ("raft", "gmflow", "pwcnet")]
    has_per_video = json_dir.is_dir() or any(d.is_dir() for d in model_json_dirs)
    if predictions_path.is_file() and not has_per_video:
        return [predictions_path]

    if json_dir.is_dir():
        paths.extend(sorted(json_dir.glob("*.json")))

    for model_json in model_json_dirs:
        if model_json.is_dir():
            paths.extend(sorted(model_json.glob("*.json")))

    for path in sorted(run_dir.glob("*.json")):
        if path.name in _META_JSON_NAMES:
            continue
        if path.name.startswith("infer_summary_") or path.name.startswith("metrics_"):
            continue
        paths.append(path)

    # S3 layout: deepfake/results/infer/raft/{profile}/{fake,real}/*.json
    for sub in ("fake", "real"):
        label_dir = run_dir / sub
        if label_dir.is_dir():
            paths.extend(sorted(label_dir.glob("*.json")))

    # nested layouts (e.g. synced from S3)
    for sub in ("reports", "output", "results"):
        nested = run_dir / sub
        if nested.is_dir():
            paths.extend(sorted(nested.rglob("*.json")))

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _load_report_file(path: Path) -> dict | list[dict] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if path.name == "predictions.json" and isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list) and items:
            return items
    if isinstance(data, dict) and _is_per_video_report(data):
        return data
    return None


def _tag_label_from_parent(path: Path, report: dict) -> dict:
    if report.get("ground_truth_label") in ("fake", "real"):
        return report
    parent = path.parent.name.lower()
    if parent in ("fake", "real"):
        report["ground_truth_label"] = parent
    return report


def _load_reports(run_dir: Path, model: str | None = None) -> list[dict]:
    reports: list[dict] = []
    for path in _collect_report_json_paths(run_dir, model=model):
        loaded = _load_report_file(path)
        if loaded is None:
            continue
        if isinstance(loaded, list):
            for item in loaded:
                if isinstance(item, dict):
                    reports.append(_tag_label_from_parent(path, item))
        else:
            reports.append(_tag_label_from_parent(path, loaded))

    if reports:
        if model:
            reports = [r for r in reports if not r.get("model") or r.get("model") == model]
        return reports

    tried = [str(p) for p in _collect_report_json_paths(run_dir, model=model)]
    raise FileNotFoundError(
        f"No per-video JSON under {run_dir}\n"
        f"Expected predictions.json, json/*.json, fake/*.json, real/*.json, or pair_stats/score_breakdown.\n"
        f"Tried ({len(tried)} paths). Re-run with --sync-s3 to download from S3."
    )


def _s3_prefixes_for_run(run_id: str) -> list[str]:
    legacy_optical = s3p.legacy_reports("video-optical-flow-benchmark")
    raft_ffpp = s3p.infer_model("raft", "ffpp_vox")
    prefixes = [
        f"{S3_BUCKET}/{legacy_optical}/{run_id}/",
        f"{S3_BUCKET}/{legacy_optical}/{run_id}",
        f"{S3_BUCKET}/{raft_ffpp}/",
        f"{S3_BUCKET}/{raft_ffpp}/{run_id}/",
    ]
    lowered = run_id.lower()
    if "raft" in lowered and "celeb" in lowered:
        raft_celeb = s3p.infer_model("raft", "celebdf")
        prefixes.extend(
            [
                f"{S3_BUCKET}/{raft_celeb}/",
                f"{S3_BUCKET}/{raft_celeb}/{run_id}/",
            ]
        )
    return prefixes


def _sync_from_s3(run_dir: Path, run_id: str) -> str | None:
    run_dir.mkdir(parents=True, exist_ok=True)
    for s3_uri in _s3_prefixes_for_run(run_id):
        print(f"Trying S3 sync: {s3_uri} -> {run_dir}")
        result = subprocess.run(
            ["aws", "s3", "sync", s3_uri, str(run_dir), "--exclude", ".git/*"],
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            continue
        try:
            _load_reports(run_dir)
        except FileNotFoundError:
            continue
        return s3_uri
    return None


def _write_back_per_file_json(run_dir: Path, reports: list[dict]) -> None:
    json_dir = run_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    for report in reports:
        file_name = report.get("file")
        if not file_name:
            source = report.get("source_path")
            if source:
                file_name = Path(source).name
        if not file_name:
            continue
        stem = Path(file_name).stem
        out_path = json_dir / f"{stem}.json"
        save_report_json(out_path, report)

    model = reports[0].get("model") if reports else None
    predictions = {
        "runId": run_dir.name,
        "model": model,
        "count": len(reports),
        "items": reports,
    }
    (run_dir / "predictions.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _infer_meta(run_dir: Path, run_id: str) -> dict:
    default_model = _infer_model_from_run_id(run_id)
    for name in ("infer_summary_pwcnet.json", "infer_summary_raft.json", "infer_summary_gmflow.json"):
        path = run_dir / name
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                "model_name": data.get("model") or default_model,
                "profile": data.get("profile"),
                "threshold": data.get("threshold", DEFAULT_ANOMALY_THRESHOLD),
            }

    for name in ("benchmark_report.json", "summary.json"):
        path = run_dir / name
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return {
                "model_name": data.get("model") or default_model,
                "profile": data.get("profile"),
                "threshold": data.get("threshold", DEFAULT_ANOMALY_THRESHOLD),
            }

    return {"model_name": default_model, "profile": None, "threshold": DEFAULT_ANOMALY_THRESHOLD}


def regenerate(
    run_dir: Path,
    run_id: str | None = None,
    *,
    sync_s3: bool = False,
    model: str | None = None,
) -> None:
    run_id = run_id or run_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        reports = _load_reports(run_dir, model=model)
    except FileNotFoundError as exc:
        if sync_s3 is False:
            raise exc
        synced = _sync_from_s3(run_dir, run_id)
        if not synced:
            prefixes = "\n".join(f"  aws s3 sync {p} {run_dir}/" for p in _s3_prefixes_for_run(run_id))
            raise FileNotFoundError(
                f"Could not find or sync per-video JSON for {run_id}.\n"
                f"Manual sync examples:\n{prefixes}"
            )
        print(f"Synced from {synced}")
        reports = _load_reports(run_dir, model=model)

    meta = _infer_meta(run_dir, run_id)
    if model:
        meta["model_name"] = model
    if not meta["model_name"] and reports:
        meta["model_name"] = reports[0].get("model") or _infer_model_from_run_id(run_id)

    profile = _infer_profile(run_id, run_dir, meta["profile"])
    cohort = enrich_reports_with_scores(reports, threshold=float(meta["threshold"]))
    _write_back_per_file_json(run_dir, reports)

    doc = build_benchmark_report_document(
        reports,
        run_id=run_id,
        model_name=meta["model_name"],
        profile=profile,
        threshold=float(meta["threshold"]),
        cohort=cohort,
        fake_dir=None,
        real_dir=None,
    )

    (run_dir / "summary.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "benchmark_report.json").write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "benchmark_report.html").write_text(generate_benchmark_html(doc), encoding="utf-8")

    model_name = meta["model_name"]
    (run_dir / f"infer_summary_{model_name}.json").write_text(
        json.dumps(doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / f"metrics_{model_name}.json").write_text(
        json.dumps(doc["metrics"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Regenerated reports in {run_dir}")
    print(f"  items: {len(doc['items'])}")
    print(f"  profile: {profile}")
    print(f"  benchmark_report.html")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="Run id under results/infer/<run_id>")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="forenShield-ai root (default: repo ai/)",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override run directory (default: <root>/results/infer/<run_id>)",
    )
    parser.add_argument(
        "--sync-s3",
        action="store_true",
        default=True,
        help="If local json is missing, aws s3 sync from known prefixes (default: on)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optical-flow backend subdir: raft | gmflow | pwcnet (loads <run_dir>/<model>/json/)",
    )
    parser.add_argument(
        "--no-sync-s3",
        action="store_false",
        dest="sync_s3",
        help="Do not download from S3; fail if local json/ is missing",
    )
    args = parser.parse_args()
    run_dir = args.run_dir or (args.root / "results" / "infer" / args.run_id)
    regenerate(run_dir, args.run_id, sync_s3=args.sync_s3, model=args.model)


if __name__ == "__main__":
    main()
