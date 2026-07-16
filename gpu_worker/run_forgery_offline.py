#!/usr/bin/env python3
"""Offline forgery lane smoke test (TruFor + TimeSformer) without RabbitMQ."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _OfflineCfg:
    project_root: Path
    work_dir: Path
    device: str = "cuda"


def _as_dict(obj: object) -> dict:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")  # type: ignore[union-attr]
    if isinstance(obj, dict):
        return obj
    return {
        key: getattr(obj, key)
        for key in (
            "moduleName",
            "score",
            "detected",
            "modelName",
            "modelVersion",
            "module",
            "videoScore",
            "frameRisks",
            "clipRisks",
        )
        if hasattr(obj, key)
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run forgery lane on a local mp4 and print merged JSON snippet")
    parser.add_argument("video", type=Path, help="Path to mp4")
    parser.add_argument("--root", type=Path, default=Path.home() / "forenShield-ai")
    parser.add_argument("--out", type=Path, default=None, help="Write full merged payload JSON here")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from gpu_worker.pipeline.forgery_infer import run_forgery_modules
    from gpu_worker.pipeline.forgery_merge import merge_forgery_into_response

    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"video not found: {video}", file=sys.stderr)
        return 1

    cfg = _OfflineCfg(project_root=root, work_dir=root / "work" / "forgery_offline")
    cfg.work_dir.mkdir(parents=True, exist_ok=True)

    class _Video:
        modelScores = None
        moduleTimelines = None

    class _Resp:
        status = "COMPLETED"
        modelScores = []
        analysisReasons = []
        results = [_Video()]

    # Parent only — run_forgery_modules creates a unique child and deletes it after.
    forgery = run_forgery_modules(video, cfg, work_dir=cfg.work_dir)
    merged = merge_forgery_into_response(_Resp(), forgery, worker_cfg=cfg)

    payload = {
        "modelScores": [
            {
                "moduleName": row.get("moduleName"),
                "score": row.get("score"),
                "detected": row.get("detected"),
                "modelName": row.get("modelName"),
                "modelVersion": row.get("modelVersion"),
            }
            for row in (_as_dict(m) for m in (merged.modelScores or []))
        ],
        "moduleTimelines": [
            {
                "module": row.get("module"),
                "videoScore": row.get("videoScore"),
                "frameRisks": row.get("frameRisks"),
                "clipRisks": row.get("clipRisks"),
            }
            for row in (_as_dict(t) for t in (merged.results[0].moduleTimelines or []))
        ],
    }

    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        print(f"\nWrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
