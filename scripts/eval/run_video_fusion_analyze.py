#!/usr/bin/env python3
"""Run late-fusion video analysis locally and emit BE-compatible JSON."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.model_settings import load_model_settings
from app.schemas.analysis import AnalysisRequest
from app.services.video_deepfake_analyzer import analyze_video_request


def main() -> None:
    parser = argparse.ArgumentParser(description="Local late-fusion analyze → BE JSON")
    parser.add_argument("--video", required=True, help="Local mp4 path")
    parser.add_argument("--output", default=None, help="Write JSON to this path")
    parser.add_argument("--mock", action="store_true", help="Skip GPU infer; return mock fusion")
    parser.add_argument("--analysis-request-id", type=int, default=1)
    parser.add_argument("--evidence-id", type=int, default=1)
    args = parser.parse_args()

    settings = load_model_settings()
    if args.mock:
        settings = replace(settings, use_mock_infer=True)

    request = AnalysisRequest(
        analysisRequestId=args.analysis_request_id,
        evidenceId=args.evidence_id,
        fileType="video",
        localVideoPath=str(Path(args.video).resolve()),
        requestedAt=datetime.now(timezone.utc).isoformat(),
    )
    response = analyze_video_request(request, settings=settings)
    payload = response.model_dump(mode="json")

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
