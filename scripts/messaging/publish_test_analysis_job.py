#!/usr/bin/env python3
"""Publish a test AnalysisJobMessage to forenshield.analysis.queue."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pika

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.messaging.settings import RabbitMqSettings
from app.messaging.topology import ensure_topology


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish test analysis job to RabbitMQ")
    parser.add_argument("--analysis-request-id", type=int, default=9001)
    parser.add_argument("--evidence-id", type=int, default=9001)
    parser.add_argument("--presigned-url", default=None)
    parser.add_argument("--local-video", default=None, help="Dev: set localVideoPath in payload")
    parser.add_argument("--file-path", default="cases/test/copy/evidence.mp4")
    args = parser.parse_args()

    settings = RabbitMqSettings.from_env()
    payload = {
        "analysisRequestId": args.analysis_request_id,
        "evidenceId": args.evidence_id,
        "fileType": "video",
        "filePath": args.file_path,
        "s3ObjectKey": args.file_path,
        "presignedDownloadUrl": args.presigned_url,
        "localVideoPath": args.local_video,
        "originalHash": "test-hash",
        "requestedAt": datetime.now(timezone.utc).isoformat(),
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    connection = pika.BlockingConnection(pika.URLParameters(settings.url))
    channel = connection.channel()
    ensure_topology(channel, settings)
    channel.basic_publish(
        exchange=settings.analysis_exchange,
        routing_key=settings.video_analysis_routing_key,
        body=body,
        properties=pika.BasicProperties(content_type="application/json", delivery_mode=2),
    )
    connection.close()
    print(f"published job analysisRequestId={args.analysis_request_id} evidenceId={args.evidence_id}")


if __name__ == "__main__":
    main()
