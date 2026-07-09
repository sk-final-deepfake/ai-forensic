# AI visualization artifacts (heatmap / overlay)

AI worker can attach **representative frame thumbnails**, a **heatmap image URL**, and an **overlay video URL** to the RabbitMQ `results[0]` payload. Field names match the frontend contract in `frontend/lib/api/evidence-detail.ts`.

## Response fields (`results[0]`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `representativeFrames` | array | Top-N high-risk frames (default N=3) |
| `representativeFrames[].timeSec` | number | Timestamp in seconds |
| `representativeFrames[].timestamp` | string | `MM:SS` label |
| `representativeFrames[].frameNumber` | int | Frame index in source video |
| `representativeFrames[].score` | number | CNN fake score (0~1) |
| `representativeFrames[].imageUrl` | string \| null | Presigned S3 URL for cropped frame |
| `representativeFrames[].heatmapUrl` | string \| null | Presigned S3 URL for face-region heatmap |
| `heatmapImageUrl` | string \| null | Same as first representative frame heatmap |
| `overlayVideoUrl` | string \| null | MP4 with face bbox + risk color overlay |

## Environment

| Variable | Default | Description |
| :--- | :--- | :--- |
| `AI_VISUALIZATION_ENABLED` | `1` | Master switch |
| `AI_VISUALIZATION_UPLOAD` | `1` | Upload artifacts to S3 |
| `AI_VISUALIZATION_MAX_FRAMES` | `3` | Representative frame count |
| `AI_VISUALIZATION_OVERLAY` | `1` | Build overlay MP4 |
| `AI_VISUALIZATION_OVERLAY_MAX_SEC` | `60` | Max overlay duration |
| `S3_EVIDENCE_BUCKET` | — | Target bucket (required for URLs) |
| `AI_VISUALIZATION_PREFIX` | `cases/analysis-artifacts/{evidence_id}/{analysis_request_id}` | S3 key prefix |
| `AI_VISUALIZATION_PRESIGN_SEC` | `604800` | Presigned URL TTL (7 days) |

## Notes

- Artifacts are generated only when CNN `per_frame_scores` exist **and** YuNet detects a human face on sampled frames.
- When `NO_HUMAN_FACE` gate triggers, visualization is skipped (analysis fails before artifact generation).
- Backend must forward these fields into `analysisInfo` for the detail API; until then FE falls back to mock overlay/heatmap.
