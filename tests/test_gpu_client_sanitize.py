from __future__ import annotations

from app.gpu_client import _sanitize_gateway_response


def test_sanitize_gateway_response_coerces_null_module_scores() -> None:
    raw = {
        "analysisRequestId": 228,
        "evidenceId": 1,
        "status": "COMPLETED",
        "analyzedAt": "2026-07-13T00:00:00Z",
        "modelScores": [
            {"moduleName": "deepfake", "detected": False, "score": 0.2, "modelName": "Late Fusion", "modelVersion": "v4"},
            {"moduleName": "deepfake_cnn", "detected": False, "score": 0.1, "modelName": "Xception", "modelVersion": "v1"},
            {"moduleName": "deepfake_temporal", "detected": False, "score": None, "modelName": "TimeSformer", "modelVersion": "v1"},
        ],
        "results": [
            {
                "type": "video",
                "moduleTimelines": [
                    {
                        "module": "cnn",
                        "modelName": "Xception",
                        "modelVersion": "v1",
                        "videoScore": 0.1,
                        "threshold": 0.5,
                        "detected": False,
                    },
                    {
                        "module": "temporal",
                        "modelName": "TimeSformer",
                        "modelVersion": "v1",
                        "videoScore": None,
                        "threshold": 0.6,
                        "detected": False,
                    },
                ],
            }
        ],
    }

    sanitized = _sanitize_gateway_response(raw)
    assert sanitized["modelScores"][2]["score"] == 0.0
    assert sanitized["results"][0]["moduleTimelines"][1]["videoScore"] == 0.0
