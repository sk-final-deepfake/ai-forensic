from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from app.schemas.messaging import (
    AnalysisJobMessage,
    AnalysisResponseMessage,
    AnalysisVideoResultItem,
    FaceBBoxItem,
    FrameRiskItem,
    ModelOverlayArtifactItem,
    ModuleTimelineItem,
    PerFrameFaceScoreItem,
)
from app.services.response_visualization import (
    attach_visualization_artifacts,
    build_visualization_payload,
    per_frame_face_scores_from_video_item,
    per_frame_scores_from_cnn_raw,
    per_frame_scores_from_video_item,
)


class ResponseVisualizationTests(unittest.TestCase):
    def test_per_frame_scores_from_cnn_raw(self) -> None:
        rows = per_frame_scores_from_cnn_raw(
            {
                "score_breakdown": {
                    "per_frame": [
                        {"frame_index": 10, "prob_fake": 0.81},
                        {"frame_index": 20, "prob_fake": 0.62},
                    ]
                }
            }
        )
        self.assertEqual(rows[0]["frame_index"], 10)
        self.assertAlmostEqual(rows[0]["fake_score"], 0.81)

    def test_per_frame_scores_from_video_item(self) -> None:
        video = AnalysisVideoResultItem(
            deepfakeDetected=True,
            deepfakeScore=0.8,
            frameRisks=[
                FrameRiskItem(frameIndex=5, timestampSec=0.2, riskScore=0.77),
            ],
        )
        rows = per_frame_scores_from_video_item(video)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["frame_index"], 5)

    def test_per_frame_scores_from_video_item_prefers_face_scores(self) -> None:
        video = AnalysisVideoResultItem(
            frameRisks=[
                FrameRiskItem(frameIndex=10, timestampSec=0.4, riskScore=0.9),
            ],
            perFrameFaceScores=[
                PerFrameFaceScoreItem(
                    frameIndex=10,
                    faceIndex=0,
                    riskScore=0.4,
                    bbox=FaceBBoxItem(x=1, y=2, w=3, h=4),
                ),
                PerFrameFaceScoreItem(
                    frameIndex=10,
                    faceIndex=1,
                    riskScore=0.9,
                    bbox=FaceBBoxItem(x=20, y=2, w=3, h=4),
                ),
            ],
        )
        rows = per_frame_scores_from_video_item(video)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["face_index"], 1)
        self.assertEqual(rows[1]["bbox"]["x"], 20)

    def test_per_frame_face_scores_from_video_item(self) -> None:
        video = AnalysisVideoResultItem(
            perFrameFaceScores=[
                PerFrameFaceScoreItem(frameIndex=3, faceIndex=0, riskScore=0.5),
            ],
        )
        rows = per_frame_face_scores_from_video_item(video)
        self.assertEqual(rows[0]["frame_index"], 3)

    def test_attach_visualization_skips_when_already_present(self) -> None:
        job = AnalysisJobMessage(
            analysisRequestId=1,
            evidenceId=2,
            filePath="evidence.mp4",
        )
        response = AnalysisResponseMessage(
            analysisRequestId=1,
            evidenceId=2,
            status="COMPLETED",
            analyzedAt="2026-07-10T00:00:00Z",
            results=[
                AnalysisVideoResultItem(
                    deepfakeDetected=True,
                    deepfakeScore=0.8,
                    overlayVideoUrl="https://cdn.example/overlay.mp4",
                )
            ],
        )
        updated = attach_visualization_artifacts(job, response)
        self.assertEqual(updated.results[0].overlayVideoUrl, "https://cdn.example/overlay.mp4")

    @patch("app.services.response_visualization.build_visualization_payload")
    @patch("app.services.response_visualization.download_messaging_job_video")
    def test_attach_visualization_skips_when_gpu_model_overlays_present(
        self,
        mock_download: unittest.mock.Mock,
        mock_payload: unittest.mock.Mock,
    ) -> None:
        job = AnalysisJobMessage(
            analysisRequestId=99,
            evidenceId=203,
            filePath="evidence.mp4",
        )
        response = AnalysisResponseMessage(
            analysisRequestId=99,
            evidenceId=203,
            status="COMPLETED",
            analyzedAt="2026-07-10T00:00:00Z",
            results=[
                AnalysisVideoResultItem(
                    deepfakeDetected=True,
                    deepfakeScore=0.8,
                    perFrameFaceScores=[
                        PerFrameFaceScoreItem(frameIndex=1, faceIndex=0, riskScore=0.5),
                    ],
                    modelOverlayArtifacts=[
                        ModelOverlayArtifactItem(
                            key="deepfake:cnn",
                            category="deepfake",
                            label="Xception",
                            overlayVideoUrl="https://cdn.example/cnn.mp4",
                            status="ready",
                        ),
                        ModelOverlayArtifactItem(
                            key="deepfake:temporal",
                            category="deepfake",
                            label="TimeSformer",
                            overlayVideoUrl="https://cdn.example/temporal.mp4",
                            status="ready",
                        ),
                        ModelOverlayArtifactItem(
                            key="deepfake:optical",
                            category="deepfake",
                            label="GMFlow",
                            overlayVideoUrl="https://cdn.example/optical.mp4",
                            status="ready",
                        ),
                    ],
                )
            ],
        )

        updated = attach_visualization_artifacts(job, response)
        video = updated.results[0]
        self.assertEqual(video.modelOverlayArtifacts[1].overlayVideoUrl, "https://cdn.example/temporal.mp4")
        mock_download.assert_not_called()
        mock_payload.assert_not_called()

    @patch("app.services.response_visualization.build_visualization_payload")
    @patch("app.services.response_visualization.download_messaging_job_video")
    def test_attach_visualization_skips_when_gpu_module_timelines_present(
        self,
        mock_download: unittest.mock.Mock,
        mock_payload: unittest.mock.Mock,
    ) -> None:
        """GPU OOM may leave empty overlays; still skip expensive EKS legacy fallback."""
        job = AnalysisJobMessage(
            analysisRequestId=101,
            evidenceId=257,
            filePath="evidence.mp4",
        )
        response = AnalysisResponseMessage(
            analysisRequestId=101,
            evidenceId=257,
            status="COMPLETED",
            analyzedAt="2026-07-13T00:00:00Z",
            results=[
                AnalysisVideoResultItem(
                    deepfakeDetected=True,
                    deepfakeScore=0.7,
                    perFrameFaceScores=[
                        PerFrameFaceScoreItem(frameIndex=1, faceIndex=0, riskScore=0.6),
                    ],
                    moduleTimelines=[
                        ModuleTimelineItem(
                            module="cnn",
                            modelName="Xception",
                            modelVersion="v1",
                            videoScore=0.7,
                            threshold=0.5,
                            detected=True,
                        ),
                        ModuleTimelineItem(
                            module="temporal",
                            modelName="TimeSformer",
                            modelVersion="v1",
                            videoScore=0.6,
                            threshold=0.5,
                            detected=True,
                        ),
                        ModuleTimelineItem(
                            module="optical",
                            modelName="GMFlow",
                            modelVersion="v1",
                            videoScore=0.5,
                            threshold=0.4,
                            detected=True,
                        ),
                    ],
                )
            ],
        )

        updated = attach_visualization_artifacts(job, response)
        self.assertIsNone(updated.results[0].overlayVideoUrl)
        mock_download.assert_not_called()
        mock_payload.assert_not_called()

    @patch("app.services.response_visualization.build_visualization_payload")
    @patch("app.services.response_visualization.download_messaging_job_video")
    def test_attach_visualization_enriches_completed_response(
        self,
        mock_download: unittest.mock.Mock,
        mock_payload: unittest.mock.Mock,
    ) -> None:
        mock_download.return_value = Path("video.mp4")
        mock_payload.return_value = {
            "representativeFrames": [
                {
                    "timeSec": 1.0,
                    "timestamp": "00:01",
                    "frameNumber": 25,
                    "score": 0.9,
                    "imageUrl": "https://cdn.example/frame.jpg",
                }
            ],
            "overlayVideoUrl": "https://cdn.example/overlay.mp4",
        }

        job = AnalysisJobMessage(
            analysisRequestId=11,
            evidenceId=22,
            filePath="evidence.mp4",
            presignedDownloadUrl="https://cdn.example/video.mp4",
        )
        response = AnalysisResponseMessage(
            analysisRequestId=11,
            evidenceId=22,
            status="COMPLETED",
            analyzedAt="2026-07-10T00:00:00Z",
            results=[
                AnalysisVideoResultItem(
                    deepfakeDetected=True,
                    deepfakeScore=0.8,
                    frameRisks=[
                        FrameRiskItem(frameIndex=25, timestampSec=1.0, riskScore=0.9),
                    ],
                )
            ],
        )

        updated = attach_visualization_artifacts(job, response)
        video = updated.results[0]
        self.assertEqual(video.overlayVideoUrl, "https://cdn.example/overlay.mp4")
        self.assertEqual(len(video.representativeFrames or []), 1)

    def test_build_visualization_payload_returns_none_without_scores(self) -> None:
        payload = build_visualization_payload(
            video_path=Path("missing.mp4"),
            per_frame_scores=[],
            evidence_id=1,
            analysis_request_id=2,
            work_dir=Path("."),
        )
        self.assertIsNone(payload)


if __name__ == "__main__":
    unittest.main()
