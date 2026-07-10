from __future__ import annotations

import unittest

import numpy as np

from app.core.paths import ensure_infer_scripts_on_path


class MultiFaceInferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_infer_scripts_on_path()

    def test_build_score_breakdown_keeps_every_face_crop(self) -> None:
        from video_xception_infer import build_score_breakdown

        face_samples = [
            {"frame_index": 10, "face_index": 0, "bbox": (10, 20, 30, 40)},
            {"frame_index": 10, "face_index": 1, "bbox": (100, 20, 25, 25)},
            {"frame_index": 20, "face_index": 0, "bbox": (12, 22, 28, 38)},
        ]
        logits = np.array(
            [
                [1.0, -1.0],
                [-1.0, 2.5],
                [0.0, 0.0],
            ],
            dtype=np.float64,
        )

        breakdown = build_score_breakdown(
            face_samples,
            logits,
            threshold=0.5,
            frames_sampled=32,
            frames_without_face=30,
            aggregate="max",
        )

        self.assertEqual(len(breakdown["per_frame_scores"]), 3)
        self.assertEqual(breakdown["unique_frames_with_face"], 2)
        self.assertTrue(breakdown["multi_face"])
        self.assertGreaterEqual(float(breakdown["aggregate_fake_score"]), 0.9)
        frame_10_scores = [
            row["fake_score"]
            for row in breakdown["per_frame_scores"]
            if row["frame_index"] == 10
        ]
        self.assertEqual(len(frame_10_scores), 2)

    def test_score_map_by_frame_groups_faces(self) -> None:
        from app.services import visualization_artifacts as viz

        grouped = viz._score_map_by_frame(
            [
                {"frame_index": 1, "face_index": 0, "fake_score": 0.2},
                {"frame_index": 1, "face_index": 1, "fake_score": 0.9},
            ]
        )
        self.assertEqual(len(grouped[1]), 2)
        self.assertEqual(grouped[1][1]["score"], 0.9)

    def test_build_clip_score_breakdown_keeps_face_slots(self) -> None:
        from video_clip_transformer_common import build_clip_score_breakdown

        per_clip = [
            {
                "clip_index": 0,
                "face_index": 0,
                "prob_fake": 0.2,
                "prob_real": 0.8,
                "logit_real": 0.0,
                "logit_fake": 0.0,
                "margin": 0.0,
                "entropy": 0.0,
                "confidence": 0.8,
                "pred_label": "real",
                "frame_indices": [10, 11, 12],
                "clip_start_frame": 10,
                "clip_end_frame": 12,
            },
            {
                "clip_index": 1,
                "face_index": 1,
                "prob_fake": 0.95,
                "prob_real": 0.05,
                "logit_real": 0.0,
                "logit_fake": 0.0,
                "margin": 0.0,
                "entropy": 0.0,
                "confidence": 0.95,
                "pred_label": "fake",
                "frame_indices": [10, 11, 12],
                "clip_start_frame": 10,
                "clip_end_frame": 12,
            },
        ]
        face_samples = [
            {"frame_index": 10, "face_index": 0, "bbox": (1, 2, 3, 4), "crop": None},
            {"frame_index": 10, "face_index": 1, "bbox": (20, 2, 3, 4), "crop": None},
        ]
        breakdown = build_clip_score_breakdown(
            per_clip,
            method="timesformer_clip_classification_outputs",
            threshold=0.5,
            frames_sampled=32,
            frames_without_face=30,
            clip_frames=8,
            clip_size=224,
            max_clips=4,
            aggregate="max",
            face_samples=face_samples,
            multi_face=True,
        )
        self.assertTrue(breakdown["multi_face"])
        self.assertGreaterEqual(float(breakdown["aggregate_fake_score"]), 0.95)
        self.assertEqual(len({row["face_index"] for row in breakdown["per_frame_scores"]}), 2)

    def test_build_fused_per_frame_scores_uses_temporal_face(self) -> None:
        from app.services.late_fusion import build_fused_per_frame_scores

        fused = build_fused_per_frame_scores(
            cnn_scores=[
                {"frame_index": 10, "face_index": 0, "fake_score": 0.4, "bbox": (1, 2, 3, 4)},
                {"frame_index": 10, "face_index": 1, "fake_score": 0.4, "bbox": (20, 2, 3, 4)},
            ],
            temporal_scores=[
                {"frame_index": 10, "face_index": 0, "fake_score": 0.1},
                {"frame_index": 10, "face_index": 1, "fake_score": 0.9},
            ],
            optical_score=0.0,
            fuse_fn=lambda *, cnn_score, temporal_score, optical_score: max(
                cnn_score, temporal_score, optical_score
            ),
            temporal_video_score=0.5,
        )
        by_face = {row["face_index"]: row["fake_score"] for row in fused}
        self.assertEqual(by_face[0], 0.4)
        self.assertEqual(by_face[1], 0.9)


if __name__ == "__main__":
    unittest.main()
