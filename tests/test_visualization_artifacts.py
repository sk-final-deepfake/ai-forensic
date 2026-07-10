from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.services import visualization_artifacts as viz


class VisualizationArtifactUnitTests(unittest.TestCase):
    def test_pick_representative_rows_sorts_by_score(self) -> None:
        rows = viz._pick_representative_rows(
            [
                {"frame_index": 1, "fake_score": 0.2},
                {"frame_index": 2, "fake_score": 0.9},
                {"frame_index": 3, "fake_score": 0.5},
            ],
            limit=2,
        )
        self.assertEqual([row["frame_index"] for row in rows], [2, 3])

    def test_render_heatmap_layer_shape(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        heat = viz._render_heatmap_layer(frame.shape, (40, 30, 50, 60), 0.8)
        self.assertEqual(heat.shape, frame.shape)

    def test_timestamp_label(self) -> None:
        self.assertEqual(viz._timestamp_label(65.0), "01:05")

    def test_finalize_overlay_video_prefers_h264_when_ffmpeg_available(self) -> None:
        if not viz._ffmpeg_path():
            self.skipTest("ffmpeg not installed")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "overlay_raw.mp4"
            out_path = tmp_path / "overlay.mp4"
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            writer = cv2.VideoWriter(
                str(raw_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (64, 64),
            )
            for _ in range(5):
                writer.write(frame)
            writer.release()

            result = viz._finalize_overlay_video(raw_path, out_path)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertTrue(result.is_file())
            self.assertFalse(raw_path.exists())

            ffprobe = shutil.which("ffprobe")
            if not ffprobe:
                self.skipTest("ffprobe not installed")

            probe = subprocess.run(
                [
                    ffprobe,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=codec_name",
                    "-of",
                    "default=nokey=1:noprint_wrappers=1",
                    str(result),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(probe.stdout.strip(), "h264")


class VisualizationArtifactIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from app.core.paths import ensure_infer_scripts_on_path

        ensure_infer_scripts_on_path()

    def test_build_artifacts_from_human_frame_video(self) -> None:
        ai_root = Path(__file__).resolve().parents[1]
        human = ai_root / "docs" / "notebooks" / "output" / "output" / "face-check" / "ai_KTYafGwBb9A_frame0.jpg"
        if not human.is_file():
            self.skipTest(f"missing fixture: {human}")

        frame = cv2.imread(str(human))
        self.assertIsNotNone(frame)

        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "sample.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                25.0,
                (frame.shape[1], frame.shape[0]),
            )
            for _ in range(10):
                writer.write(frame)
            writer.release()

            with patch.dict(os.environ, {"AI_VISUALIZATION_UPLOAD": "0"}, clear=False):
                artifacts = viz.build_visualization_artifacts(
                    video_path=video_path,
                    per_frame_scores=[{"frame_index": 3, "fake_score": 0.91}],
                    evidence_id=42,
                    analysis_request_id=7,
                    work_dir=Path(tmp) / "viz",
                )

            self.assertIsNotNone(artifacts)
            assert artifacts is not None
            self.assertEqual(len(artifacts.representative_frames), 1)
            frame_item = artifacts.representative_frames[0]
            self.assertEqual(frame_item["frameNumber"], 3)
            self.assertAlmostEqual(frame_item["score"], 0.91, places=5)
            self.assertIsNone(frame_item["imageUrl"])
            self.assertTrue((Path(tmp) / "viz" / "frame_00.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
