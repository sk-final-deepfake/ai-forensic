from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.services.module_overlays import _frame_risks_to_bboxes
from app.services.trufor_overlay import draw_trufor_bboxes, tamper_map_to_bboxes


class TruForBBoxOverlayTests(unittest.TestCase):
    def test_tamper_map_to_bboxes_finds_blob(self) -> None:
        m = np.zeros((64, 64), dtype=np.float32)
        m[20:40, 10:30] = 0.9
        boxes = tamper_map_to_bboxes(m, 640, 480, threshold=0.5)
        self.assertGreaterEqual(len(boxes), 1)
        box = boxes[0]
        self.assertGreater(box.w, 50)
        self.assertGreater(box.h, 50)
        # Roughly in left-center of the frame
        self.assertLess(box.x, 320)

    def test_draw_trufor_bboxes_paints_rectangle(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        out = draw_trufor_bboxes(frame, [{"x": 20, "y": 10, "w": 40, "h": 50, "score": 0.8}])
        self.assertFalse(np.array_equal(frame, out))

    def test_frame_risks_to_bboxes(self) -> None:
        mapped = _frame_risks_to_bboxes(
            [
                {
                    "frameIndex": 3,
                    "riskScore": 0.7,
                    "bboxes": [{"x": 1, "y": 2, "w": 10, "h": 12, "score": 0.9}],
                }
            ]
        )
        self.assertIn(3, mapped)
        self.assertEqual(mapped[3][0]["w"], 10)

    def test_overlay_bbox_style_from_npz(self) -> None:
        from app.services.trufor_overlay import overlay_trufor_on_frame

        with tempfile.TemporaryDirectory() as tmp:
            npz = Path(tmp) / "sample_f000.npz"
            m = np.zeros((32, 32), dtype=np.float32)
            m[8:20, 8:20] = 0.95
            np.savez(npz, map=m, score=np.array(0.95))
            frame = np.full((64, 64, 3), 30, dtype=np.uint8)
            blended, heatmap, score = overlay_trufor_on_frame(frame, npz, style="bbox")
            self.assertGreater(score, 0.9)
            self.assertEqual(heatmap.shape[:2], (32, 32))
            self.assertFalse(np.array_equal(frame, blended))


if __name__ == "__main__":
    unittest.main()
