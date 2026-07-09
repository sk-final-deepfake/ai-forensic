"""Xception operational defaults (Step 2 + threshold decision 2026-06-24)."""

from __future__ import annotations

# Step 2 preprocessing + aggregate; Step 3 fine-tune keeps the same infer pipeline.
DEFAULT_CROP_METHOD = "mediapipe"
DEFAULT_CROP_PADDING = 0.3
DEFAULT_CROP_SQUARE = True
DEFAULT_AGGREGATE = "topk"
DEFAULT_TOP_K = 5
DEFAULT_NUM_FRAMES = 32

# Operating classification threshold (Youden on Step 2 combined 200).
DEFAULT_FAKE_THRESHOLD = 0.78
DEFAULT_SUSPICIOUS_LOW = 0.5

DEFAULT_WEIGHTS = "models/test/video/xception/v1.0.0/xception_best.pth"
DEFAULT_FINETUNED_WEIGHTS = "models/test/video/xception/v1.0.0/xception_finetuned.pth"
