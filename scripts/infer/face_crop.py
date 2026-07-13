"""Face crop utilities for video CNN / transformer infer.

Production default: OpenCV YuNet human-face detector (method=yunet, human_only=True).
Legacy: Haar / MediaPipe blaze-face (human_only=False only).
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_MP_MODEL_URLS = {
    0: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
    1: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite",
}
_YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
NO_HUMAN_FACE_STATUS = "no_human_face"
FACE_TOO_SMALL_STATUS = "face_too_small"
INSUFFICIENT_FACE_SAMPLES_STATUS = "insufficient_face_samples"
# Reject crops smaller than this (min side in px). ~30px full-body faces fail this gate.
DEFAULT_MIN_FACE_SIDE_PX = 48


def _mediapipe_model_cache_dir() -> Path:
    return Path.home() / ".cache" / "forenshield" / "mediapipe"


def _ensure_mediapipe_model(model_selection: int) -> Path:
    key = 1 if model_selection else 0
    cache_dir = _mediapipe_model_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = "blaze_face_full_range.tflite" if key == 1 else "blaze_face_short_range.tflite"
    path = cache_dir / filename
    if not path.is_file():
        urllib.request.urlretrieve(_MP_MODEL_URLS[key], path)
    return path


def _opencv_model_cache_dir() -> Path:
    return Path.home() / ".cache" / "forenshield" / "opencv"


def _ensure_yunet_model() -> Path:
    cache_dir = _opencv_model_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "face_detection_yunet_2023mar.onnx"
    if not path.is_file():
        urllib.request.urlretrieve(_YUNET_MODEL_URL, path)
    return path


@dataclass(frozen=True)
class FaceCropConfig:
    method: str = "yunet"  # yunet | mediapipe | haar
    size: int = 256
    padding: float = 0.2
    square: bool = True
    mediapipe_model_selection: int = 1
    mediapipe_min_confidence: float = 0.5
    human_only: bool = True
    yunet_score_threshold: float = 0.75
    yunet_nms_threshold: float = 0.3
    min_sample_faces: int = 4
    min_face_side_px: int = DEFAULT_MIN_FACE_SIDE_PX


def _clip_bbox(x1: int, y1: int, x2: int, y2: int, w_img: int, h_img: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(x1 + 1, min(x2, w_img))
    y2 = max(y1 + 1, min(y2, h_img))
    return x1, y1, x2, y2


def _apply_padding_square(
    x: int,
    y: int,
    w: int,
    h: int,
    *,
    padding: float,
    square: bool,
    h_img: int,
    w_img: int,
) -> tuple[int, int, int, int]:
    if square:
        side = int(max(w, h) * (1.0 + padding))
        cx = x + w // 2
        cy = y + h // 2
        x1 = cx - side // 2
        y1 = cy - side // 2
        x2 = x1 + side
        y2 = y1 + side
    else:
        pad = int(padding * max(w, h))
        x1 = x - pad
        y1 = y - pad
        x2 = x + w + pad
        y2 = y + h + pad
    return _clip_bbox(x1, y1, x2, y2, w_img, h_img)


def _resize_crop(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, size: int) -> np.ndarray | None:
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


class FaceCropper:
    def __init__(self, config: FaceCropConfig | None = None) -> None:
        self.config = config or FaceCropConfig()
        if self.config.human_only and self.config.method in {"haar", "mediapipe"}:
            raise ValueError(
                "human_only=True requires method='yunet' (OpenCV YuNet human-face detector). "
                f"Got method={self.config.method!r}."
            )
        self._haar: cv2.CascadeClassifier | None = None
        self._mp_detector: Any = None
        self._mp_api: str | None = None
        self._mp_image_module: Any = None
        self._yunet: cv2.FaceDetectorYN | None = None
        self._yunet_input_size: tuple[int, int] | None = None
        self.last_detect_stats: dict[str, int] = {"raw": 0, "kept": 0, "rejected_small": 0}

    @property
    def method(self) -> str:
        return self.config.method

    def reset_detect_stats(self) -> None:
        self.last_detect_stats = {"raw": 0, "kept": 0, "rejected_small": 0}

    def accumulate_detect_stats(self, stats: dict[str, int] | None = None) -> dict[str, int]:
        src = stats or self.last_detect_stats
        return {
            "raw": int(src.get("raw", 0)),
            "kept": int(src.get("kept", 0)),
            "rejected_small": int(src.get("rejected_small", 0)),
        }

    def classify_empty_face_status(
        self,
        *,
        unique_usable_frames: int,
        min_faces: int,
        raw_detections: int,
        rejected_small: int,
    ) -> str:
        """Map face sampling outcomes to an explicit gate status (never overuse no_human_face)."""
        if unique_usable_frames >= min_faces:
            return "ok"
        if unique_usable_frames == 0 and rejected_small > 0:
            return FACE_TOO_SMALL_STATUS
        if unique_usable_frames == 0 and raw_detections == 0:
            return self.no_face_status()
        if 0 < unique_usable_frames < min_faces:
            if rejected_small > unique_usable_frames:
                return FACE_TOO_SMALL_STATUS
            return INSUFFICIENT_FACE_SAMPLES_STATUS
        return self.no_face_status()

    def _haar_cascade(self) -> cv2.CascadeClassifier:
        if self._haar is None:
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(path)
        return self._haar

    def _mp_face_detector(self) -> Any:
        if self._mp_detector is None:
            try:
                import mediapipe as mp
            except ImportError as exc:
                raise ImportError(
                    "MediaPipe is required for --crop-method mediapipe. "
                    "Install with: pip install mediapipe"
                ) from exc

            if hasattr(mp, "solutions"):
                self._mp_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=self.config.mediapipe_model_selection,
                    min_detection_confidence=self.config.mediapipe_min_confidence,
                )
                self._mp_api = "solutions"
            else:
                from mediapipe.tasks import python as mp_tasks
                from mediapipe.tasks.python import vision

                model_path = _ensure_mediapipe_model(self.config.mediapipe_model_selection)
                options = vision.FaceDetectorOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
                    min_detection_confidence=self.config.mediapipe_min_confidence,
                )
                try:
                    self._mp_detector = vision.FaceDetector.create_from_options(options)
                except OSError as exc:
                    if "libGLES" in str(exc) or "libEGL" in str(exc):
                        raise OSError(
                            "MediaPipe Tasks API requires OpenGL ES libs on headless Linux.\n"
                            "  Ubuntu 24.04+: sudo apt-get install -y libgles2 libegl1\n"
                            "  Older Ubuntu: sudo apt-get install -y libgles2-mesa libegl1-mesa\n"
                            "Or pin MediaPipe with solutions API (no GLES):\n"
                            "  pip install 'mediapipe==0.10.30'"
                        ) from exc
                    raise
                self._mp_api = "tasks"
                self._mp_image_module = mp
        return self._mp_detector

    def close(self) -> None:
        if self._mp_detector is not None:
            self._mp_detector.close()
            self._mp_detector = None
            self._mp_api = None
            self._mp_image_module = None
        self._yunet = None
        self._yunet_input_size = None

    def no_face_status(self) -> str:
        return NO_HUMAN_FACE_STATUS if self.config.human_only else "no_face"

    def _yunet_detector(self, w_img: int, h_img: int) -> cv2.FaceDetectorYN:
        size = (w_img, h_img)
        if self._yunet is None or self._yunet_input_size != size:
            model_path = _ensure_yunet_model()
            self._yunet = cv2.FaceDetectorYN.create(
                str(model_path),
                "",
                size,
                self.config.yunet_score_threshold,
                self.config.yunet_nms_threshold,
                5000,
            )
            self._yunet_input_size = size
        else:
            self._yunet.setInputSize(size)
        return self._yunet

    def detect_all_human_face_bboxes(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        """Return usable human face bboxes as (x, y, w, h), largest first.

        Faces smaller than ``min_face_side_px`` are rejected and counted in
        ``last_detect_stats`` so callers can emit FACE_TOO_SMALL instead of NO_HUMAN.
        """
        if self.config.method == "yunet":
            raw = self._detect_all_yunet(frame)
        elif self.config.method == "mediapipe":
            raw = self._detect_all_mediapipe(frame)
        else:
            raw = self._detect_all_haar(frame)
        return self._filter_min_face_size(raw)

    def detect_human_face_bbox(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return (x, y, w, h) for the largest human face, or None."""
        bboxes = self.detect_all_human_face_bboxes(frame)
        return bboxes[0] if bboxes else None

    def _filter_min_face_size(
        self, bboxes: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        min_side = max(1, int(self.config.min_face_side_px))
        kept: list[tuple[int, int, int, int]] = []
        rejected = 0
        for bbox in bboxes:
            _x, _y, w, h = bbox
            if min(int(w), int(h)) < min_side:
                rejected += 1
                continue
            kept.append(bbox)
        self.last_detect_stats = {
            "raw": len(bboxes),
            "kept": len(kept),
            "rejected_small": rejected,
        }
        return kept

    def _detect_all_yunet(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h_img, w_img = frame.shape[:2]
        detector = self._yunet_detector(w_img, h_img)
        _, faces = detector.detect(frame)
        if faces is None or len(faces) == 0:
            return []
        ordered = sorted(
            faces,
            key=lambda f: float(f[2]) * float(f[3]),
            reverse=True,
        )
        return [
            (int(face[0]), int(face[1]), int(face[2]), int(face[3]))
            for face in ordered
        ]

    def _detect_all_haar(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._haar_cascade().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return []
        ordered = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in ordered]

    def _detect_all_mediapipe(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        h_img, w_img = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detector = self._mp_face_detector()
        bboxes: list[tuple[int, int, int, int]] = []

        if getattr(self, "_mp_api", None) == "tasks":
            mp = self._mp_image_module
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = detector.detect(mp_image)
            for detection in results.detections or []:
                box = detection.bounding_box
                bboxes.append((int(box.origin_x), int(box.origin_y), int(box.width), int(box.height)))
        else:
            results = detector.process(rgb)
            for detection in results.detections or []:
                box = detection.location_data.relative_bounding_box
                bboxes.append(
                    (
                        int(box.xmin * w_img),
                        int(box.ymin * h_img),
                        int(box.width * w_img),
                        int(box.height * h_img),
                    )
                )

        return sorted(bboxes, key=lambda b: b[2] * b[3], reverse=True)

    def crop_from_bbox(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x, y, w, h = bbox
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = _apply_padding_square(
            x, y, w, h,
            padding=self.config.padding,
            square=self.config.square,
            h_img=h_img,
            w_img=w_img,
        )
        return _resize_crop(frame, x1, y1, x2, y2, self.config.size)

    def crop_all(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Return a crop entry for every detected face in the frame."""
        entries: list[dict[str, Any]] = []
        for face_index, bbox in enumerate(self.detect_all_human_face_bboxes(frame)):
            crop = self.crop_from_bbox(frame, bbox)
            if crop is None:
                continue
            entries.append(
                {
                    "face_index": face_index,
                    "bbox": bbox,
                    "crop": crop,
                }
            )
        return entries

    def __enter__(self) -> FaceCropper:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def crop(self, frame: np.ndarray) -> np.ndarray | None:
        if self.config.method == "yunet":
            return self._crop_yunet(frame)
        if self.config.method == "mediapipe":
            return self._crop_mediapipe(frame)
        return self._crop_haar(frame)

    def _crop_yunet(self, frame: np.ndarray) -> np.ndarray | None:
        bbox = self.detect_human_face_bbox(frame)
        if bbox is None:
            return None
        return self.crop_from_bbox(frame, bbox)

    def _crop_haar(self, frame: np.ndarray) -> np.ndarray | None:
        bboxes = self._detect_all_haar(frame)
        if not bboxes:
            return None
        return self.crop_from_bbox(frame, bboxes[0])

    def _crop_mediapipe(self, frame: np.ndarray) -> np.ndarray | None:
        bboxes = self._detect_all_mediapipe(frame)
        if not bboxes:
            return None
        return self.crop_from_bbox(frame, bboxes[0])

    def to_metadata(self) -> dict[str, Any]:
        return {
            "crop_method": self.config.method,
            "crop_size": self.config.size,
            "crop_padding": self.config.padding,
            "crop_square": self.config.square,
            "human_only": self.config.human_only,
            "face_gate": "human_yunet" if self.config.method == "yunet" else self.config.method,
            "yunet_score_threshold": self.config.yunet_score_threshold,
            "min_sample_faces": self.config.min_sample_faces,
            "min_face_side_px": self.config.min_face_side_px,
            "multi_face": True,
        }


def create_face_cropper(
    *,
    method: str | None = None,
    size: int = 256,
    padding: float = 0.2,
    square: bool = True,
    human_only: bool = True,
    yunet_score_threshold: float = 0.75,
    min_sample_faces: int = 4,
    min_face_side_px: int = DEFAULT_MIN_FACE_SIDE_PX,
) -> FaceCropper:
    resolved_method = method or ("yunet" if human_only else "haar")
    return FaceCropper(
        FaceCropConfig(
            method=resolved_method,
            size=size,
            padding=padding,
            square=square,
            human_only=human_only,
            yunet_score_threshold=yunet_score_threshold,
            min_sample_faces=min_sample_faces,
            min_face_side_px=min_face_side_px,
        )
    )


def crop_face(
    frame: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    size: int = 256,
    *,
    padding: float = 0.2,
    square: bool = False,
    face_cropper: FaceCropper | None = None,
) -> np.ndarray | None:
    """Crop a face patch. Prefer face_cropper (YuNet human gate) over legacy Haar."""
    if face_cropper is not None:
        return face_cropper.crop(frame)
    del face_cascade
    cropper = create_face_cropper(method="haar", size=size, padding=padding, square=square, human_only=False)
    return cropper.crop(frame)
