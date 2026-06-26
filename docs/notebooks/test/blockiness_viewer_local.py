"""
로컬용 영상 화질/압축 컨텍스트 뷰어 (Colab google.colab 제거 버전)

사용:
  python blockiness_viewer_local.py path/to/video.mp4
  python blockiness_viewer_local.py 0
  python blockiness_viewer_local.py video.mp4 --no-display --json-out summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def calculate_blockiness_heatmap(gray_frame: np.ndarray):
    h, w = gray_frame.shape
    mask = np.zeros((h, w), dtype=np.float32)
    gray_float = gray_frame.astype(np.float32)

    h_bnd = np.arange(7, h - 1, 8)
    w_bnd = np.arange(7, w - 1, 8)

    diff_h = np.abs(gray_float[h_bnd, :] - gray_float[h_bnd + 1, :])
    diff_v = np.abs(gray_float[:, w_bnd] - gray_float[:, w_bnd + 1])

    mask[h_bnd, :] += diff_h
    mask[h_bnd + 1, :] += diff_h
    mask[:, w_bnd] += diff_v
    mask[:, w_bnd + 1] += diff_v

    raw_score = (np.mean(diff_h) + np.mean(diff_v)) / 2.0
    current_score = min(raw_score * 10.0, 100.0)

    heatmap_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
    heatmap_absolute = np.clip(heatmap_blurred * 10.0, 0, 255).astype(np.uint8)
    color_map = cv2.applyColorMap(heatmap_absolute, cv2.COLORMAP_JET)

    return color_map, current_score


def calculate_blur_score(gray_frame: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())


def calculate_fft_grid_peak(gray_frame: np.ndarray) -> float:
    f_shift = np.fft.fftshift(np.fft.fft2(gray_frame))
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
    mask_radius = 30
    y_idx, x_idx = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
    high_freq_region = magnitude_spectrum[dist_from_center > mask_radius]

    if len(high_freq_region) == 0:
        return 0.0
    return min(float(np.std(high_freq_region) / 100.0), 1.0)


def build_summary(metrics_timeline: dict, total_frames: int) -> dict:
    if not metrics_timeline["blur"]:
        return {"samples": 0, "total_frames": total_frames}

    avg_blur = float(np.mean(metrics_timeline["blur"]))
    avg_block = float(np.mean(metrics_timeline["blockiness"]))
    avg_fft = float(np.mean(metrics_timeline["fft_peak"]))

    return {
        "total_frames": total_frames,
        "samples": len(metrics_timeline["blur"]),
        "quality_metrics": {
            "globalBlurScore": avg_blur,
            "globalBlockiness": round(avg_block / 100.0, 4),
            "fftGridPeakStrength": round(avg_fft, 4),
        },
        "interpretation": {
            "isHeavilyCompressed": avg_block >= 30 or avg_blur < 100,
            "recommendation": "NORMAL_PROCESS",
        },
    }


def run_blockiness_viewer(
    video_path,
    *,
    sample_every: int = 10,
    show_window: bool = True,
    wait_ms: int = 1,
) -> dict | None:
    source = int(video_path) if str(video_path).isdigit() else str(video_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"'{video_path}' 영상을 열 수 없습니다.", file=sys.stderr)
        return None

    print("프레임별 종합 화질/압축 컨텍스트 분석을 시작합니다...")
    if show_window:
        print("종료: 영상 창에서 'q' 키")

    frame_idx = 0
    metrics_timeline = {"blur": [], "blockiness": [], "fft_peak": []}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_every != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        heatmap, blockiness_score = calculate_blockiness_heatmap(gray)
        blur_score = calculate_blur_score(gray)
        fft_peak_score = calculate_fft_grid_peak(gray)

        metrics_timeline["blockiness"].append(blockiness_score)
        metrics_timeline["blur"].append(blur_score)
        metrics_timeline["fft_peak"].append(fft_peak_score)

        if not show_window:
            print(
                f"Frame {frame_idx}: blur={blur_score:.1f} "
                f"block={blockiness_score:.1f} fft={fft_peak_score:.2f}"
            )
            continue

        target_w, target_h = 480, 270
        frame_resized = cv2.resize(frame, (target_w, target_h))
        heatmap_resized = cv2.resize(heatmap, (target_w, target_h))
        overlay = cv2.addWeighted(frame_resized, 0.4, heatmap_resized, 0.6, 0)

        cv2.putText(
            frame_resized, f"Frame: {frame_idx}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        blur_color = (0, 0, 255) if blur_score < 100 else (0, 255, 0)
        cv2.putText(
            frame_resized, f"Blur: {blur_score:.1f}", (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, blur_color, 2,
        )
        fft_color = (0, 0, 255) if fft_peak_score > 0.4 else (0, 255, 255)
        cv2.putText(
            frame_resized, f"FFT Peak: {fft_peak_score:.2f}", (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, fft_color, 2,
        )
        status_color = (0, 0, 255) if blockiness_score > 30 else (0, 255, 255)
        cv2.putText(
            overlay, f"Block Loss: {blockiness_score:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2,
        )

        combined = np.hstack((frame_resized, overlay))
        cv2.imshow("Blockiness / Quality Context", combined)
        if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
            print("사용자 중단 (q)")
            break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    summary = build_summary(metrics_timeline, frame_idx)
    if summary.get("samples", 0) > 0:
        q = summary["quality_metrics"]
        print(f"\n=== 요약 (samples={summary['samples']}) ===")
        print(f"Blur avg: {q['globalBlurScore']:.1f}")
        print(f"Blockiness avg: {q['globalBlockiness'] * 100:.1f}")
        print(f"FFT peak avg: {q['fftGridPeakStrength']:.2f}")

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="영상 화질/압축 컨텍스트 분석")
    parser.add_argument("video", help="영상 경로 또는 웹캠 번호(0)")
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args(argv)

    summary = run_blockiness_viewer(
        args.video,
        sample_every=args.sample_every,
        show_window=not args.no_display,
    )
    if summary is None:
        return 1

    if args.json_out:
        out = Path(args.json_out)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"JSON 저장: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
