# scripts/profile/__init__.py
#
# ForenShield AI — 영상 품질·압축 프로파일 패키지 진입점.
#
# 이 디렉터리(profile/)는 AI 분석 전 영상이 분석에 적합한지 판단하는
# 스크립트들을 모아 둔 곳입니다.
#
# 역할:
#   - video_readiness.py 에서 구현한 함수·타입을 외부에서 import 할 수 있게 re-export
#   - 예: from profile.video_readiness import analyze_video_readiness
#         (scripts/ 를 PYTHONPATH 에 넣었을 때)
#
# 관련 노트북: docs/notebooks/test/실시간_블록_*.ipynb

"""Video compression / quality profiling scripts."""

from .video_readiness import (
    ReadinessThresholds,
    VideoReadinessResult,
    analyze_video_readiness,
    calculate_blur_score,
    calculate_blockiness_heatmap,
    calculate_fft_grid_peak,
)

__all__ = [
    "ReadinessThresholds",
    "VideoReadinessResult",
    "analyze_video_readiness",
    "calculate_blur_score",
    "calculate_blockiness_heatmap",
    "calculate_fft_grid_peak",
]
