from __future__ import annotations

from gpu_worker.schemas import SuspiciousSegmentItem


def build_suspicious_segments(
    points: list[tuple[float, float, float]],
    *,
    threshold: float,
    min_duration_sec: float = 1.0,
    reason: str,
) -> list[SuspiciousSegmentItem]:
    """(start_time, end_time, risk_score) 리스트에서 연속 고위험 구간 생성."""
    if not points:
        return []

    segments: list[SuspiciousSegmentItem] = []
    active_start: float | None = None
    active_end: float | None = None
    active_max = 0.0

    def flush() -> None:
        nonlocal active_start, active_end, active_max
        if active_start is None or active_end is None:
            return
        if (active_end - active_start) >= min_duration_sec:
            segments.append(
                SuspiciousSegmentItem(
                    startTime=round(active_start, 3),
                    endTime=round(active_end, 3),
                    maxRiskScore=round(active_max, 4),
                    reason=reason,
                )
            )
        active_start = None
        active_end = None
        active_max = 0.0

    for start_time, end_time, risk in sorted(points, key=lambda row: row[0]):
        if risk >= threshold:
            if active_start is None:
                active_start = start_time
                active_end = end_time
                active_max = risk
            elif start_time <= (active_end or start_time) + 0.5:
                active_end = max(active_end or end_time, end_time)
                active_max = max(active_max, risk)
            else:
                flush()
                active_start = start_time
                active_end = end_time
                active_max = risk
        else:
            flush()
    flush()
    return segments
