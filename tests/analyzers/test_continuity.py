import numpy as np
import pytest
from agent.analyzers.continuity import ContinuityAnalyzer
from agent.analyzers.base import ExtractedData


def _jumpy_data() -> ExtractedData:
    """Random texture frames — creates detectable optical flow."""
    np.random.seed(42)
    frames = []
    for i in range(10):
        frames.append(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
    return ExtractedData(frames=frames, audio_frames=None, sensor_series={}, duration_seconds=5.0)


def test_smooth_scores_higher_than_jumpy(sharp_data):
    """Identical frames → zero flow → high continuity score."""
    analyzer = ContinuityAnalyzer()
    smooth = analyzer.analyze(sharp_data)       # identical frames
    jumpy = analyzer.analyze(_jumpy_data())
    assert smooth["score"] > jumpy["score"]


def test_score_normalized(sharp_data):
    result = ContinuityAnalyzer().analyze(sharp_data)
    assert 0.0 <= result["score"] <= 1.0


def test_empty_frames_returns_zero(empty_data):
    result = ContinuityAnalyzer().analyze(empty_data)
    assert result["score"] == 0.0
    assert result["detail"]["frame_count"] == 0


def test_single_frame_returns_perfect(sharp_data):
    """Single frame → no pairs → nothing to be discontinuous → score 1.0."""
    data = ExtractedData(frames=[sharp_data["frames"][0]], audio_frames=None, sensor_series={}, duration_seconds=1.0)
    result = ContinuityAnalyzer().analyze(data)
    assert result["score"] == 1.0


def test_name():
    assert ContinuityAnalyzer().name() == "continuity"


def test_detail_keys(sharp_data):
    d = ContinuityAnalyzer().analyze(sharp_data)["detail"]
    assert "mean_flow_magnitude" in d
    assert "discontinuity_frames" in d
    assert "frame_count" in d
