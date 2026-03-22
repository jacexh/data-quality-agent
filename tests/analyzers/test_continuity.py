import numpy as np
import pytest
import cv2
from agent.analyzers.continuity import ContinuityAnalyzer


def _make_frame(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _pan_frames(n: int = 6, shift: int = 4) -> list[np.ndarray]:
    """Smooth horizontal pan — high magnitude but coherent direction."""
    base = _make_frame(64, 128)
    return [base[:, i*shift:i*shift+64] for i in range(n)]


def _jumpy_frames(n: int = 10) -> list[np.ndarray]:
    """Independent random frames — chaotic, incoherent flow."""
    return [_make_frame(seed=i) for i in range(n)]


def test_smooth_scores_higher_than_jumpy(sharp_frames):
    """Identical frames → zero flow → high continuity score."""
    analyzer = ContinuityAnalyzer()
    smooth = analyzer.analyze(sharp_frames)       # identical frames
    jumpy = analyzer.analyze(_jumpy_frames())
    assert smooth["score"] > jumpy["score"]


def test_score_normalized(sharp_frames):
    result = ContinuityAnalyzer().analyze(sharp_frames)
    assert 0.0 <= result["score"] <= 1.0


def test_empty_frames_returns_zero():
    result = ContinuityAnalyzer().analyze([])
    assert result["score"] == 0.0
    assert result["detail"]["frame_count"] == 0


def test_single_frame_returns_perfect(sharp_frames):
    """Single frame → no pairs → nothing to be discontinuous → score 1.0."""
    result = ContinuityAnalyzer().analyze([sharp_frames[0]])
    assert result["score"] == 1.0


def test_name():
    assert ContinuityAnalyzer().name() == "continuity"


def test_detail_keys(sharp_frames):
    d = ContinuityAnalyzer().analyze(sharp_frames)["detail"]
    assert "mean_flow_magnitude" in d
    assert "flow_magnitude_std" in d
    assert "flow_direction_std" in d
    assert "discontinuity_frames" in d
    assert "frame_count" in d


def test_flow_direction_std_low_for_smooth_pan():
    """Uniform horizontal pan → all flow vectors point the same way → low direction std."""
    result = ContinuityAnalyzer().analyze(_pan_frames())
    assert result["detail"]["flow_direction_std"] < 0.5  # radians


def test_flow_direction_std_high_for_chaotic_frames():
    """Random independent frames → chaotic flow directions → high direction std."""
    result = ContinuityAnalyzer().analyze(_jumpy_frames())
    # Chaotic flow should have higher direction std than smooth pan
    pan_result = ContinuityAnalyzer().analyze(_pan_frames())
    assert result["detail"]["flow_direction_std"] > pan_result["detail"]["flow_direction_std"]


def test_flow_magnitude_std_zero_for_identical_frames(sharp_frames):
    """All identical frames → zero flow every pair → magnitude std = 0."""
    result = ContinuityAnalyzer().analyze(sharp_frames)
    assert result["detail"]["flow_magnitude_std"] == pytest.approx(0.0, abs=1e-4)


def test_smooth_pan_scores_higher_than_chaotic():
    """Smooth pan should not be penalized the same way as chaotic jumps."""
    analyzer = ContinuityAnalyzer()
    assert analyzer.analyze(_pan_frames())["score"] >= analyzer.analyze(_jumpy_frames())["score"]
