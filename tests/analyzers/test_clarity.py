import numpy as np
import pytest
import cv2
from agent.analyzers.clarity import ClarityAnalyzer
from agent.analyzers.base import ExtractedData


def _make_sharp_frame(h: int = 64, w: int = 64) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[::4, :] = 255
    return frame


def _make_blurry_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _make_checkerboard_frame(h: int = 64, w: int = 64, block: int = 8) -> np.ndarray:
    """2D checkerboard — high-frequency content in both X and Y."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(0, h, block):
        for c in range(0, w, block):
            if (r // block + c // block) % 2 == 0:
                frame[r:r+block, c:c+block] = 255
    return frame


def _make_motion_blurred_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """2D checkerboard smeared horizontally — simulates camera motion blur."""
    sharp = _make_checkerboard_frame(h, w)
    kernel = np.zeros((1, 15))
    kernel[0, :] = 1.0 / 15
    return cv2.filter2D(sharp, -1, kernel)


def test_sharp_frames_score_higher_than_blurry(sharp_data, blurry_data):
    analyzer = ClarityAnalyzer()
    sharp_result = analyzer.analyze(sharp_data)
    blurry_result = analyzer.analyze(blurry_data)
    assert sharp_result["score"] > blurry_result["score"]


def test_score_is_normalized(sharp_data):
    analyzer = ClarityAnalyzer()
    result = analyzer.analyze(sharp_data)
    assert 0.0 <= result["score"] <= 1.0


def test_empty_frames_returns_zero(empty_data):
    analyzer = ClarityAnalyzer()
    result = analyzer.analyze(empty_data)
    assert result["score"] == 0.0
    assert result["detail"]["frame_count"] == 0


def test_name():
    assert ClarityAnalyzer().name() == "clarity"


def test_method_field(sharp_data):
    result = ClarityAnalyzer().analyze(sharp_data)
    assert result["method"] == "laplacian+fft"


def test_detail_keys(sharp_data):
    result = ClarityAnalyzer().analyze(sharp_data)
    d = result["detail"]
    assert "mean_laplacian_variance" in d
    assert "fft_high_freq_ratio" in d
    assert "frame_score_std" in d
    assert "frame_count" in d


def test_fft_high_freq_ratio_higher_for_sharp_than_blurry():
    """FFT metric should distinguish edge-rich (sharp) from flat (blurry) frames."""
    analyzer = ClarityAnalyzer()
    sharp = ExtractedData(
        frames=[_make_sharp_frame() for _ in range(5)],
        audio_frames=None, sensor_series={}, duration_seconds=1.0,
    )
    blurry = ExtractedData(
        frames=[_make_blurry_frame() for _ in range(5)],
        audio_frames=None, sensor_series={}, duration_seconds=1.0,
    )
    assert analyzer.analyze(sharp)["detail"]["fft_high_freq_ratio"] > \
           analyzer.analyze(blurry)["detail"]["fft_high_freq_ratio"]


def test_motion_blurred_scores_lower_than_sharp():
    """Sharp checkerboard should score higher than horizontally motion-blurred version."""
    analyzer = ClarityAnalyzer()
    sharp = ExtractedData(
        frames=[_make_sharp_frame() for _ in range(5)],
        audio_frames=None, sensor_series={}, duration_seconds=1.0,
    )
    motion = ExtractedData(
        frames=[_make_motion_blurred_frame() for _ in range(5)],
        audio_frames=None, sensor_series={}, duration_seconds=1.0,
    )
    assert analyzer.analyze(sharp)["score"] > analyzer.analyze(motion)["score"]


def test_frame_score_std_is_zero_for_uniform_frames():
    """When all frames have identical sharpness, std should be 0."""
    analyzer = ClarityAnalyzer()
    data = ExtractedData(
        frames=[_make_sharp_frame() for _ in range(8)],
        audio_frames=None, sensor_series={}, duration_seconds=1.0,
    )
    result = analyzer.analyze(data)
    assert result["detail"]["frame_score_std"] == pytest.approx(0.0, abs=1e-4)


def test_frame_score_std_is_positive_for_mixed_frames():
    """Mixing sharp and blurry frames should produce non-zero std."""
    analyzer = ClarityAnalyzer()
    frames = [_make_sharp_frame() for _ in range(4)] + \
             [_make_blurry_frame() for _ in range(4)]
    data = ExtractedData(
        frames=frames, audio_frames=None, sensor_series={}, duration_seconds=1.0,
    )
    result = analyzer.analyze(data)
    assert result["detail"]["frame_score_std"] > 0.0


def test_single_frame_std_is_zero():
    """Single-frame input should have std of 0."""
    analyzer = ClarityAnalyzer()
    data = ExtractedData(
        frames=[_make_sharp_frame()],
        audio_frames=None, sensor_series={}, duration_seconds=0.1,
    )
    result = analyzer.analyze(data)
    assert result["detail"]["frame_score_std"] == pytest.approx(0.0, abs=1e-4)
