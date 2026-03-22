import numpy as np
import pytest
import cv2
from agent.analyzers.clarity import ClarityAnalyzer


def _make_sharp_frame(h: int = 64, w: int = 64) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[::4, :] = 255
    return frame


def _make_blurry_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _make_checkerboard_frame(h: int = 64, w: int = 64, block: int = 8) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(0, h, block):
        for c in range(0, w, block):
            if (r // block + c // block) % 2 == 0:
                frame[r:r+block, c:c+block] = 255
    return frame


def _make_motion_blurred_frame(h: int = 64, w: int = 64) -> np.ndarray:
    sharp = _make_checkerboard_frame(h, w)
    kernel = np.zeros((1, 15))
    kernel[0, :] = 1.0 / 15
    return cv2.filter2D(sharp, -1, kernel)


def test_sharp_frames_score_higher_than_blurry(sharp_frames, blurry_frames):
    analyzer = ClarityAnalyzer()
    assert analyzer.analyze(sharp_frames)["score"] > analyzer.analyze(blurry_frames)["score"]


def test_score_is_normalized(sharp_frames):
    result = ClarityAnalyzer().analyze(sharp_frames)
    assert 0.0 <= result["score"] <= 1.0


def test_empty_frames_returns_zero():
    result = ClarityAnalyzer().analyze([])
    assert result["score"] == 0.0
    assert result["detail"]["frame_count"] == 0


def test_name():
    assert ClarityAnalyzer().name() == "clarity"


def test_method_field(sharp_frames):
    result = ClarityAnalyzer().analyze(sharp_frames)
    assert result["method"] == "laplacian+fft"


def test_detail_keys(sharp_frames):
    d = ClarityAnalyzer().analyze(sharp_frames)["detail"]
    assert "mean_laplacian_variance" in d
    assert "fft_high_freq_ratio" in d
    assert "frame_score_std" in d
    assert "frame_count" in d


def test_fft_high_freq_ratio_higher_for_sharp_than_blurry():
    analyzer = ClarityAnalyzer()
    sharp = [_make_sharp_frame() for _ in range(5)]
    blurry = [_make_blurry_frame() for _ in range(5)]
    assert analyzer.analyze(sharp)["detail"]["fft_high_freq_ratio"] > \
           analyzer.analyze(blurry)["detail"]["fft_high_freq_ratio"]


def test_motion_blurred_scores_lower_than_sharp():
    analyzer = ClarityAnalyzer()
    sharp = [_make_sharp_frame() for _ in range(5)]
    motion = [_make_motion_blurred_frame() for _ in range(5)]
    assert analyzer.analyze(sharp)["score"] > analyzer.analyze(motion)["score"]


def test_frame_score_std_is_zero_for_uniform_frames():
    result = ClarityAnalyzer().analyze([_make_sharp_frame() for _ in range(8)])
    assert result["detail"]["frame_score_std"] == pytest.approx(0.0, abs=1e-4)


def test_frame_score_std_is_positive_for_mixed_frames():
    frames = [_make_sharp_frame() for _ in range(4)] + [_make_blurry_frame() for _ in range(4)]
    result = ClarityAnalyzer().analyze(frames)
    assert result["detail"]["frame_score_std"] > 0.0


def test_single_frame_std_is_zero():
    result = ClarityAnalyzer().analyze([_make_sharp_frame()])
    assert result["detail"]["frame_score_std"] == pytest.approx(0.0, abs=1e-4)
