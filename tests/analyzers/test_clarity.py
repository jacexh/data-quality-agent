import numpy as np
import pytest
from agent.analyzers.clarity import ClarityAnalyzer


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
    assert result["method"] == "laplacian+tenengrad"


def test_detail_keys(sharp_data):
    result = ClarityAnalyzer().analyze(sharp_data)
    d = result["detail"]
    assert "mean_laplacian_variance" in d
    assert "mean_tenengrad" in d
    assert "frame_count" in d
