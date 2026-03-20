import numpy as np
import pytest
from agent.analyzers.gait import GaitDetector
from agent.analyzers.base import ExtractedData


def test_empty_frames_no_gait(empty_data):
    detector = GaitDetector()
    result = detector.analyze(empty_data)
    assert result["has_human_gait"] is False


def test_small_uniform_frames_no_gait(blurry_data):
    """64×64 uniform frames contain no walking person."""
    detector = GaitDetector()
    result = detector.analyze(blurry_data)
    assert result["has_human_gait"] is False


def test_real_person_detected(person_data):
    """Street photo with pedestrian — HOG must detect at least one person (positive recall test)."""
    detector = GaitDetector()
    result = detector.analyze(person_data)
    assert result["has_human_gait"] is True


def test_all_frames_below_minimum_size_no_crash():
    """All frames below HOG minimum 128×64 → skipped, returns False without crashing (spec contract)."""
    tiny = np.zeros((32, 32, 3), dtype=np.uint8)
    data = ExtractedData(frames=[tiny, tiny], audio_frames=None, sensor_series={}, duration_seconds=1.0)
    detector = GaitDetector()
    result = detector.analyze(data)
    assert result["has_human_gait"] is False


def test_name():
    assert GaitDetector().name() == "gait"


def test_person_frame_ratio_zero_when_no_person(blurry_data):
    result = GaitDetector().analyze(blurry_data)
    assert result["person_frame_ratio"] == pytest.approx(0.0)


def test_person_frame_ratio_positive_when_person_present(person_data):
    """Single-frame person data → ratio should be 1.0."""
    result = GaitDetector().analyze(person_data)
    assert result["person_frame_ratio"] == pytest.approx(1.0)


def test_person_frame_ratio_partial(person_data):
    """Person frame mixed with blank frames → ratio between 0 and 1."""
    blank = np.zeros((person_data["frames"][0].shape), dtype=np.uint8)
    mixed = ExtractedData(
        frames=[person_data["frames"][0], blank, blank],
        audio_frames=None, sensor_series={}, duration_seconds=1.0,
    )
    result = GaitDetector().analyze(mixed)
    assert 0.0 < result["person_frame_ratio"] < 1.0


def test_max_detection_weight_zero_when_no_person(blurry_data):
    result = GaitDetector().analyze(blurry_data)
    assert result["max_detection_weight"] == pytest.approx(0.0)


def test_max_detection_weight_positive_when_person_present(person_data):
    result = GaitDetector().analyze(person_data)
    assert result["max_detection_weight"] > 0.0
