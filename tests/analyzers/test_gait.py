from agent.analyzers.gait import GaitDetector


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
    import numpy as np
    from agent.analyzers.base import ExtractedData
    tiny = np.zeros((32, 32, 3), dtype=np.uint8)
    data = ExtractedData(frames=[tiny, tiny], audio_frames=None, sensor_series={}, duration_seconds=1.0)
    detector = GaitDetector()
    result = detector.analyze(data)
    assert result["has_human_gait"] is False


def test_name():
    assert GaitDetector().name() == "gait"
