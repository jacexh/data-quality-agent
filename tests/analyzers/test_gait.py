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


def test_name():
    assert GaitDetector().name() == "gait"
