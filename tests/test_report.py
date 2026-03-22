import uuid
from datetime import datetime
import pytest
from agent.report import ReportBuilder, evaluate_strategy
from agent.config import Settings
from agent.analyzers.base import CameraResult, AudioResult


def _settings(**kwargs):
    return Settings(
        clarity_threshold=kwargs.get("clarity_threshold", 0.6),
        continuity_threshold=kwargs.get("continuity_threshold", 0.6),
        minimum_duration_seconds=kwargs.get("minimum_duration_seconds", 1.0),
        camera_pass_strategy=kwargs.get("camera_pass_strategy", "all"),
        audio_pass_strategy=kwargs.get("audio_pass_strategy", "all"),
        anthropic_api_key="fake",
    )


def _good_camera(topic: str = "/cam") -> CameraResult:
    return CameraResult(
        topic=topic,
        frame_count=60,
        clarity={"score": 0.9, "method": "laplacian+fft", "detail": {}},
        continuity={"score": 0.9, "method": "optical_flow", "detail": {}},
        face={"has_face": False, "face_count": 0, "face_frame_ratio": 0.0, "max_confidence": 0.0},
        gait={"has_human_gait": False, "person_frame_ratio": 0.0, "max_detection_weight": 0.0},
        llm_assessment=None,
        llm_skipped_reason="all_detectors_clear_no_borderline_scores",
        passed=True,
        failure_reasons=[],
        analyzer_errors=[],
    )


def _good_audio(topic: str = "/audio") -> AudioResult:
    return AudioResult(
        topic=topic,
        audio_frame_count=400,
        voice={"has_human_voice": False, "speech_frame_ratio": 0.0},
        llm_assessment=None,
        llm_skipped_reason="all_detectors_clear_no_borderline_scores",
        passed=True,
        failure_reasons=[],
        analyzer_errors=[],
    )


# ── evaluate_strategy ────────────────────────────────────────────────────────

def test_strategy_all_passes_when_all_true():
    assert evaluate_strategy([True, True, True], "all") is True


def test_strategy_all_fails_when_any_false():
    assert evaluate_strategy([True, False, True], "all") is False


def test_strategy_any_passes_when_one_true():
    assert evaluate_strategy([False, True, False], "any") is True


def test_strategy_any_fails_when_all_false():
    assert evaluate_strategy([False, False], "any") is False


def test_strategy_majority_passes():
    assert evaluate_strategy([True, True, False], "majority") is True


def test_strategy_majority_fails():
    assert evaluate_strategy([True, False, False], "majority") is False


def test_strategy_empty_list_returns_false():
    """Zero topics → failure. No silent pass."""
    assert evaluate_strategy([], "all") is False
    assert evaluate_strategy([], "any") is False
    assert evaluate_strategy([], "majority") is False


def test_strategy_unknown_raises():
    with pytest.raises(ValueError):
        evaluate_strategy([True], "unknown")


# ── ReportBuilder.build() ────────────────────────────────────────────────────

def test_overall_passed_when_all_topics_pass():
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[_good_camera()],
        audio_results=[_good_audio()],
        analyzer_errors=[],
    )
    assert report["overall_passed"] is True
    assert report["cameras"][0]["passed"] is True
    assert report["audios"][0]["passed"] is True


def test_overall_fails_when_camera_fails():
    bad_cam = _good_camera()
    bad_cam["passed"] = False
    bad_cam["failure_reasons"] = ["clarity"]
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[bad_cam],
        audio_results=[_good_audio()],
        analyzer_errors=[],
    )
    assert report["overall_passed"] is False


def test_overall_fails_when_audio_fails():
    bad_audio = _good_audio()
    bad_audio["passed"] = False
    bad_audio["failure_reasons"] = ["has_human_voice"]
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[_good_camera()],
        audio_results=[bad_audio],
        analyzer_errors=[],
    )
    assert report["overall_passed"] is False


def test_zero_cameras_fails():
    """No camera topics → failure (evaluate_strategy returns False for empty list)."""
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[],
        audio_results=[_good_audio()],
        analyzer_errors=[],
    )
    assert report["overall_passed"] is False


def test_zero_audios_fails():
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[_good_camera()],
        audio_results=[],
        analyzer_errors=[],
    )
    assert report["overall_passed"] is False


def test_any_strategy_passes_when_one_camera_passes():
    bad_cam = _good_camera("/cam1")
    bad_cam["passed"] = False
    good_cam = _good_camera("/cam2")
    builder = ReportBuilder(_settings(camera_pass_strategy="any"))
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[bad_cam, good_cam],
        audio_results=[],  # no audio → fail
        analyzer_errors=[],
    )
    # cameras: "any" → True; audios: empty → False → overall False
    assert report["overall_passed"] is False
    assert report["camera_pass_strategy"] == "any"


def test_multi_camera_all_strategy():
    cam1 = _good_camera("/cam1")
    cam2 = _good_camera("/cam2")
    builder = ReportBuilder(_settings(camera_pass_strategy="all"))
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[cam1, cam2],
        audio_results=[_good_audio()],
        analyzer_errors=[],
    )
    assert report["overall_passed"] is True
    assert len(report["cameras"]) == 2


def test_report_id_is_valid_uuid4():
    builder = ReportBuilder(_settings())
    report = builder.build("f.mcap", 5.0, [_good_camera()], [_good_audio()], [])
    assert uuid.UUID(report["report_id"]).version == 4


def test_report_id_is_unique():
    builder = ReportBuilder(_settings())
    r1 = builder.build("f.mcap", 5.0, [_good_camera()], [_good_audio()], [])
    r2 = builder.build("f.mcap", 5.0, [_good_camera()], [_good_audio()], [])
    assert r1["report_id"] != r2["report_id"]


def test_analyzed_at_is_iso8601_utc():
    builder = ReportBuilder(_settings())
    report = builder.build("f.mcap", 5.0, [_good_camera()], [_good_audio()], [])
    dt = datetime.fromisoformat(report["analyzed_at"].replace("Z", "+00:00"))
    assert dt.tzinfo is not None


def test_short_duration_fails():
    builder = ReportBuilder(_settings(minimum_duration_seconds=2.0))
    report = builder.build("f.mcap", 0.5, [_good_camera()], [_good_audio()], [])
    assert report["overall_passed"] is False
    assert "duration_too_short" in report["failure_reasons"]


def test_analyzer_errors_in_report():
    builder = ReportBuilder(_settings())
    report = builder.build("f.mcap", 5.0, [_good_camera()], [_good_audio()], ["mcap_extraction"])
    assert "mcap_extraction" in report["analyzer_errors"]
