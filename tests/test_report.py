import uuid
from datetime import datetime, timezone
import pytest
from agent.report import ReportBuilder
from agent.config import Settings


def _settings(**kwargs):
    return Settings(
        clarity_threshold=kwargs.get("clarity_threshold", 0.6),
        continuity_threshold=kwargs.get("continuity_threshold", 0.6),
        minimum_duration_seconds=kwargs.get("minimum_duration_seconds", 1.0),
        anthropic_api_key="fake",
    )


def _good_results():
    return {
        "clarity": {"score": 0.9, "method": "laplacian+tenengrad", "detail": {}},
        "continuity": {"score": 0.9, "method": "optical_flow", "detail": {}},
        "face": {"has_face": False, "face_count": 0},
        "voice": {"has_human_voice": False},
        "gait": {"has_human_gait": False},
    }


def test_passed_true_on_clean_data():
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap", bucket="bucket",
        detector_results=_good_results(), detector_errors=[],
        llm_assessment=None, llm_error=None,
        duration_seconds=5.0,
    )
    assert report["passed"] is True
    assert report["failure_reasons"] == []


def test_failed_on_face():
    results = _good_results()
    results["face"] = {"has_face": True, "face_count": 1}
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap", bucket="bucket",
        detector_results=results, detector_errors=[],
        llm_assessment=None, llm_error=None,
        duration_seconds=5.0,
    )
    assert report["passed"] is False
    assert "has_face" in report["failure_reasons"]


def test_null_score_causes_failure():
    results = _good_results()
    results["clarity"] = None
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap", bucket="bucket",
        detector_results=results, detector_errors=["clarity"],
        llm_assessment=None, llm_error=None,
        duration_seconds=5.0,
    )
    assert report["passed"] is False
    assert "analyzer_error:clarity" in report["failure_reasons"]


def test_report_id_is_valid_uuid4():
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap", bucket="bucket",
        detector_results=_good_results(), detector_errors=[],
        llm_assessment=None, llm_error=None, duration_seconds=5.0,
    )
    parsed = uuid.UUID(report["report_id"])
    assert parsed.version == 4


def test_report_id_is_unique():
    builder = ReportBuilder(_settings())
    kwargs = dict(
        source_file="test.mcap", bucket="b",
        detector_results=_good_results(), detector_errors=[],
        llm_assessment=None, llm_error=None, duration_seconds=5.0,
    )
    r1 = builder.build(**kwargs)
    r2 = builder.build(**kwargs)
    assert r1["report_id"] != r2["report_id"]


def test_analyzed_at_is_iso8601_utc():
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap", bucket="b",
        detector_results=_good_results(), detector_errors=[],
        llm_assessment=None, llm_error=None, duration_seconds=5.0,
    )
    dt = datetime.fromisoformat(report["analyzed_at"].replace("Z", "+00:00"))
    assert dt.tzinfo is not None


def test_llm_assessment_overrides_verdict():
    results = _good_results()
    results["face"] = {"has_face": True, "face_count": 1}
    llm = {"passed": True, "overrode_detector": True, "override_detail": "face is on screen", "narrative": "ok", "frames_reviewed": [], "imu_windows_reviewed": []}
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap", bucket="b",
        detector_results=results, detector_errors=[],
        llm_assessment=llm, llm_error=None,
        duration_seconds=5.0,
    )
    assert report["passed"] is True
    assert report["failure_reasons"] == []


def test_short_duration_fails():
    builder = ReportBuilder(_settings(minimum_duration_seconds=2.0))
    report = builder.build(
        source_file="test.mcap", bucket="b",
        detector_results=_good_results(), detector_errors=[],
        llm_assessment=None, llm_error=None,
        duration_seconds=0.5,
    )
    assert report["passed"] is False
    assert "duration_too_short" in report["failure_reasons"]
