import numpy as np
from agent.analyzers.base import ExtractedData, CameraResult, AudioResult

def test_extracted_data_has_videos_and_audios():
    d = ExtractedData(
        videos={"/cam": []},
        audios={"/audio": []},
        sensor_series={},
        duration_seconds=1.0,
        extraction_warnings={},
    )
    assert "/cam" in d["videos"]
    assert "/audio" in d["audios"]
    assert d["extraction_warnings"] == {}


def test_camera_result_has_required_keys():
    r = CameraResult(
        topic="/cam",
        frame_count=10,
        clarity={"score": 0.9, "method": "laplacian+fft", "detail": {}},
        continuity={"score": 0.8, "method": "optical_flow", "detail": {}},
        face={"has_face": False, "face_count": 0, "face_frame_ratio": 0.0, "max_confidence": 0.0},
        gait={"has_human_gait": False, "person_frame_ratio": 0.0, "max_detection_weight": 0.0},
        llm_assessment=None,
        llm_skipped_reason="no_sensitive_detection",
        passed=True,
        failure_reasons=[],
        analyzer_errors=[],
    )
    assert r["topic"] == "/cam"
    assert r["passed"] is True


def test_audio_result_has_required_keys():
    r = AudioResult(
        topic="/audio",
        audio_frame_count=100,
        voice={"has_human_voice": False, "speech_frame_ratio": 0.0},
        llm_assessment=None,
        llm_skipped_reason="no_sensitive_detection",
        passed=True,
        failure_reasons=[],
        analyzer_errors=[],
    )
    assert r["topic"] == "/audio"
    assert r["passed"] is True
