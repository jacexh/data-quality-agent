import numpy as np
import pytest
from agent.analyzers.face import FaceDetector
from agent.analyzers.base import ExtractedData


def test_empty_frames_no_face(empty_data):
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(empty_data)
    assert result["has_face"] is False
    assert result["face_count"] == 0


def test_uniform_frame_no_face(blurry_data):
    """Uniform grey frame contains no face."""
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(blurry_data)
    assert result["has_face"] is False


def test_real_face_detected(face_data):
    """lena.jpg contains one face — YuNet must detect it (positive recall test)."""
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(face_data)
    assert result["has_face"] is True
    assert result["face_count"] >= 1


def test_name():
    assert FaceDetector(model_path="models/yunet.onnx").name() == "face"


def test_face_frame_ratio_zero_when_no_face(blurry_data):
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(blurry_data)
    assert result["face_frame_ratio"] == pytest.approx(0.0)


def test_face_frame_ratio_positive_when_face_present(face_data):
    """Single-frame face data → ratio should be 1.0."""
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(face_data)
    assert result["face_frame_ratio"] == pytest.approx(1.0)


def test_face_frame_ratio_partial(face_data, blurry_data):
    """Half face frames + half blank → ratio ~0.5."""
    import cv2
    detector = FaceDetector(model_path="models/yunet.onnx")
    face_frame = face_data["frames"][0]
    blank = blurry_data["frames"][0]
    mixed = ExtractedData(
        frames=[face_frame, blank, face_frame, blank],
        audio_frames=None, sensor_series={}, duration_seconds=1.0,
    )
    result = detector.analyze(mixed)
    assert 0.2 < result["face_frame_ratio"] < 0.8


def test_max_confidence_zero_when_no_face(empty_data):
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(empty_data)
    assert result["max_confidence"] == pytest.approx(0.0)


def test_max_confidence_high_for_real_face(face_data):
    """YuNet confidence for lena.jpg face should be above threshold (0.6)."""
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(face_data)
    assert result["max_confidence"] >= 0.6
