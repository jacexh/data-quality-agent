import numpy as np
import pytest
import cv2
from agent.analyzers.face import FaceDetector

_CAM = "/camera/image_raw"


def test_empty_frames_no_face():
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze([])
    assert result["has_face"] is False
    assert result["face_count"] == 0


def test_uniform_frame_no_face(blurry_frames):
    """Uniform grey frame contains no face."""
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(blurry_frames)
    assert result["has_face"] is False


def test_real_face_detected(face_data):
    """lena.jpg contains one face — YuNet must detect it (positive recall test)."""
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(face_data["videos"][_CAM])
    assert result["has_face"] is True
    assert result["face_count"] >= 1


def test_name():
    assert FaceDetector(model_path="models/yunet.onnx").name() == "face"


def test_face_frame_ratio_zero_when_no_face(blurry_frames):
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(blurry_frames)
    assert result["face_frame_ratio"] == pytest.approx(0.0)


def test_face_frame_ratio_positive_when_face_present(face_data):
    """Single-frame face data → ratio should be 1.0."""
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(face_data["videos"][_CAM])
    assert result["face_frame_ratio"] == pytest.approx(1.0)


def test_face_frame_ratio_partial(face_data, blurry_frames):
    """Half face frames + half blank → ratio ~0.5."""
    detector = FaceDetector(model_path="models/yunet.onnx")
    face_frame = face_data["videos"][_CAM][0]
    blank = blurry_frames[0]
    mixed = [face_frame, blank, face_frame, blank]
    result = detector.analyze(mixed)
    assert 0.2 < result["face_frame_ratio"] < 0.8


def test_max_confidence_zero_when_no_face():
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze([])
    assert result["max_confidence"] == pytest.approx(0.0)


def test_max_confidence_high_for_real_face(face_data):
    """YuNet confidence for lena.jpg face should be above threshold (0.6)."""
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(face_data["videos"][_CAM])
    assert result["max_confidence"] >= 0.6
