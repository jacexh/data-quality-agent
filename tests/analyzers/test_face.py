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


def test_name():
    assert FaceDetector(model_path="models/yunet.onnx").name() == "face"
