import cv2
import numpy as np
from agent.analyzers.base import ExtractedData, FaceResult


class FaceDetector:
    def __init__(self, model_path: str = "models/yunet.onnx", conf_threshold: float = 0.6) -> None:
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._detector = None  # lazy init — avoid loading on import

    def _get_detector(self, width: int, height: int):
        detector = cv2.FaceDetectorYN.create(
            self._model_path,
            "",
            (width, height),
            score_threshold=self._conf_threshold,
            nms_threshold=0.3,
        )
        return detector

    def name(self) -> str:
        return "face"

    def analyze(self, data: ExtractedData) -> FaceResult:
        frames = data["frames"]
        if not frames:
            return FaceResult(has_face=False, face_count=0)

        max_faces = 0
        for frame in frames:
            h, w = frame.shape[:2]
            detector = self._get_detector(w, h)
            _, detections = detector.detect(frame)
            count = len(detections) if detections is not None else 0
            max_faces = max(max_faces, count)

        return FaceResult(has_face=max_faces > 0, face_count=max_faces)
