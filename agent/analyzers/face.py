import cv2
import numpy as np
from agent.analyzers.base import ExtractedData, FaceResult


class FaceDetector:
    """Face detector using OpenCV's YuNet ONNX model."""

    def __init__(self, model_path: str = "models/yunet.onnx", conf_threshold: float = 0.6) -> None:
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        # Cache by (width, height) to avoid reloading the ONNX model every frame.
        # Each FaceDetector instance is called from a single thread (ThreadPoolExecutor
        # dispatches one analyzer per worker), so this dict is not shared across threads.
        self._detector_cache: dict[tuple[int, int], cv2.FaceDetectorYN] = {}

    def _get_detector(self, width: int, height: int) -> cv2.FaceDetectorYN:
        key = (width, height)
        if key not in self._detector_cache:
            self._detector_cache[key] = cv2.FaceDetectorYN.create(
                self._model_path,
                "",
                (width, height),
                score_threshold=self._conf_threshold,
                nms_threshold=0.3,
            )
        return self._detector_cache[key]

    def name(self) -> str:
        return "face"

    def analyze(self, data: ExtractedData) -> FaceResult:
        frames = data["frames"]
        if not frames:
            return FaceResult(has_face=False, face_count=0, face_frame_ratio=0.0, max_confidence=0.0)

        max_faces = 0
        max_confidence = 0.0
        frames_with_face = 0

        for frame in frames:
            h, w = frame.shape[:2]
            detector = self._get_detector(w, h)
            _, detections = detector.detect(frame)
            if detections is not None and len(detections) > 0:
                count = len(detections)
                max_faces = max(max_faces, count)
                frames_with_face += 1
                # YuNet detection columns: [x,y,w,h, landmarks×10, confidence]
                frame_max_conf = float(detections[:, -1].max())
                max_confidence = max(max_confidence, frame_max_conf)

        face_frame_ratio = frames_with_face / len(frames)

        return FaceResult(
            has_face=max_faces > 0,
            face_count=max_faces,
            face_frame_ratio=round(face_frame_ratio, 4),
            max_confidence=round(max_confidence, 4),
        )
