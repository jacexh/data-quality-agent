import cv2
from agent.analyzers.base import ExtractedData, GaitResult


class GaitDetector:
    def __init__(self) -> None:
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def name(self) -> str:
        return "gait"

    def analyze(self, data: ExtractedData) -> GaitResult:
        frames = data["frames"]
        if not frames:
            return GaitResult(has_human_gait=False)

        for frame in frames:
            # HOG requires minimum size of ~64×128; skip tiny frames
            h, w = frame.shape[:2]
            if h < 128 or w < 64:
                continue
            rects, _ = self._hog.detectMultiScale(
                frame, winStride=(8, 8), padding=(4, 4), scale=1.05
            )
            if len(rects) > 0:
                return GaitResult(has_human_gait=True)

        return GaitResult(has_human_gait=False)
