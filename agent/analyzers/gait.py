import cv2
import numpy as np
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
            return GaitResult(has_human_gait=False, person_frame_ratio=0.0, max_detection_weight=0.0)

        frames_with_person = 0
        max_weight = 0.0
        eligible = 0

        for frame in frames:
            h, w = frame.shape[:2]
            if h < 128 or w < 64:
                continue
            eligible += 1
            rects, weights = self._hog.detectMultiScale(
                frame, winStride=(8, 8), padding=(4, 4), scale=1.05
            )
            if len(rects) > 0:
                frames_with_person += 1
                max_weight = max(max_weight, float(weights.max()))

        if eligible == 0:
            return GaitResult(has_human_gait=False, person_frame_ratio=0.0, max_detection_weight=0.0)

        person_frame_ratio = frames_with_person / eligible
        return GaitResult(
            has_human_gait=frames_with_person > 0,
            person_frame_ratio=round(person_frame_ratio, 4),
            max_detection_weight=round(max_weight, 4),
        )
