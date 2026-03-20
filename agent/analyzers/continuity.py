import cv2
import numpy as np
from agent.analyzers.base import ExtractedData, ContinuityResult, ContinuityDetail

_DISCONTINUITY_THRESHOLD = 2.0  # pixels/frame — above this counts as a jump


class ContinuityAnalyzer:
    def name(self) -> str:
        return "continuity"

    def analyze(self, data: ExtractedData) -> ContinuityResult:
        frames = data["frames"]
        if not frames:
            return ContinuityResult(
                score=0.0,
                method="optical_flow",
                detail=ContinuityDetail(mean_flow_magnitude=0.0, discontinuity_frames=0, frame_count=0),
            )
        if len(frames) == 1:
            return ContinuityResult(
                score=1.0,
                method="optical_flow",
                detail=ContinuityDetail(mean_flow_magnitude=0.0, discontinuity_frames=0, frame_count=1),
            )

        magnitudes, discontinuities = [], 0
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
            magnitudes.append(mag)
            if mag > _DISCONTINUITY_THRESHOLD:
                discontinuities += 1
            prev_gray = curr_gray

        mean_mag = float(np.mean(magnitudes))
        score = 1.0 - discontinuities / len(magnitudes)

        return ContinuityResult(
            score=round(score, 4),
            method="optical_flow",
            detail=ContinuityDetail(
                mean_flow_magnitude=round(mean_mag, 4),
                discontinuity_frames=discontinuities,
                frame_count=len(frames),
            ),
        )
