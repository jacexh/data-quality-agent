import cv2
import numpy as np
from agent.analyzers.base import ExtractedData, ClarityResult, ClarityDetail

# Normalisation caps: values above these map to score=1.0
_LAP_CAP = 500.0
_TEN_CAP = 3000.0


class ClarityAnalyzer:
    def name(self) -> str:
        return "clarity"

    def analyze(self, data: ExtractedData) -> ClarityResult:
        frames = data["frames"]
        if not frames:
            return ClarityResult(
                score=0.0,
                method="laplacian+tenengrad",
                detail=ClarityDetail(mean_laplacian_variance=0.0, mean_tenengrad=0.0, frame_count=0),
            )

        lap_vars, tenegrads = [], []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_vars.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            tenegrads.append(float((sx**2 + sy**2).mean()))

        mean_lap = float(np.mean(lap_vars))
        mean_ten = float(np.mean(tenegrads))

        score_lap = min(mean_lap / _LAP_CAP, 1.0)
        score_ten = min(mean_ten / _TEN_CAP, 1.0)
        score = (score_lap + score_ten) / 2.0

        return ClarityResult(
            score=round(score, 4),
            method="laplacian+tenengrad",
            detail=ClarityDetail(
                mean_laplacian_variance=round(mean_lap, 4),
                mean_tenengrad=round(mean_ten, 4),
                frame_count=len(frames),
            ),
        )
