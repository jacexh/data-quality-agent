import cv2
import numpy as np
from agent.analyzers.base import ExtractedData, ClarityResult, ClarityDetail

# Normalisation caps: values above these map to score=1.0.
# Using 95th-percentile estimates from natural scene datasets.
_LAP_CAP = 500.0
_FFT_CAP = 0.15  # high-freq energy ratio: sharp natural scenes typically 0.05–0.15


def _frame_scores(gray: np.ndarray) -> tuple[float, float]:
    """Return (laplacian_variance, fft_high_freq_ratio) for a single grayscale frame."""
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # FFT: ratio of energy in outer 50% of frequency spectrum vs total.
    # High-frequency attenuation is the primary signature of blur (including motion blur).
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift) ** 2
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    r_thresh = min(h, w) // 4  # inner 25% radius = low-freq band

    yy, xx = np.ogrid[:h, :w]
    mask_low = (yy - cy) ** 2 + (xx - cx) ** 2 <= r_thresh ** 2
    total = mag.sum()
    high_freq_ratio = float(mag[~mask_low].sum() / total) if total > 0 else 0.0

    return lap, high_freq_ratio


class ClarityAnalyzer:
    """Image clarity analyzer using Laplacian variance and FFT high-frequency ratio."""

    def name(self) -> str:
        return "clarity"

    def analyze(self, data: ExtractedData) -> ClarityResult:
        frames = data["frames"]
        if not frames:
            return ClarityResult(
                score=0.0,
                method="laplacian+fft",
                detail=ClarityDetail(
                    mean_laplacian_variance=0.0,
                    fft_high_freq_ratio=0.0,
                    frame_score_std=0.0,
                    frame_count=0,
                ),
            )

        lap_vars, fft_ratios, per_frame_scores = [], [], []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap, fft = _frame_scores(gray)
            lap_vars.append(lap)
            fft_ratios.append(fft)
            # Per-frame composite score (used to compute temporal std)
            s_lap = min(lap / _LAP_CAP, 1.0)
            s_fft = min(fft / _FFT_CAP, 1.0)
            per_frame_scores.append((s_lap + s_fft) / 2.0)

        mean_lap = float(np.mean(lap_vars))
        mean_fft = float(np.mean(fft_ratios))
        frame_std = float(np.std(per_frame_scores))

        score = (min(mean_lap / _LAP_CAP, 1.0) + min(mean_fft / _FFT_CAP, 1.0)) / 2.0

        return ClarityResult(
            score=round(score, 4),
            method="laplacian+fft",
            detail=ClarityDetail(
                mean_laplacian_variance=round(mean_lap, 4),
                fft_high_freq_ratio=round(mean_fft, 4),
                frame_score_std=round(frame_std, 4),
                frame_count=len(frames),
            ),
        )
