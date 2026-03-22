import cv2
import numpy as np
from agent.analyzers.base import ContinuityResult, ContinuityDetail

_DISCONTINUITY_THRESHOLD = 2.0  # pixels/frame — above this counts as a jump


def _flow_stats(flow: np.ndarray) -> tuple[float, float]:
    """Return (mean_magnitude, circular_direction_std) for a flow field.

    Uses circular statistics so that a uniform horizontal pan (all angles ≈ π
    or ≈ -π) correctly reads as coherent rather than maximally dispersed.
    circular_std ≈ 0  → all vectors aligned (smooth pan)
    circular_std large → random directions (chaotic / discontinuity)
    """
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    mean_mag = float(mag.mean())
    angles = np.arctan2(flow[..., 1], flow[..., 0])
    # Resultant length R ∈ [0, 1]: 1 = perfectly aligned, 0 = uniform dispersion
    R = float(np.abs(np.mean(np.exp(1j * angles))))
    # Circular std: 0 when coherent, grows large as dispersion increases
    circular_std = float(np.sqrt(-2.0 * np.log(max(R, 1e-6))))
    return mean_mag, circular_std


class ContinuityAnalyzer:
    """Temporal continuity analyzer using Farneback optical flow."""

    def name(self) -> str:
        return "continuity"

    def analyze(self, frames: list) -> ContinuityResult:
        if not frames:
            return ContinuityResult(
                score=0.0,
                method="optical_flow",
                detail=ContinuityDetail(
                    mean_flow_magnitude=0.0,
                    flow_magnitude_std=0.0,
                    flow_direction_std=0.0,
                    discontinuity_frames=0,
                    frame_count=0,
                ),
            )
        if len(frames) == 1:
            return ContinuityResult(
                score=1.0,
                method="optical_flow",
                detail=ContinuityDetail(
                    mean_flow_magnitude=0.0,
                    flow_magnitude_std=0.0,
                    flow_direction_std=0.0,
                    discontinuity_frames=0,
                    frame_count=1,
                ),
            )

        magnitudes, dir_stds, discontinuities = [], [], 0
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mean_mag, dir_std = _flow_stats(flow)
            magnitudes.append(mean_mag)
            dir_stds.append(dir_std)
            # Penalise incoherent high-magnitude motion (true discontinuity),
            # but not coherent high-magnitude motion (smooth pan).
            if mean_mag > _DISCONTINUITY_THRESHOLD and dir_std > 0.5:
                discontinuities += 1
            prev_gray = curr_gray

        mean_mag = float(np.mean(magnitudes))
        mag_std = float(np.std(magnitudes))
        mean_dir_std = float(np.mean(dir_stds))
        score = 1.0 - discontinuities / len(magnitudes)

        return ContinuityResult(
            score=round(score, 4),
            method="optical_flow",
            detail=ContinuityDetail(
                mean_flow_magnitude=round(mean_mag, 4),
                flow_magnitude_std=round(mag_std, 4),
                flow_direction_std=round(mean_dir_std, 4),
                discontinuity_frames=discontinuities,
                frame_count=len(frames),
            ),
        )
