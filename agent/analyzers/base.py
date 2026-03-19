from typing import Protocol, TypedDict
import numpy as np


class ExtractedData(TypedDict):
    frames: list[np.ndarray]              # BGR frames HxWxC uint8; may be []
    audio_frames: list[bytes] | None      # 30ms PCM chunks, 16kHz mono int16; None if absent
    sensor_series: dict[str, np.ndarray]  # topic → (T,D) float64; {} if absent
    duration_seconds: float


class ClarityDetail(TypedDict):
    mean_laplacian_variance: float
    mean_tenengrad: float
    frame_count: int


class ContinuityDetail(TypedDict):
    mean_flow_magnitude: float
    discontinuity_frames: int
    frame_count: int


class ClarityResult(TypedDict):
    score: float       # [0.0, 1.0]
    method: str        # "laplacian+tenengrad"
    detail: ClarityDetail


class ContinuityResult(TypedDict):
    score: float       # [0.0, 1.0]
    method: str        # "optical_flow"
    detail: ContinuityDetail


class FaceResult(TypedDict):
    has_face: bool
    face_count: int


class VoiceResult(TypedDict):
    has_human_voice: bool


class GaitResult(TypedDict):
    has_human_gait: bool


class Analyzer(Protocol):
    def name(self) -> str:
        """One of: "clarity" | "continuity" | "face" | "voice" | "gait" """
        ...

    def analyze(self, data: ExtractedData) -> ClarityResult | ContinuityResult | FaceResult | VoiceResult | GaitResult:
        """Must not raise. Handle empty frames gracefully."""
        ...
