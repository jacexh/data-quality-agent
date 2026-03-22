from typing import Protocol, TypedDict
import numpy as np


class ExtractedData(TypedDict):
    videos: dict[str, list[np.ndarray]]   # topic → BGR frame list
    audios: dict[str, list[bytes]]         # topic → 30ms PCM frame list
    sensor_series: dict[str, np.ndarray]   # shared across all topics
    duration_seconds: float
    extraction_warnings: dict[str, str]    # topic → "below_min_frames" etc.


class ClarityDetail(TypedDict):
    mean_laplacian_variance: float
    fft_high_freq_ratio: float
    frame_score_std: float
    frame_count: int


class ContinuityDetail(TypedDict):
    mean_flow_magnitude: float
    flow_magnitude_std: float
    flow_direction_std: float
    discontinuity_frames: int
    frame_count: int


class ClarityResult(TypedDict):
    score: float
    method: str
    detail: ClarityDetail


class ContinuityResult(TypedDict):
    score: float
    method: str
    detail: ContinuityDetail


class FaceResult(TypedDict):
    has_face: bool
    face_count: int
    face_frame_ratio: float
    max_confidence: float


class VoiceResult(TypedDict):
    has_human_voice: bool
    speech_frame_ratio: float


class GaitResult(TypedDict):
    has_human_gait: bool
    person_frame_ratio: float
    max_detection_weight: float


class CameraResult(TypedDict):
    topic: str
    frame_count: int
    clarity: ClarityResult
    continuity: ContinuityResult
    face: FaceResult
    gait: GaitResult
    llm_assessment: dict | None
    llm_skipped_reason: str | None
    passed: bool
    failure_reasons: list[str]
    analyzer_errors: list[str]


class AudioResult(TypedDict):
    topic: str
    audio_frame_count: int
    voice: VoiceResult
    llm_assessment: dict | None
    llm_skipped_reason: str | None
    passed: bool
    failure_reasons: list[str]
    analyzer_errors: list[str]


class VisualAnalyzer(Protocol):
    def name(self) -> str: ...
    def analyze(self, frames: list[np.ndarray]) -> ClarityResult | ContinuityResult | FaceResult | GaitResult:
        """Must not raise. Handle empty frames gracefully."""
        ...


class AudioAnalyzer(Protocol):
    def name(self) -> str: ...
    def analyze(self, audio_frames: list[bytes]) -> VoiceResult:
        """Must not raise. Handle empty audio_frames gracefully."""
        ...
