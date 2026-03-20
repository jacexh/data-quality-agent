from typing import Protocol, TypedDict
import numpy as np


class ExtractedData(TypedDict):
    frames: list[np.ndarray]              # BGR frames HxWxC uint8; may be []
    audio_frames: list[bytes] | None      # 30ms PCM chunks, 16kHz mono int16; None if absent
    sensor_series: dict[str, np.ndarray]  # topic → (T,D) float64; {} if absent
    duration_seconds: float


class ClarityDetail(TypedDict):
    mean_laplacian_variance: float
    fft_high_freq_ratio: float
    frame_score_std: float
    frame_count: int


class ContinuityDetail(TypedDict):
    mean_flow_magnitude: float
    flow_magnitude_std: float   # temporal jitter in motion speed
    flow_direction_std: float   # spatial incoherence of motion vectors (radians)
    discontinuity_frames: int
    frame_count: int


class ClarityResult(TypedDict):
    score: float       # [0.0, 1.0]
    method: str        # "laplacian+fft"
    detail: ClarityDetail


class ContinuityResult(TypedDict):
    score: float       # [0.0, 1.0]
    method: str        # "optical_flow"
    detail: ContinuityDetail


class FaceResult(TypedDict):
    has_face: bool
    face_count: int
    face_frame_ratio: float   # frames_with_face / total_frames
    max_confidence: float     # highest YuNet detection confidence score


class VoiceResult(TypedDict):
    has_human_voice: bool
    speech_frame_ratio: float  # speech_frames / total_audio_frames


class GaitResult(TypedDict):
    has_human_gait: bool
    person_frame_ratio: float      # frames_with_person / total_frames
    max_detection_weight: float    # highest HOG SVM weight (confidence proxy)


class Analyzer(Protocol):
    def name(self) -> str:
        """One of: "clarity" | "continuity" | "face" | "voice" | "gait" """
        ...

    def analyze(self, data: ExtractedData) -> ClarityResult | ContinuityResult | FaceResult | VoiceResult | GaitResult:
        """Must not raise. Handle empty frames gracefully."""
        ...
