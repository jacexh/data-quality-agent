import numpy as np
import pytest
from agent.analyzers.base import ExtractedData


def _make_sharp_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Checkerboard pattern — high Laplacian variance."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[::4, :] = 255
    return frame


def _make_blurry_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Uniform grey — near-zero Laplacian variance."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _make_silent_pcm_frame() -> bytes:
    """960 bytes of zero PCM = 30ms silence at 16kHz mono int16."""
    return b"\x00" * 960


@pytest.fixture
def sharp_data() -> ExtractedData:
    return ExtractedData(
        frames=[_make_sharp_frame() for _ in range(10)],
        audio_frames=[_make_silent_pcm_frame() for _ in range(5)],
        sensor_series={},
        duration_seconds=5.0,
    )


@pytest.fixture
def blurry_data() -> ExtractedData:
    return ExtractedData(
        frames=[_make_blurry_frame() for _ in range(10)],
        audio_frames=None,
        sensor_series={},
        duration_seconds=5.0,
    )


@pytest.fixture
def empty_data() -> ExtractedData:
    return ExtractedData(
        frames=[],
        audio_frames=None,
        sensor_series={},
        duration_seconds=0.0,
    )
