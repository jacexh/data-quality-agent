import pathlib
import numpy as np
import pytest
import cv2
from agent.analyzers.base import ExtractedData

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


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


def _make_speech_pcm_frame(freq: int = 1000) -> bytes:
    """30ms 1 kHz sine wave at 16 kHz mono int16 — detected as speech by WebRTC VAD."""
    n = 480  # 480 samples = 30 ms @ 16 kHz
    t = np.linspace(0, n / 16000, n, endpoint=False)
    samples = (16000 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return samples.tobytes()


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


@pytest.fixture
def face_data() -> ExtractedData:
    """Real face photo (lena.jpg) — YuNet should detect exactly one face."""
    img = cv2.imread(str(FIXTURES_DIR / "lena.jpg"))
    assert img is not None, "tests/fixtures/lena.jpg missing"
    return ExtractedData(frames=[img], audio_frames=None, sensor_series={}, duration_seconds=1.0)


@pytest.fixture
def person_data() -> ExtractedData:
    """Street photo with full-body person — HOG should detect at least one pedestrian."""
    img = cv2.imread(str(FIXTURES_DIR / "person.jpg"))
    assert img is not None, "tests/fixtures/person.jpg missing"
    return ExtractedData(frames=[img], audio_frames=None, sensor_series={}, duration_seconds=1.0)


@pytest.fixture
def speech_data() -> ExtractedData:
    """Frames with synthetic 1 kHz speech-band audio — WebRTC VAD should detect voice."""
    return ExtractedData(
        frames=[_make_sharp_frame() for _ in range(5)],
        audio_frames=[_make_speech_pcm_frame() for _ in range(10)],
        sensor_series={},
        duration_seconds=0.3,
    )
