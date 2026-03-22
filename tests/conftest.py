import pathlib
import numpy as np
import pytest
import cv2
from agent.analyzers.base import ExtractedData

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"

_CAM = "/camera/image_raw"
_AUDIO = "/audio/data"


def _make_sharp_frame(h: int = 64, w: int = 64) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[::4, :] = 255
    return frame


def _make_blurry_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _make_silent_pcm_frame() -> bytes:
    return b"\x00" * 960


def _make_speech_pcm_frame(freq: int = 1000) -> bytes:
    n = 480
    t = np.linspace(0, n / 16000, n, endpoint=False)
    samples = (16000 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return samples.tobytes()


@pytest.fixture
def sharp_frames() -> list[np.ndarray]:
    return [_make_sharp_frame() for _ in range(10)]


@pytest.fixture
def blurry_frames() -> list[np.ndarray]:
    return [_make_blurry_frame() for _ in range(10)]


@pytest.fixture
def empty_frames() -> list[np.ndarray]:
    return []


@pytest.fixture
def silent_audio() -> list[bytes]:
    return [_make_silent_pcm_frame() for _ in range(5)]


@pytest.fixture
def speech_audio() -> list[bytes]:
    return [_make_speech_pcm_frame() for _ in range(10)]


@pytest.fixture
def sharp_data(sharp_frames, silent_audio) -> ExtractedData:
    return ExtractedData(
        videos={_CAM: sharp_frames},
        audios={_AUDIO: silent_audio},
        sensor_series={},
        duration_seconds=5.0,
        extraction_warnings={},
    )


@pytest.fixture
def blurry_data(blurry_frames) -> ExtractedData:
    return ExtractedData(
        videos={_CAM: blurry_frames},
        audios={},
        sensor_series={},
        duration_seconds=5.0,
        extraction_warnings={},
    )


@pytest.fixture
def empty_data() -> ExtractedData:
    return ExtractedData(
        videos={_CAM: []},
        audios={},
        sensor_series={},
        duration_seconds=0.0,
        extraction_warnings={},
    )


@pytest.fixture
def face_data() -> ExtractedData:
    img = cv2.imread(str(FIXTURES_DIR / "lena.jpg"))
    assert img is not None, "tests/fixtures/lena.jpg missing"
    return ExtractedData(
        videos={_CAM: [img]},
        audios={},
        sensor_series={},
        duration_seconds=1.0,
        extraction_warnings={},
    )


@pytest.fixture
def person_data() -> ExtractedData:
    img = cv2.imread(str(FIXTURES_DIR / "person.jpg"))
    assert img is not None, "tests/fixtures/person.jpg missing"
    return ExtractedData(
        videos={_CAM: [img]},
        audios={},
        sensor_series={},
        duration_seconds=1.0,
        extraction_warnings={},
    )


@pytest.fixture
def speech_data(speech_audio) -> ExtractedData:
    return ExtractedData(
        videos={_CAM: [_make_sharp_frame() for _ in range(5)]},
        audios={_AUDIO: speech_audio},
        sensor_series={},
        duration_seconds=0.3,
        extraction_warnings={},
    )
