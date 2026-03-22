import pytest
import numpy as np
from agent.analyzers.voice import VoiceDetector


def _make_speech_frame(freq: int = 1000) -> bytes:
    n = 480
    t = np.linspace(0, n / 16000, n, endpoint=False)
    samples = (16000 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return samples.tobytes()


def _make_silent_frame() -> bytes:
    return b"\x00" * 960


def test_silent_pcm_no_voice(silent_audio):
    """All-zero PCM frames are silence → no voice."""
    detector = VoiceDetector()
    result = detector.analyze(silent_audio)
    assert result["has_human_voice"] is False


def test_none_audio_no_voice():
    detector = VoiceDetector()
    result = detector.analyze([])
    assert result["has_human_voice"] is False


def test_speech_band_audio_detected(speech_audio):
    """1 kHz sine wave (speech frequency band) must be detected as voice (positive recall test)."""
    detector = VoiceDetector()
    result = detector.analyze(speech_audio)
    assert result["has_human_voice"] is True


def test_empty_audio_frames_list_no_voice():
    """Empty list [] (distinct from None) — falsy, must also return no voice."""
    detector = VoiceDetector()
    result = detector.analyze([])
    assert result["has_human_voice"] is False


def test_name():
    assert VoiceDetector().name() == "voice"


def test_speech_frame_ratio_zero_for_silence(silent_audio):
    """All silence → ratio = 0.0."""
    result = VoiceDetector().analyze(silent_audio)
    assert result["speech_frame_ratio"] == pytest.approx(0.0)


def test_speech_frame_ratio_zero_for_no_audio():
    result = VoiceDetector().analyze([])
    assert result["speech_frame_ratio"] == pytest.approx(0.0)


def test_speech_frame_ratio_one_for_all_speech(speech_audio):
    """All speech frames → ratio = 1.0."""
    result = VoiceDetector().analyze(speech_audio)
    assert result["speech_frame_ratio"] == pytest.approx(1.0)


def test_speech_frame_ratio_partial():
    """Mix of speech and silence → ratio is strictly between 0 and 1.

    VAD mode 2 may classify some silence frames as speech (false positives),
    so we only assert the ratio is > 0 and < 1, not an exact midpoint.
    The essential property being tested: we count all frames, not early-exit.
    """
    frames = [_make_speech_frame() for _ in range(3)] + \
             [_make_silent_frame() for _ in range(7)]
    result = VoiceDetector().analyze(frames)
    assert 0.0 < result["speech_frame_ratio"] < 1.0
