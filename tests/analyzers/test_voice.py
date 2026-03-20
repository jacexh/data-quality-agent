from agent.analyzers.voice import VoiceDetector
from agent.analyzers.base import ExtractedData


def test_silent_pcm_no_voice(sharp_data):
    """All-zero PCM frames are silence → no voice."""
    detector = VoiceDetector()
    result = detector.analyze(sharp_data)
    assert result["has_human_voice"] is False


def test_none_audio_no_voice(empty_data):
    detector = VoiceDetector()
    result = detector.analyze(empty_data)
    assert result["has_human_voice"] is False


def test_speech_band_audio_detected(speech_data):
    """1 kHz sine wave (speech frequency band) must be detected as voice (positive recall test)."""
    detector = VoiceDetector()
    result = detector.analyze(speech_data)
    assert result["has_human_voice"] is True


def test_name():
    assert VoiceDetector().name() == "voice"
