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


def test_name():
    assert VoiceDetector().name() == "voice"
