import webrtcvad
from loguru import logger
from agent.analyzers.base import ExtractedData, VoiceResult

_SAMPLE_RATE = 16000
_VAD_MODE = 2  # aggressiveness 0-3; 2 = balanced


class VoiceDetector:
    """Voice activity detector using WebRTC VAD."""

    def __init__(self, mode: int = _VAD_MODE) -> None:
        self._vad = webrtcvad.Vad(mode)

    def name(self) -> str:
        return "voice"

    def analyze(self, data: ExtractedData) -> VoiceResult:
        audio_frames = data["audio_frames"]
        if not audio_frames:
            return VoiceResult(has_human_voice=False, speech_frame_ratio=0.0)

        speech_count = 0
        for frame in audio_frames:
            try:
                if self._vad.is_speech(frame, _SAMPLE_RATE):
                    speech_count += 1
            except Exception as exc:
                logger.debug("VAD check failed on frame ({}B): {}", len(frame), exc)
                continue

        speech_frame_ratio = speech_count / len(audio_frames)
        return VoiceResult(
            has_human_voice=speech_count > 0,
            speech_frame_ratio=round(speech_frame_ratio, 4),
        )
