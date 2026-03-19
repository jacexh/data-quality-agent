import webrtcvad
from agent.analyzers.base import ExtractedData, VoiceResult

_SAMPLE_RATE = 16000
_VAD_MODE = 2  # aggressiveness 0-3; 2 = balanced


class VoiceDetector:
    def __init__(self, mode: int = _VAD_MODE) -> None:
        self._vad = webrtcvad.Vad(mode)

    def name(self) -> str:
        return "voice"

    def analyze(self, data: ExtractedData) -> VoiceResult:
        audio_frames = data["audio_frames"]
        if not audio_frames:
            return VoiceResult(has_human_voice=False)

        for frame in audio_frames:
            try:
                if self._vad.is_speech(frame, _SAMPLE_RATE):
                    return VoiceResult(has_human_voice=True)
            except Exception:
                continue

        return VoiceResult(has_human_voice=False)
