import pytest
import numpy as np
from agent.pipeline import AnalysisPipeline


class _OkVisualAnalyzer:
    def name(self) -> str:
        return "ok_visual"

    def analyze(self, frames: list[np.ndarray]):
        return {"value": 42}


class _BrokenVisualAnalyzer:
    def name(self) -> str:
        return "broken"

    def analyze(self, frames: list[np.ndarray]):
        raise RuntimeError("oops")


class _OkAudioAnalyzer:
    def name(self) -> str:
        return "ok_audio"

    def analyze(self, audio_frames: list[bytes]):
        return {"value": 99}


def test_run_visual_returns_results():
    pipeline = AnalysisPipeline(visual_analyzers=[_OkVisualAnalyzer()], audio_analyzers=[])
    frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    results, errors = pipeline.run_visual(frames)
    assert results["ok_visual"] == {"value": 42}
    assert errors == []


def test_run_visual_broken_analyzer_does_not_abort_others():
    pipeline = AnalysisPipeline(
        visual_analyzers=[_OkVisualAnalyzer(), _BrokenVisualAnalyzer()],
        audio_analyzers=[],
    )
    results, errors = pipeline.run_visual([np.zeros((64, 64, 3), dtype=np.uint8)])
    assert results["ok_visual"] == {"value": 42}
    assert results["broken"] is None
    assert "broken" in errors


def test_run_audio_returns_results():
    pipeline = AnalysisPipeline(visual_analyzers=[], audio_analyzers=[_OkAudioAnalyzer()])
    results, errors = pipeline.run_audio([b"\x00" * 960])
    assert results["ok_audio"] == {"value": 99}
    assert errors == []


def test_run_audio_empty_input():
    pipeline = AnalysisPipeline(visual_analyzers=[], audio_analyzers=[_OkAudioAnalyzer()])
    results, errors = pipeline.run_audio([])
    assert results["ok_audio"] == {"value": 99}  # analyzer handles empty gracefully
    assert errors == []
