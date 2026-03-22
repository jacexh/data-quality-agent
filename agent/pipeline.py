from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import numpy as np
from loguru import logger
from agent.analyzers.base import VisualAnalyzer, AudioAnalyzer


class AnalysisPipeline:
    """Runs visual and audio analyzers concurrently via a thread pool."""

    def __init__(
        self,
        visual_analyzers: list[VisualAnalyzer],
        audio_analyzers: list[AudioAnalyzer],
        max_workers: int = 5,
    ) -> None:
        self._visual = visual_analyzers
        self._audio = audio_analyzers
        self._max_workers = max_workers

    def run_visual(self, frames: list[np.ndarray]) -> tuple[dict[str, Any], list[str]]:
        """Run all visual analyzers concurrently. Returns (results_dict, error_names)."""
        return self._run([(a, frames) for a in self._visual])

    def run_audio(self, audio_frames: list[bytes]) -> tuple[dict[str, Any], list[str]]:
        """Run all audio analyzers concurrently. Returns (results_dict, error_names)."""
        return self._run([(a, audio_frames) for a in self._audio])

    def _run(self, analyzer_inputs: list[tuple]) -> tuple[dict[str, Any], list[str]]:
        results: dict[str, Any] = {}
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(a.analyze, inp): a for a, inp in analyzer_inputs}
            for future, analyzer in futures.items():
                name = analyzer.name()
                try:
                    results[name] = future.result()
                except Exception as exc:
                    logger.error("Analyzer {!r} failed: {}", name, exc, exc_info=True)
                    results[name] = None
                    errors.append(name)
        return results, errors


# ── Picklable top-level workers for ProcessPoolExecutor dispatch ────────────

def _run_visual_worker(
    topic: str,
    frames: list[np.ndarray],
    model_path: str,
) -> tuple[str, dict[str, Any], list[str]]:
    """Per-topic visual detection worker. Runs in a separate process.

    Instantiates a fresh AnalysisPipeline with all visual analyzers.
    model_path is passed explicitly because the worker process has no access
    to the parent's singleton state.
    """
    from agent.analyzers.clarity import ClarityAnalyzer
    from agent.analyzers.continuity import ContinuityAnalyzer
    from agent.analyzers.face import FaceDetector
    from agent.analyzers.gait import GaitDetector

    pipeline = AnalysisPipeline(
        visual_analyzers=[
            ClarityAnalyzer(),
            ContinuityAnalyzer(),
            FaceDetector(model_path=model_path),
            GaitDetector(),
        ],
        audio_analyzers=[],
    )
    results, errors = pipeline.run_visual(frames)
    return topic, results, errors


def _run_audio_worker(
    topic: str,
    audio_frames: list[bytes],
) -> tuple[str, dict[str, Any], list[str]]:
    """Per-topic audio detection worker. Runs in a separate process."""
    from agent.analyzers.voice import VoiceDetector

    pipeline = AnalysisPipeline(visual_analyzers=[], audio_analyzers=[VoiceDetector()])
    results, errors = pipeline.run_audio(audio_frames)
    return topic, results, errors
