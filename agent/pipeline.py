from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
import cv2
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

    def run_visual(
        self, frames: list[np.ndarray], progress=None
    ) -> tuple[dict[str, Any], list[str]]:
        """Run all visual analyzers concurrently. Returns (results_dict, error_names)."""
        return self._run([(a, frames) for a in self._visual], progress=progress)

    def run_audio(
        self, audio_frames: list[bytes], progress=None
    ) -> tuple[dict[str, Any], list[str]]:
        """Run all audio analyzers concurrently. Returns (results_dict, error_names)."""
        return self._run([(a, audio_frames) for a in self._audio], progress=progress)

    def _run(
        self, analyzer_inputs: list[tuple], progress=None
    ) -> tuple[dict[str, Any], list[str]]:
        results: dict[str, Any] = {}
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(a.analyze, inp): a for a, inp in analyzer_inputs}
            for future in as_completed(futures):
                analyzer = futures[future]
                name = analyzer.name()
                try:
                    results[name] = future.result()
                    if progress is not None:
                        progress(f"        ✓ {name}")
                except Exception as exc:
                    logger.error("Analyzer {!r} failed: {}", name, exc, exc_info=True)
                    results[name] = None
                    errors.append(name)
                    if progress is not None:
                        progress(f"        ✗ {name} (error)")
        return results, errors


# ── Picklable top-level workers for ProcessPoolExecutor dispatch ────────────

def _resize_frames(frames: list[np.ndarray], max_dim: int) -> list[np.ndarray]:
    """Resize frames so max(h, w) <= max_dim. Returns new list; does not mutate input."""
    if max_dim <= 0:
        return frames
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        if max(h, w) <= max_dim:
            resized.append(f)
        else:
            scale = max_dim / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            resized.append(cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA))
    return resized


def _run_visual_worker(
    topic: str,
    frames: list[np.ndarray],
    model_path: str,
    max_analysis_dim: int = 640,
    progress=None,
) -> tuple[str, dict[str, Any], list[str]]:
    """Per-topic visual detection worker."""
    from agent.analyzers.clarity import ClarityAnalyzer
    from agent.analyzers.continuity import ContinuityAnalyzer
    from agent.analyzers.face import FaceDetector
    from agent.analyzers.gait import GaitDetector

    analysis_frames = _resize_frames(frames, max_analysis_dim)

    pipeline = AnalysisPipeline(
        visual_analyzers=[
            ClarityAnalyzer(),
            ContinuityAnalyzer(),
            FaceDetector(model_path=model_path),
            GaitDetector(),
        ],
        audio_analyzers=[],
    )
    results, errors = pipeline.run_visual(analysis_frames, progress=progress)
    return topic, results, errors


def _run_audio_worker(
    topic: str,
    audio_frames: list[bytes],
    progress=None,
) -> tuple[str, dict[str, Any], list[str]]:
    """Per-topic audio detection worker."""
    from agent.analyzers.voice import VoiceDetector

    pipeline = AnalysisPipeline(visual_analyzers=[], audio_analyzers=[VoiceDetector()])
    results, errors = pipeline.run_audio(audio_frames, progress=progress)
    return topic, results, errors
