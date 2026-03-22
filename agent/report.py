from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Any
from agent.analyzers.base import CameraResult, AudioResult
from agent.config import Settings


def evaluate_strategy(results: list[bool], strategy: str) -> bool:
    """Evaluate a list of per-topic pass/fail booleans using the configured strategy.

    Empty list always returns False — no silent pass with zero topics.
    """
    if not results:
        return False
    if strategy == "all":
        return all(results)
    elif strategy == "any":
        return any(results)
    elif strategy == "majority":
        return sum(results) > len(results) / 2
    raise ValueError(f"Unknown pass strategy: {strategy!r}")


class ReportBuilder:
    """Assembles camera/audio per-topic results into a structured JSON report."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def build(
        self,
        source_file: str,
        duration_seconds: float | None,
        camera_results: list[CameraResult],
        audio_results: list[AudioResult],
        analyzer_errors: list[str],
        bucket: str = "",
    ) -> dict[str, Any]:
        """Build the final report dict from per-topic results."""
        failure_reasons: list[str] = []

        # Duration check
        if duration_seconds is None or duration_seconds < self._s.minimum_duration_seconds:
            failure_reasons.append("duration_too_short")

        cameras_passed = evaluate_strategy(
            [r["passed"] for r in camera_results], self._s.camera_pass_strategy
        )
        audios_passed = evaluate_strategy(
            [r["passed"] for r in audio_results], self._s.audio_pass_strategy
        )
        overall_passed = cameras_passed and audios_passed and not failure_reasons

        return {
            "report_id": str(uuid.uuid4()),
            "source_file": source_file,
            "minio_bucket": bucket,
            "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_seconds": duration_seconds,
            "camera_pass_strategy": self._s.camera_pass_strategy,
            "audio_pass_strategy": self._s.audio_pass_strategy,
            "cameras": list(camera_results),
            "audios": list(audio_results),
            "overall_passed": overall_passed,
            "failure_reasons": sorted(set(failure_reasons)),
            "analyzer_errors": list(analyzer_errors),
        }
