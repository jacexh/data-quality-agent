from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Any
from agent.config import Settings


class ReportBuilder:
    """Merges detector and LLM results into a structured JSON report."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def build(
        self,
        source_file: str,
        bucket: str,
        detector_results: dict[str, Any],
        detector_errors: list[str],
        llm_assessment: dict[str, Any] | None,
        llm_error: str | None,
        duration_seconds: float | None,
    ) -> dict[str, Any]:
        """Build the final report dict from pipeline outputs."""
        analyzer_errors = list(detector_errors)
        if llm_error:
            analyzer_errors.append(llm_error)

        failure_reasons: list[str] = []

        # Duration check
        if duration_seconds is None or duration_seconds < self._s.minimum_duration_seconds:
            failure_reasons.append("duration_too_short")

        # Build scores section
        clarity = detector_results.get("clarity")
        continuity = detector_results.get("continuity")
        scores = None
        if clarity is not None or continuity is not None:
            scores = {}
            if clarity is not None:
                scores["clarity"] = clarity
            if continuity is not None:
                scores["continuity"] = continuity

        # Build sensitive_info section
        face = detector_results.get("face")
        voice = detector_results.get("voice")
        gait = detector_results.get("gait")
        sensitive_info = None
        if any(x is not None for x in [face, voice, gait]):
            sensitive_info = {
                "has_face": face["has_face"] if face else None,
                "face_count": face["face_count"] if face else None,
                "has_human_voice": voice["has_human_voice"] if voice else None,
                "has_human_gait": gait["has_human_gait"] if gait else None,
            }

        # Collect detector-based failure reasons
        for name in detector_errors:
            failure_reasons.append(f"analyzer_error:{name}")

        if clarity is None and "clarity" not in detector_errors:
            pass  # not run
        elif clarity is not None:
            if clarity["score"] < self._s.clarity_threshold:
                failure_reasons.append("clarity")

        if continuity is not None:
            if continuity["score"] < self._s.continuity_threshold:
                failure_reasons.append("continuity")

        if face is not None and face["has_face"]:
            failure_reasons.append("has_face")
        elif face is None and "face" not in detector_errors:
            pass
        elif face is None:
            pass  # already in analyzer_errors

        if voice is not None and voice["has_human_voice"]:
            failure_reasons.append("has_human_voice")

        if gait is not None and gait["has_human_gait"]:
            failure_reasons.append("has_human_gait")

        # LLM overrides verdict if it ran successfully
        if llm_assessment is not None:
            passed = llm_assessment["passed"]
            failure_reasons = [] if passed else failure_reasons
        else:
            passed = len(failure_reasons) == 0 and not analyzer_errors

        llm_skipped_reason = None
        if llm_assessment is None and llm_error is None:
            if detector_errors:
                llm_skipped_reason = "detector_error_no_llm_review"
            elif failure_reasons:
                llm_skipped_reason = "clear_failure_no_borderline_scores"
            else:
                llm_skipped_reason = "all_detectors_clear_no_borderline_scores"

        return {
            "report_id": str(uuid.uuid4()),
            "source_file": source_file,
            "minio_bucket": bucket,
            "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_seconds": duration_seconds,
            "scores": scores,
            "sensitive_info": sensitive_info,
            "llm_assessment": llm_assessment,
            "llm_skipped_reason": llm_skipped_reason,
            "analyzer_errors": analyzer_errors,
            "passed": passed,
            "failure_reasons": sorted(set(failure_reasons)),
        }
