# agent/runner.py
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any
from loguru import logger
from agent.config import settings
from agent.extractor import McapExtractor
from agent.pipeline import _run_visual_worker, _run_audio_worker
from agent.llm_judge import LLMJudge, should_invoke_llm
from agent.report import ReportBuilder
from agent.analyzers.base import CameraResult, AudioResult

# ── Singletons ─────────────────────────────────────────────────────────────

_model_path = os.path.join(settings.model_dir, "yunet.onnx")
if not os.path.exists(_model_path):
    _fallback = os.path.join(os.getcwd(), "models", "yunet.onnx")
    logger.info("Model not found at {!r}, trying fallback {!r}", _model_path, _fallback)
    _model_path = _fallback

_extractor = McapExtractor(
    camera_topics=settings.camera_topics,
    audio_topics=settings.audio_topics,
    frame_sample_rate=settings.frame_sample_rate,
    max_frames_per_topic=settings.max_frames_per_topic,
)

_judge = LLMJudge(
    api_key=settings.anthropic_api_key,
    model=settings.llm_model,
    clarity_threshold=settings.clarity_threshold,
    continuity_threshold=settings.continuity_threshold,
    margin=settings.llm_review_margin,
    base_url=settings.anthropic_base_url,
)
_builder = ReportBuilder(settings)


# ── Per-topic result assembly helpers ───────────────────────────────────────

def _build_camera_result(topic: str, item: dict[str, Any]) -> CameraResult:
    results = item["results"]
    errors: list[str] = list(item["errors"])
    llm_result: dict | None = item.get("llm_result")
    llm_error: str | None = item.get("llm_error")

    if llm_error:
        errors.append(llm_error)

    failure_reasons: list[str] = [f"analyzer_error:{e}" for e in item["errors"]]

    clarity = results.get("clarity")
    continuity = results.get("continuity")
    face = results.get("face")
    gait = results.get("gait")

    if clarity is not None and clarity["score"] < settings.clarity_threshold:
        failure_reasons.append("clarity")
    if continuity is not None and continuity["score"] < settings.continuity_threshold:
        failure_reasons.append("continuity")
    if face is not None and face.get("has_face"):
        failure_reasons.append("has_face")
    if gait is not None and gait.get("has_human_gait"):
        failure_reasons.append("has_human_gait")

    # LLM overrides verdict when it ran
    if llm_result is not None:
        passed = llm_result["passed"]
        failure_reasons = [] if passed else failure_reasons
    else:
        passed = not failure_reasons

    llm_skipped_reason: str | None = None
    if llm_result is None and llm_error is None:
        if item["errors"]:
            llm_skipped_reason = "detector_error_no_llm_review"
        elif failure_reasons:
            llm_skipped_reason = "clear_failure_no_borderline_scores"
        else:
            llm_skipped_reason = "all_detectors_clear_no_borderline_scores"

    frames = item.get("frames", [])
    return CameraResult(
        topic=topic,
        frame_count=len(frames),
        clarity=clarity or {},
        continuity=continuity or {},
        face=face or {},
        gait=gait or {},
        llm_assessment=llm_result,
        llm_skipped_reason=llm_skipped_reason,
        passed=passed,
        failure_reasons=sorted(set(failure_reasons)),
        analyzer_errors=list(item["errors"]),
    )


def _build_audio_result(topic: str, item: dict[str, Any]) -> AudioResult:
    results = item["results"]
    errors: list[str] = list(item["errors"])
    llm_result: dict | None = item.get("llm_result")
    llm_error: str | None = item.get("llm_error")

    if llm_error:
        errors.append(llm_error)

    failure_reasons: list[str] = [f"analyzer_error:{e}" for e in item["errors"]]

    voice = results.get("voice")
    if voice is not None and voice.get("has_human_voice"):
        failure_reasons.append("has_human_voice")

    if llm_result is not None:
        passed = llm_result["passed"]
        failure_reasons = [] if passed else failure_reasons
    else:
        passed = not failure_reasons

    llm_skipped_reason: str | None = None
    if llm_result is None and llm_error is None:
        if item["errors"]:
            llm_skipped_reason = "detector_error_no_llm_review"
        elif failure_reasons:
            llm_skipped_reason = "clear_failure_no_borderline_scores"
        else:
            llm_skipped_reason = "all_detectors_clear_no_borderline_scores"

    audio_frames = item.get("audio_frames", [])
    return AudioResult(
        topic=topic,
        audio_frame_count=len(audio_frames),
        voice=voice or {},
        llm_assessment=llm_result,
        llm_skipped_reason=llm_skipped_reason,
        passed=passed,
        failure_reasons=sorted(set(failure_reasons)),
        analyzer_errors=list(item["errors"]),
    )


# ── Shared analysis function ────────────────────────────────────────────────

def analyze_local_file(local_path: str, source_file: str = "", bucket: str = "") -> dict:
    """Run the full pipeline on a local MCAP file. Returns a report dict (never raises)."""
    src = source_file or local_path

    try:
        data = _extractor.extract(local_path)
    except Exception as exc:
        logger.error("MCAP extraction failed for {!r}: {}", local_path, exc, exc_info=True)
        return _builder.build(
            source_file=src, bucket=bucket,
            duration_seconds=None,
            camera_results=[],
            audio_results=[],
            analyzer_errors=["mcap_extraction"],
        )

    camera_intermediates: dict[str, dict] = {}
    audio_intermediates: dict[str, dict] = {}

    # Pre-populate empty / below-min-frames topics (no worker needed)
    for topic, frames in data["videos"].items():
        warning = data["extraction_warnings"].get(topic)
        error = warning or ("zero_frames" if not frames else None)
        if error:
            camera_intermediates[topic] = {
                "frames": [], "results": {}, "errors": [error], "needs_llm": False,
                "llm_result": None, "llm_error": None,
            }

    for topic, audio_frames in data["audios"].items():
        warning = data["extraction_warnings"].get(topic)
        error = warning or ("zero_frames" if not audio_frames else None)
        if error:
            audio_intermediates[topic] = {
                "audio_frames": [], "results": {}, "errors": [error], "needs_llm": False,
                "llm_result": None, "llm_error": None,
            }

    # Detection phase: per-topic ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=settings.max_concurrent_topics) as executor:
        cam_futures = {
            executor.submit(_run_visual_worker, topic, frames, _model_path): topic
            for topic, frames in data["videos"].items()
            if frames and topic not in camera_intermediates
        }
        aud_futures = {
            executor.submit(_run_audio_worker, topic, audio_frames): topic
            for topic, audio_frames in data["audios"].items()
            if audio_frames and topic not in audio_intermediates
        }

        for future, topic in cam_futures.items():
            try:
                _, results, errors = future.result()
            except Exception as exc:
                logger.error("Visual worker failed for {!r}: {}", topic, exc, exc_info=True)
                results, errors = {}, ["worker_crash"]
            camera_intermediates[topic] = {
                "frames": data["videos"][topic],
                "results": results,
                "errors": errors,
                "needs_llm": should_invoke_llm(
                    results, settings.clarity_threshold,
                    settings.continuity_threshold, settings.llm_review_margin,
                ),
                "llm_result": None,
                "llm_error": None,
            }

        for future, topic in aud_futures.items():
            try:
                _, results, errors = future.result()
            except Exception as exc:
                logger.error("Audio worker failed for {!r}: {}", topic, exc, exc_info=True)
                results, errors = {}, ["worker_crash"]
            voice = results.get("voice", {})
            audio_intermediates[topic] = {
                "audio_frames": data["audios"][topic],
                "results": results,
                "errors": errors,
                "needs_llm": bool(voice.get("has_human_voice")),
                "llm_result": None,
                "llm_error": None,
            }

    # LLM phase: concurrent ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=settings.llm_max_concurrent_calls) as executor:
        cam_llm_futures = {
            executor.submit(
                _judge.judge, topic, item["results"],
                item["frames"], None, data["sensor_series"]
            ): topic
            for topic, item in camera_intermediates.items() if item["needs_llm"]
        }
        aud_llm_futures = {
            executor.submit(
                _judge.judge, topic, item["results"],
                None, item["audio_frames"], data["sensor_series"]
            ): topic
            for topic, item in audio_intermediates.items() if item["needs_llm"]
        }

        for future, topic in cam_llm_futures.items():
            llm_result, llm_error = future.result()
            camera_intermediates[topic]["llm_result"] = llm_result
            camera_intermediates[topic]["llm_error"] = llm_error

        for future, topic in aud_llm_futures.items():
            llm_result, llm_error = future.result()
            audio_intermediates[topic]["llm_result"] = llm_result
            audio_intermediates[topic]["llm_error"] = llm_error

    # Assembly phase
    camera_results = [_build_camera_result(t, item) for t, item in camera_intermediates.items()]
    audio_results  = [_build_audio_result(t, item)  for t, item in audio_intermediates.items()]

    return _builder.build(
        source_file=src,
        bucket=bucket,
        duration_seconds=data["duration_seconds"],
        camera_results=camera_results,
        audio_results=audio_results,
        analyzer_errors=[],
    )
