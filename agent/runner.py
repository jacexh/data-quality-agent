# agent/runner.py
from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
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

def analyze_local_file(
    local_path: str,
    source_file: str = "",
    bucket: str = "",
    progress=None,
) -> dict:
    """Run the full pipeline on a local MCAP file. Returns a report dict (never raises).

    Args:
        progress: optional callable(msg: str) for progress reporting (e.g. print to stderr).
    """
    def _progress(msg: str) -> None:
        if progress is not None:
            progress(msg)

    src = source_file or local_path

    _progress(f"[1/4] Extracting MCAP: {local_path}")
    try:
        data = _extractor.extract(local_path)
    except Exception as exc:
        logger.error("MCAP extraction failed for {!r}: {}", local_path, exc, exc_info=True)
        _progress("      Extraction failed.")
        return _builder.build(
            source_file=src, bucket=bucket,
            duration_seconds=None,
            camera_results=[],
            audio_results=[],
            analyzer_errors=["mcap_extraction"],
        )

    n_cam = len(data["videos"])
    n_aud = len(data["audios"])
    dur = data.get("duration_seconds")
    dur_str = f"{dur:.1f}s" if dur is not None else "unknown duration"
    _progress(
        f"      Done: {n_cam} camera topic(s), {n_aud} audio topic(s), {dur_str}"
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

    # Detection phase: per-topic ThreadPoolExecutor (OpenCV releases GIL, no IPC needed)
    active_cam = sum(1 for t, f in data["videos"].items() if f and t not in camera_intermediates)
    active_aud = sum(1 for t, f in data["audios"].items() if f and t not in audio_intermediates)
    _progress(f"[2/4] Running detectors on {active_cam} camera + {active_aud} audio topic(s)...")
    with ThreadPoolExecutor(max_workers=settings.max_concurrent_topics) as executor:
        def _make_progress(prefix: str):
            def _p(msg: str) -> None:
                _progress(f"    {prefix}: {msg.strip()}")
            return _p

        cam_futures: dict = {}
        for topic, frames in data["videos"].items():
            if frames and topic not in camera_intermediates:
                _progress(f"    [camera] {topic} ({len(frames)} frames) → clarity, continuity, face, gait")
                cam_futures[executor.submit(
                    _run_visual_worker, topic, frames, _model_path,
                    settings.max_analysis_dim,
                    _make_progress(f"[camera] {topic}"),
                )] = topic

        aud_futures: dict = {}
        for topic, audio_frames in data["audios"].items():
            if audio_frames and topic not in audio_intermediates:
                _progress(f"    [audio]  {topic} ({len(audio_frames)} frames) → voice")
                aud_futures[executor.submit(
                    _run_audio_worker, topic, audio_frames,
                    _make_progress(f"[audio]  {topic}"),
                )] = topic

        for future in as_completed(cam_futures):
            topic = cam_futures[future]
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

        for future in as_completed(aud_futures):
            topic = aud_futures[future]
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

    n_llm = sum(1 for item in camera_intermediates.values() if item["needs_llm"]) + \
            sum(1 for item in audio_intermediates.values() if item["needs_llm"])
    if n_llm:
        _progress(f"[3/4] LLM review triggered for {n_llm} topic(s)...")
    else:
        _progress("[3/4] LLM review: not required (scores clear)")

    # LLM phase: concurrent ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=settings.llm_max_concurrent_calls) as executor:
        cam_llm_futures: dict = {}
        for topic, item in camera_intermediates.items():
            if item["needs_llm"]:
                _progress(f"    [LLM] {topic} → sending to Claude...")
                cam_llm_futures[executor.submit(
                    _judge.judge, topic, item["results"],
                    item["frames"], None, data["sensor_series"]
                )] = topic

        aud_llm_futures: dict = {}
        for topic, item in audio_intermediates.items():
            if item["needs_llm"]:
                _progress(f"    [LLM] {topic} → sending to Claude...")
                aud_llm_futures[executor.submit(
                    _judge.judge, topic, item["results"],
                    None, item["audio_frames"], data["sensor_series"]
                )] = topic

        for future in as_completed(cam_llm_futures):
            topic = cam_llm_futures[future]
            llm_result, llm_error = future.result()
            camera_intermediates[topic]["llm_result"] = llm_result
            camera_intermediates[topic]["llm_error"] = llm_error
            status = "✓ passed" if (llm_result and llm_result.get("passed")) else ("✗ failed" if llm_result else "✗ error")
            _progress(f"    [LLM] {topic} → {status}")

        for future in as_completed(aud_llm_futures):
            topic = aud_llm_futures[future]
            llm_result, llm_error = future.result()
            audio_intermediates[topic]["llm_result"] = llm_result
            audio_intermediates[topic]["llm_error"] = llm_error
            status = "✓ passed" if (llm_result and llm_result.get("passed")) else ("✗ failed" if llm_result else "✗ error")
            _progress(f"    [LLM] {topic} → {status}")

    # Assembly phase
    _progress("[4/4] Building report...")
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
