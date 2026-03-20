# agent/runner.py
from __future__ import annotations
import os
from loguru import logger
from agent.config import settings
from agent.extractor import McapExtractor
from agent.pipeline import AnalysisPipeline
from agent.llm_judge import LLMJudge
from agent.report import ReportBuilder
from agent.analyzers.clarity import ClarityAnalyzer
from agent.analyzers.continuity import ContinuityAnalyzer
from agent.analyzers.face import FaceDetector
from agent.analyzers.voice import VoiceDetector
from agent.analyzers.gait import GaitDetector

# ── Singletons ─────────────────────────────────────────────────────────────

_extractor = McapExtractor(frame_sample_rate=settings.frame_sample_rate)

_model_path = os.path.join(settings.model_dir, "yunet.onnx")
if not os.path.exists(_model_path):
    _model_path = os.path.join(os.getcwd(), "models", "yunet.onnx")

_pipeline = AnalysisPipeline(analyzers=[
    ClarityAnalyzer(),
    ContinuityAnalyzer(),
    FaceDetector(model_path=_model_path),
    VoiceDetector(),
    GaitDetector(),
])
_judge = LLMJudge(
    api_key=settings.anthropic_api_key,
    model=settings.llm_model,
    clarity_threshold=settings.clarity_threshold,
    continuity_threshold=settings.continuity_threshold,
    margin=settings.llm_review_margin,
    base_url=settings.anthropic_base_url,
)
_builder = ReportBuilder(settings)


# ── Shared analysis function ────────────────────────────────────────────────

def analyze_local_file(local_path: str, source_file: str = "", bucket: str = "") -> dict:
    """Run the full pipeline on a local MCAP file. Returns a report dict (never raises)."""
    src = source_file or local_path
    try:
        data = _extractor.extract(local_path)
    except Exception as exc:
        logger.error("MCAP extraction failed for {!r}: {}", local_path, exc)
        return _builder.build(
            source_file=src, bucket=bucket,
            detector_results={}, detector_errors=["mcap_extraction"],
            llm_assessment=None, llm_error=None, duration_seconds=None,
        )
    detector_results, detector_errors = _pipeline.run(data)
    llm_assessment, llm_error = _judge.judge(detector_results, data)
    return _builder.build(
        source_file=src, bucket=bucket,
        detector_results=detector_results, detector_errors=detector_errors,
        llm_assessment=llm_assessment, llm_error=llm_error,
        duration_seconds=data["duration_seconds"],
    )
