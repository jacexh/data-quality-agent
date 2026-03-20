from __future__ import annotations
import tempfile
import os
from loguru import logger
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request, Depends
from fastapi.responses import JSONResponse

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

import json

app = FastAPI()

_extractor = McapExtractor()

model_path = os.path.join(settings.model_dir, "yunet.onnx")
if not os.path.exists(model_path):
    model_path = os.path.join(os.getcwd(), "models", "yunet.onnx")

_pipeline = AnalysisPipeline(analyzers=[
    ClarityAnalyzer(),
    ContinuityAnalyzer(),
    FaceDetector(model_path=model_path),
    VoiceDetector(),
    GaitDetector(),
])
_judge = LLMJudge(
    api_key=settings.anthropic_api_key,
    model=settings.llm_model,
    clarity_threshold=settings.clarity_threshold,
    continuity_threshold=settings.continuity_threshold,
    margin=settings.llm_review_margin,
)
_builder = ReportBuilder(settings)


def _check_auth(request: Request) -> None:
    token = settings.webhook_auth_token
    if not token:
        return
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/notify")
async def notify(request: Request, background_tasks: BackgroundTasks):
    _check_auth(request)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "ignored", "reason": "invalid json"})

    # Extract object key from MinIO notification format
    records = body.get("Records", [])
    if not records:
        return JSONResponse({"status": "ignored", "reason": "no records"})

    key = records[0].get("s3", {}).get("object", {}).get("key", "")
    bucket = records[0].get("s3", {}).get("bucket", {}).get("name", settings.minio_bucket)

    if not key.endswith(".mcap"):
        logger.info(f"Skipping non-mcap file: {key}")
        return JSONResponse({"status": "ignored", "reason": "not_mcap"})

    background_tasks.add_task(_process_mcap, bucket=bucket, key=key)
    return JSONResponse({"status": "accepted"})


async def _process_mcap(bucket: str, key: str) -> None:
    import boto3
    from botocore.client import Config

    source_file = f"{bucket}/{key}"

    # Download
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url=f"http{'s' if settings.minio_use_ssl else ''}://{settings.minio_endpoint}",
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            config=Config(signature_version="s3v4"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, os.path.basename(key))
            s3.download_file(bucket, key, local_path)
            _analyze_and_log(source_file, bucket, local_path)
    except Exception as exc:
        report = _builder.build(
            source_file=source_file, bucket=bucket,
            detector_results={}, detector_errors=["minio_download"],
            llm_assessment=None, llm_error=None, duration_seconds=None,
        )
        logger.error(json.dumps(report))


def _analyze_and_log(source_file: str, bucket: str, local_path: str) -> None:
    # Extract
    try:
        data = _extractor.extract(local_path)
    except Exception as exc:
        report = _builder.build(
            source_file=source_file, bucket=bucket,
            detector_results={}, detector_errors=["mcap_extraction"],
            llm_assessment=None, llm_error=None, duration_seconds=None,
        )
        logger.error(json.dumps(report))
        return

    # Detect
    detector_results, detector_errors = _pipeline.run(data)

    # LLM Judge
    llm_assessment, llm_error = _judge.judge(detector_results, data)

    # Build and log report
    report = _builder.build(
        source_file=source_file, bucket=bucket,
        detector_results=detector_results, detector_errors=detector_errors,
        llm_assessment=llm_assessment, llm_error=llm_error,
        duration_seconds=data["duration_seconds"],
    )
    level = "WARNING" if not report["passed"] else "INFO"
    logger.log(level, json.dumps(report, ensure_ascii=False))
