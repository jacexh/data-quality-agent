from __future__ import annotations
import asyncio
import json
import os
import re
import tempfile
from contextlib import asynccontextmanager
from typing import Any

import boto3
from botocore.client import Config
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from loguru import logger

from agent.config import settings
from agent.runner import _builder, analyze_local_file

_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=settings.max_queue_size)
_processing: set[str] = set()


# ── S3 client factory ──────────────────────────────────────────────────────

def _make_s3_client() -> Any:
    """Create a boto3 S3 client. Call once per worker — each worker owns its client."""
    return boto3.client(
        "s3",
        endpoint_url=f"http{'s' if settings.minio_use_ssl else ''}://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(
            signature_version="s3v4",
            max_pool_connections=4,
            connect_timeout=10,
            read_timeout=300,
            retries={"max_attempts": 3, "mode": "adaptive"},
        ),
    )


# ── Worker ─────────────────────────────────────────────────────────────────

async def _worker(s3_client: Any) -> None:
    """Consume jobs from _queue. Each worker owns its S3 client exclusively."""
    while True:
        bucket, key = await _queue.get()
        try:
            await asyncio.to_thread(_process_and_log, s3_client, bucket, key)
        except Exception as exc:
            logger.error(f"Worker error processing {key}: {exc}")
        finally:
            _processing.discard(key)
            _queue.task_done()


# ── Lifespan ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    clients = [_make_s3_client() for _ in range(settings.worker_count)]
    tasks = [asyncio.create_task(_worker(client)) for client in clients]
    yield
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


# ── Auth ───────────────────────────────────────────────────────────────────

def _check_auth(request: Request) -> None:
    token = settings.webhook_auth_token
    if not token:
        return
    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    """Return service health status."""
    return {"status": "ok"}


_BUCKET_RE = re.compile(r"^[a-zA-Z0-9.\-]{1,63}$")
_KEY_RE = re.compile(r"^[^\x00]{1,1024}$")  # no null bytes, reasonable length


@app.post("/notify")
async def notify(request: Request) -> JSONResponse:
    """Handle MinIO webhook notifications and enqueue MCAP files for analysis."""
    _check_auth(request)
    try:
        body = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return JSONResponse({"status": "ignored", "reason": "invalid json"})

    records = body.get("Records", [])
    if not records:
        return JSONResponse({"status": "ignored", "reason": "no records"})

    key = records[0].get("s3", {}).get("object", {}).get("key", "")
    bucket = records[0].get("s3", {}).get("bucket", {}).get("name", settings.minio_bucket)

    if not isinstance(key, str) or not _KEY_RE.match(key):
        logger.warning(f"Rejecting webhook with invalid key: {key!r}")
        return JSONResponse({"status": "ignored", "reason": "invalid_key"})

    if not isinstance(bucket, str) or not _BUCKET_RE.match(bucket):
        logger.warning(f"Rejecting webhook with invalid bucket: {bucket!r}")
        return JSONResponse({"status": "ignored", "reason": "invalid_bucket"})

    if not key.endswith(".mcap"):
        logger.info(f"Skipping non-mcap file: {key}")
        return JSONResponse({"status": "ignored", "reason": "not_mcap"})

    # Dedup: check-then-add is atomic (no await between them in asyncio)
    if key in _processing:
        logger.info(f"Duplicate webhook for {key}, skipping")
        return JSONResponse({"status": "duplicate"})
    _processing.add(key)

    try:
        _queue.put_nowait((bucket, key))
    except asyncio.QueueFull:
        _processing.discard(key)
        logger.warning(f"Queue full, rejecting {key}")
        return JSONResponse({"status": "queue_full"}, status_code=429)

    return JSONResponse({"status": "accepted"})


# ── Processing ─────────────────────────────────────────────────────────────

def _process_and_log(s3_client: Any, bucket: str, key: str) -> None:
    """Download, extract, analyze, and log report. Runs in a thread via asyncio.to_thread."""
    source_file = f"{bucket}/{key}"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, os.path.basename(key))
            s3_client.download_file(bucket, key, local_path)
            if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
                raise OSError(f"S3 download produced empty or missing file: {local_path}")
            _analyze_and_log(source_file, bucket, local_path)
    except Exception as exc:
        report = _builder.build(
            source_file=source_file, bucket=bucket,
            detector_results={}, detector_errors=["minio_download"],
            llm_assessment=None, llm_error=None, duration_seconds=None,
        )
        logger.error(json.dumps(report))


def _analyze_and_log(source_file: str, bucket: str, local_path: str) -> None:
    report = analyze_local_file(local_path, source_file=source_file, bucket=bucket)
    level = "WARNING" if not report["passed"] else "INFO"
    logger.log(level, json.dumps(report, ensure_ascii=False))
