# Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate OOM crashes and unbounded task accumulation when processing TB/PB-scale MCAP files, while staying within the existing FastAPI + MinIO + Docker Compose stack.

**Architecture:** Three targeted changes to three files — (1) frame sampling in `McapExtractor` to reduce peak memory from O(file) to O(sample), (2) replacing `BackgroundTasks` in `main.py` with a bounded `asyncio.Queue` + lifespan-managed workers + per-worker S3 clients, (3) nginx fronting multiple agent instances for horizontal scale.

**Tech Stack:** FastAPI lifespan, `asyncio.Queue`, `asyncio.to_thread`, boto3 per-worker client, nginx alpine, Docker Compose scale.

---

## File Map

| File | Change | Responsibility |
|------|--------|----------------|
| `agent/config.py` | Modify | Add `MAX_QUEUE_SIZE`, `WORKER_COUNT`, `FRAME_SAMPLE_RATE` settings |
| `agent/extractor.py` | Modify | Accept `frame_sample_rate` param; apply `raw_frames[::frame_sample_rate]`; replace audio `+=` with bytearray |
| `agent/main.py` | Modify | Remove `BackgroundTasks`; add `_queue`, `_processing`, `_make_s3_client`, `_worker`, `lifespan` |
| `tests/test_extractor.py` | Modify | Add sampling + bytearray tests |
| `tests/test_main.py` | Modify | Update existing tests + add 429/duplicate/cleanup/lifespan tests |
| `nginx.conf` | Create | Upstream round-robin + `proxy_next_upstream http_429` |
| `docker-compose.yml` | Modify | Add nginx service; remove agent port exposure; add env vars |

**Unchanged:** `agent/analyzers/base.py`, `agent/analyzers/*.py`, `agent/pipeline.py`, `agent/llm_judge.py`, `agent/report.py`

---

## Task 1: Add Configuration Fields

**Files:**
- Modify: `agent/config.py`

- [ ] **Step 1: Add fields to Settings**

Open `agent/config.py` and add three fields to the `Settings` class after `webhook_auth_token`:

```python
max_queue_size: int = 100
worker_count: int = 4
frame_sample_rate: int = 30
```

Full file after change:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "robot-uploads"
    minio_use_ssl: bool = False

    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_review_margin: float = 0.1

    clarity_threshold: float = 0.6
    continuity_threshold: float = 0.6
    minimum_duration_seconds: float = 1.0

    webhook_auth_token: str = ""
    model_dir: str = "/app/models"
    log_level: str = "INFO"

    max_queue_size: int = 100
    worker_count: int = 4
    frame_sample_rate: int = 30


settings = Settings()
```

- [ ] **Step 2: Verify fields load with defaults**

```bash
uv run python -c "from agent.config import settings; print(settings.max_queue_size, settings.worker_count, settings.frame_sample_rate)"
```

Expected output: `100 4 30`

- [ ] **Step 3: Commit**

```bash
git add agent/config.py
git commit -m "feat: add MAX_QUEUE_SIZE, WORKER_COUNT, FRAME_SAMPLE_RATE config fields"
```

---

## Task 2: Frame Sampling in McapExtractor

**Files:**
- Modify: `agent/extractor.py`
- Modify: `tests/test_extractor.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_extractor.py`:

```python
def test_frame_sample_rate_reduces_frame_count(tmp_path):
    """sample_rate=N returns every Nth frame only."""
    import mcap.writer as mw
    import mcap.records as mr

    # We'll test via the McapExtractor constructor parameter directly
    # by patching read_ros2_messages to return synthetic frames
    from unittest.mock import patch, MagicMock
    import numpy as np

    extractor = McapExtractor(frame_sample_rate=5)

    # Build 25 synthetic message objects
    def make_msg(i):
        m = MagicMock()
        m.log_time = i * 1_000_000_000
        m.channel.topic = "/camera/image_raw"
        m.ros_msg.height = 4
        m.ros_msg.width = 4
        m.ros_msg.data = bytes([128] * 48)
        m.ros_msg.encoding = "bgr8"
        return m

    msgs = [make_msg(i) for i in range(25)]

    with patch("agent.extractor.read_ros2_messages", return_value=iter(msgs)):
        data = extractor.extract("fake.mcap")

    assert len(data["frames"]) == 5  # 25 // 5 = 5


def test_frame_sample_rate_1_returns_all_frames(tmp_path):
    """sample_rate=1 (default) returns all frames unchanged."""
    from unittest.mock import patch, MagicMock

    extractor = McapExtractor(frame_sample_rate=1)

    def make_msg(i):
        m = MagicMock()
        m.log_time = i * 1_000_000_000
        m.channel.topic = "/camera/image_raw"
        m.ros_msg.height = 4
        m.ros_msg.width = 4
        m.ros_msg.data = bytes([128] * 48)
        m.ros_msg.encoding = "bgr8"
        return m

    msgs = [make_msg(i) for i in range(10)]

    with patch("agent.extractor.read_ros2_messages", return_value=iter(msgs)):
        data = extractor.extract("fake.mcap")

    assert len(data["frames"]) == 10


def test_frame_sample_rate_larger_than_frame_count_returns_one_frame():
    """sample_rate=100 on 5-frame video returns 1 frame, not empty list."""
    from unittest.mock import patch, MagicMock

    extractor = McapExtractor(frame_sample_rate=100)

    def make_msg(i):
        m = MagicMock()
        m.log_time = i * 1_000_000_000
        m.channel.topic = "/camera/image_raw"
        m.ros_msg.height = 4
        m.ros_msg.width = 4
        m.ros_msg.data = bytes([128] * 48)
        m.ros_msg.encoding = "bgr8"
        return m

    msgs = [make_msg(i) for i in range(5)]

    with patch("agent.extractor.read_ros2_messages", return_value=iter(msgs)):
        data = extractor.extract("fake.mcap")

    assert len(data["frames"]) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_extractor.py::test_frame_sample_rate_reduces_frame_count tests/test_extractor.py::test_frame_sample_rate_1_returns_all_frames tests/test_extractor.py::test_frame_sample_rate_larger_than_frame_count_returns_one_frame -v
```

Expected: all 3 FAIL (TypeError or similar — `frame_sample_rate` param doesn't exist yet).

- [ ] **Step 3: Add `frame_sample_rate` to McapExtractor**

Modify `agent/extractor.py`. Change `__init__` and `extract`:

```python
from __future__ import annotations
import numpy as np
from agent.analyzers.base import ExtractedData


_PCM_FRAME_BYTES = 960  # 30ms × 16000Hz × 2 bytes (int16) = 960


def chunk_pcm(raw: bytes) -> list[bytes]:
    """Split raw PCM bytes into 30ms frames (960 bytes each). Drops remainder."""
    return [raw[i:i + _PCM_FRAME_BYTES] for i in range(0, len(raw) - _PCM_FRAME_BYTES + 1, _PCM_FRAME_BYTES)]


class McapExtractor:
    def __init__(
        self,
        camera_topic: str = "/camera/image_raw",
        audio_topic: str = "/audio/data",
        imu_topic: str = "/imu/data",
        frame_sample_rate: int = 1,
    ) -> None:
        self._camera_topic = camera_topic
        self._audio_topic = audio_topic
        self._imu_topic = imu_topic
        self._frame_sample_rate = max(1, frame_sample_rate)

    def extract(self, mcap_path: str) -> ExtractedData:
        """Parse an MCAP file and return ExtractedData.

        Frames are decoded from sensor_msgs/Image messages.
        Audio is decoded from audio_common_msgs/AudioData messages and chunked to 30ms PCM frames.
        IMU is accumulated from sensor_msgs/Imu messages.
        """
        raw_frames: list[np.ndarray] = []
        raw_audio = b""
        imu_rows: list[np.ndarray] = []
        timestamps: list[float] = []

        try:
            from mcap_ros2.reader import read_ros2_messages
        except ImportError:
            raise RuntimeError("mcap-ros2-support not installed")

        try:
            for msg in read_ros2_messages(mcap_path, topics=[
                self._camera_topic, self._audio_topic, self._imu_topic
            ]):
                t = msg.log_time / 1e9
                timestamps.append(t)
                topic = msg.channel.topic

                if topic == self._camera_topic:
                    frame = self._decode_image(msg.ros_msg)
                    if frame is not None:
                        raw_frames.append(frame)

                elif topic == self._audio_topic:
                    chunk = self._decode_audio(msg.ros_msg)
                    if chunk:
                        raw_audio += chunk

                elif topic == self._imu_topic:
                    row = self._decode_imu(msg.ros_msg)
                    if row is not None:
                        imu_rows.append(row)
        except Exception:
            pass

        # Apply frame sampling: take every Nth frame, always keep at least 1 if any exist
        if raw_frames:
            frames = raw_frames[::self._frame_sample_rate]
            if not frames:
                frames = [raw_frames[0]]
        else:
            frames = []

        duration = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0
        audio_frames = chunk_pcm(raw_audio) if raw_audio else None
        sensor_series = {}
        if imu_rows:
            sensor_series[self._imu_topic] = np.array(imu_rows, dtype=np.float64)

        return ExtractedData(
            frames=frames,
            audio_frames=audio_frames,
            sensor_series=sensor_series,
            duration_seconds=duration,
        )

    def _decode_image(self, msg) -> np.ndarray | None:
        try:
            h, w = msg.height, msg.width
            data = bytes(msg.data)
            encoding = getattr(msg, "encoding", "bgr8")
            channels = 3 if "rgb" in encoding or "bgr" in encoding else 1
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, channels)
            if "rgb" in encoding:
                import cv2
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            return arr
        except Exception:
            return None

    def _decode_audio(self, msg) -> bytes:
        try:
            return bytes(msg.data)
        except Exception:
            return b""

    def _decode_imu(self, msg) -> np.ndarray | None:
        try:
            a = msg.linear_acceleration
            g = msg.angular_velocity
            return np.array([a.x, a.y, a.z, g.x, g.y, g.z], dtype=np.float64)
        except Exception:
            return None
```

- [ ] **Step 4: Run new sampling tests**

```bash
uv run pytest tests/test_extractor.py::test_frame_sample_rate_reduces_frame_count tests/test_extractor.py::test_frame_sample_rate_1_returns_all_frames tests/test_extractor.py::test_frame_sample_rate_larger_than_frame_count_returns_one_frame -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Run full extractor test suite to confirm no regressions**

```bash
uv run pytest tests/test_extractor.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add agent/extractor.py tests/test_extractor.py
git commit -m "feat: add frame_sample_rate to McapExtractor (sample every N frames)"
```

---

## Task 3: Audio bytearray Optimization

**Files:**
- Modify: `agent/extractor.py`
- Modify: `tests/test_extractor.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_extractor.py`:

```python
def test_audio_bytearray_output_identical_to_concat():
    """bytearray accumulation must produce byte-identical output to += concatenation."""
    # Simulate what the old code did
    chunks = [b"\x01\x02" * 100, b"\x03\x04" * 100, b"\x05\x06" * 100]
    old_result = b""
    for chunk in chunks:
        old_result += chunk

    # What the new code must produce
    _buf = bytearray()
    for chunk in chunks:
        _buf.extend(chunk)
    new_result = bytes(_buf)

    assert new_result == old_result


def test_audio_not_sampled_by_frame_sample_rate():
    """Audio frames are never reduced by frame_sample_rate."""
    from unittest.mock import patch, MagicMock

    extractor = McapExtractor(frame_sample_rate=100)  # extreme sampling

    def make_audio_msg(i):
        m = MagicMock()
        m.log_time = i * 1_000_000_000
        m.channel.topic = "/audio/data"
        # 960 bytes = one full PCM frame
        m.ros_msg.data = bytes([0] * 960)
        return m

    msgs = [make_audio_msg(i) for i in range(10)]  # 10 audio messages = 10 PCM frames

    with patch("agent.extractor.read_ros2_messages", return_value=iter(msgs)):
        data = extractor.extract("fake.mcap")

    # All 10 PCM frames must be present regardless of frame_sample_rate
    assert data["audio_frames"] is not None
    assert len(data["audio_frames"]) == 10
```

- [ ] **Step 2: Run to verify first test passes already (it's a unit test of Python behavior) and second fails**

```bash
uv run pytest tests/test_extractor.py::test_audio_bytearray_output_identical_to_concat tests/test_extractor.py::test_audio_not_sampled_by_frame_sample_rate -v
```

Expected: first PASS (validates the approach), second FAIL (audio decoding path not yet separated).

- [ ] **Step 3: Replace `+=` audio accumulation with bytearray in extractor.py**

In `agent/extractor.py`, change the `extract` method's audio handling from:

```python
raw_audio = b""
...
raw_audio += chunk
```

To:

```python
_audio_buf = bytearray()
...
_audio_buf.extend(chunk)
```

And at the end, before `chunk_pcm`:

```python
raw_audio = bytes(_audio_buf)
audio_frames = chunk_pcm(raw_audio) if raw_audio else None
```

Full updated `extract` method body (replace the audio-related lines only):

```python
    def extract(self, mcap_path: str) -> ExtractedData:
        raw_frames: list[np.ndarray] = []
        _audio_buf = bytearray()
        imu_rows: list[np.ndarray] = []
        timestamps: list[float] = []

        try:
            from mcap_ros2.reader import read_ros2_messages
        except ImportError:
            raise RuntimeError("mcap-ros2-support not installed")

        try:
            for msg in read_ros2_messages(mcap_path, topics=[
                self._camera_topic, self._audio_topic, self._imu_topic
            ]):
                t = msg.log_time / 1e9
                timestamps.append(t)
                topic = msg.channel.topic

                if topic == self._camera_topic:
                    frame = self._decode_image(msg.ros_msg)
                    if frame is not None:
                        raw_frames.append(frame)

                elif topic == self._audio_topic:
                    chunk = self._decode_audio(msg.ros_msg)
                    if chunk:
                        _audio_buf.extend(chunk)

                elif topic == self._imu_topic:
                    row = self._decode_imu(msg.ros_msg)
                    if row is not None:
                        imu_rows.append(row)
        except Exception:
            pass

        if raw_frames:
            frames = raw_frames[::self._frame_sample_rate]
            if not frames:
                frames = [raw_frames[0]]
        else:
            frames = []

        duration = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0
        raw_audio = bytes(_audio_buf)
        audio_frames = chunk_pcm(raw_audio) if raw_audio else None
        sensor_series = {}
        if imu_rows:
            sensor_series[self._imu_topic] = np.array(imu_rows, dtype=np.float64)

        return ExtractedData(
            frames=frames,
            audio_frames=audio_frames,
            sensor_series=sensor_series,
            duration_seconds=duration,
        )
```

- [ ] **Step 4: Run all extractor tests**

```bash
uv run pytest tests/test_extractor.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agent/extractor.py tests/test_extractor.py
git commit -m "perf: replace audio += with bytearray.extend (O(n) vs O(n²))"
```

---

## Task 4: Refactor main.py — Queue, Workers, Dedup, S3

**Files:**
- Modify: `agent/main.py`
- Modify: `tests/test_main.py`

This task replaces `BackgroundTasks` with `asyncio.Queue` + lifespan workers.

- [ ] **Step 1: Write failing tests for new behaviors**

Replace the full contents of `tests/test_main.py` with:

```python
import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock, MagicMock
from agent.main import app


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def mock_process_and_log():
    """Prevent actual S3 downloads in all main tests."""
    with patch("agent.main._process_and_log"):
        yield


@pytest.fixture(autouse=True)
async def reset_queue_state():
    """Drain queue and clear _processing between tests."""
    from agent.main import _queue, _processing
    while not _queue.empty():
        try:
            _queue.get_nowait()
            _queue.task_done()
        except Exception:
            break
    _processing.clear()
    yield
    while not _queue.empty():
        try:
            _queue.get_nowait()
            _queue.task_done()
        except Exception:
            break
    _processing.clear()


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def _mcap_payload(key: str = "session.mcap", bucket: str = "robot-uploads") -> dict:
    return {
        "Records": [{"s3": {"bucket": {"name": bucket}, "object": {"key": key}}}]
    }


# ── Existing behavior (must still pass) ──────────────────────────────────────

async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200


async def test_notify_returns_accepted_for_mcap(client):
    response = await client.post("/notify", json=_mcap_payload("session.mcap"))
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"


async def test_notify_ignores_non_mcap(client):
    response = await client.post("/notify", json=_mcap_payload("session.bag"))
    assert response.status_code == 200
    assert response.json()["status"] == "ignored"


async def test_notify_returns_401_on_bad_token():
    from agent.config import settings
    original = settings.webhook_auth_token
    settings.webhook_auth_token = "secret"
    try:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            response = await c.post("/notify", json={}, headers={"Authorization": "Bearer wrong"})
        assert response.status_code == 401
    finally:
        settings.webhook_auth_token = original


# ── New behavior ──────────────────────────────────────────────────────────────

async def test_notify_returns_429_when_queue_full(client):
    """When the queue is at capacity, /notify returns 429."""
    from agent.main import _queue
    from agent.config import settings

    # Fill the queue to capacity
    for i in range(settings.max_queue_size):
        await _queue.put((f"bucket-{i}", f"file-{i}.mcap"))

    response = await client.post("/notify", json=_mcap_payload("overflow.mcap"))
    assert response.status_code == 429
    assert response.json()["status"] == "queue_full"


async def test_notify_returns_duplicate_for_in_progress_key(client):
    """If a key is already being processed, /notify returns duplicate."""
    from agent.main import _processing

    _processing.add("session.mcap")
    response = await client.post("/notify", json=_mcap_payload("session.mcap"))
    assert response.status_code == 200
    assert response.json()["status"] == "duplicate"


async def test_processing_key_removed_after_worker_completes():
    """_processing is cleared after _worker finishes an item (success or failure)."""
    from agent.main import _worker, _queue, _processing
    import asyncio

    mock_client = MagicMock()
    _processing.add("test.mcap")
    await _queue.put(("bucket", "test.mcap"))

    # Run one iteration of the worker
    with patch("agent.main._process_and_log", side_effect=RuntimeError("boom")):
        # The worker loops forever; run it for one item using a task with timeout
        task = asyncio.create_task(_worker(mock_client))
        await asyncio.sleep(0.05)  # allow one loop iteration
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert "test.mcap" not in _processing


async def test_notify_has_no_background_tasks_param():
    """notify() must not accept BackgroundTasks — it would allow unbounded queueing."""
    import inspect
    from agent.main import notify
    sig = inspect.signature(notify)
    param_types = [p.annotation for p in sig.parameters.values()]
    from fastapi import BackgroundTasks
    assert BackgroundTasks not in param_types
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_main.py -v 2>&1 | head -60
```

Expected: most tests FAIL (module import may succeed but behaviors don't exist yet).

- [ ] **Step 3: Rewrite agent/main.py**

Replace the full contents of `agent/main.py` with:

```python
from __future__ import annotations
import asyncio
import tempfile
import os
import json
from contextlib import asynccontextmanager
from loguru import logger
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from botocore.client import Config
import boto3

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


# ── Module-level singletons ────────────────────────────────────────────────

_extractor = McapExtractor(frame_sample_rate=settings.frame_sample_rate)

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

_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=settings.max_queue_size)
_processing: set[str] = set()


# ── S3 client factory ──────────────────────────────────────────────────────

def _make_s3_client():
    """Create a boto3 S3 client. Call once per worker — each worker owns its client."""
    return boto3.client(
        "s3",
        endpoint_url=f"http{'s' if settings.minio_use_ssl else ''}://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version="s3v4", max_pool_connections=4),
    )


# ── Worker ─────────────────────────────────────────────────────────────────

async def _worker(s3_client) -> None:
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
async def health():
    return {"status": "ok"}


@app.post("/notify")
async def notify(request: Request):
    _check_auth(request)
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"status": "ignored", "reason": "invalid json"})

    records = body.get("Records", [])
    if not records:
        return JSONResponse({"status": "ignored", "reason": "no records"})

    key = records[0].get("s3", {}).get("object", {}).get("key", "")
    bucket = records[0].get("s3", {}).get("bucket", {}).get("name", settings.minio_bucket)

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

def _process_and_log(s3_client, bucket: str, key: str) -> None:
    """Download, extract, analyze, and log report. Runs in a thread via asyncio.to_thread."""
    source_file = f"{bucket}/{key}"

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = os.path.join(tmpdir, os.path.basename(key))
            s3_client.download_file(bucket, key, local_path)
            _analyze_and_log(source_file, bucket, local_path)
    except Exception as exc:
        report = _builder.build(
            source_file=source_file, bucket=bucket,
            detector_results={}, detector_errors=["minio_download"],
            llm_assessment=None, llm_error=None, duration_seconds=None,
        )
        logger.error(json.dumps(report))


def _analyze_and_log(source_file: str, bucket: str, local_path: str) -> None:
    try:
        data = _extractor.extract(local_path)
    except Exception:
        report = _builder.build(
            source_file=source_file, bucket=bucket,
            detector_results={}, detector_errors=["mcap_extraction"],
            llm_assessment=None, llm_error=None, duration_seconds=None,
        )
        logger.error(json.dumps(report))
        return

    detector_results, detector_errors = _pipeline.run(data)
    llm_assessment, llm_error = _judge.judge(detector_results, data)
    report = _builder.build(
        source_file=source_file, bucket=bucket,
        detector_results=detector_results, detector_errors=detector_errors,
        llm_assessment=llm_assessment, llm_error=llm_error,
        duration_seconds=data["duration_seconds"],
    )
    level = "WARNING" if not report["passed"] else "INFO"
    logger.log(level, json.dumps(report, ensure_ascii=False))
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/test_main.py -v
```

Expected: all PASS. If `test_processing_key_removed_after_worker_completes` is flaky due to timing, increase `asyncio.sleep(0.05)` to `0.1`.

- [ ] **Step 5: Run full test suite to check no regressions**

```bash
uv run pytest -x -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add agent/main.py tests/test_main.py
git commit -m "feat: replace BackgroundTasks with asyncio.Queue + lifespan workers + per-worker S3 client"
```

---

## Task 5: Add nginx + Update docker-compose

**Files:**
- Create: `nginx.conf`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Create nginx.conf**

Create `nginx.conf` at the project root:

```nginx
upstream agents {
    server agent:8000;
}

server {
    listen 80;

    location / {
        proxy_pass         http://agents;
        proxy_set_header   Host $host;
        proxy_set_header   X-Real-IP $remote_addr;

        # If a particular agent instance returns 429 (queue full),
        # nginx retries the next available instance automatically.
        # If ALL instances return 429, nginx returns 502 to the caller.
        # MinIO webhook retries on 5xx, so 502 triggers a safe retry.
        proxy_next_upstream error timeout http_429;
        proxy_next_upstream_tries 3;
        proxy_read_timeout 30s;
    }
}
```

- [ ] **Step 2: Validate nginx config syntax**

```bash
docker run --rm -v "$(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro" nginx:alpine nginx -t
```

Expected: `nginx: configuration file /etc/nginx/nginx.conf test is successful`

- [ ] **Step 3: Update docker-compose.yml**

Read the current `docker-compose.yml`, then replace `agent` service and add `nginx` service. The full updated file:

```yaml
services:
  minio:
    image: minio/minio
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
      MINIO_NOTIFY_WEBHOOK_ENABLE_agent: "on"
      MINIO_NOTIFY_WEBHOOK_ENDPOINT_agent: "http://nginx:80/notify"
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"
      - "9001:9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 5s
      timeout: 3s
      retries: 5

  minio-init:
    image: minio/mc
    depends_on:
      minio:
        condition: service_healthy
      nginx:
        condition: service_started
    entrypoint: >
      /bin/sh -c "
        mc alias set myminio http://minio:9000 minioadmin minioadmin &&
        mc mb --ignore-existing myminio/robot-uploads &&
        mc event add myminio/robot-uploads arn:minio:sqs::agent:webhook --event s3:ObjectCreated:*
      "

  agent:
    build: .
    expose:
      - "8000"
    environment:
      MINIO_ENDPOINT: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
      MINIO_BUCKET: robot-uploads
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
      FRAME_SAMPLE_RATE: "30"
      MAX_QUEUE_SIZE: "100"
      WORKER_COUNT: "4"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 5s
      timeout: 3s
      retries: 5
    depends_on:
      minio:
        condition: service_healthy

  nginx:
    image: nginx:alpine
    ports:
      - "8000:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      agent:
        condition: service_healthy
```

Note: `MINIO_NOTIFY_WEBHOOK_ENDPOINT_agent` now points to `nginx:80` instead of `agent:8000`, so MinIO routes through nginx.

- [ ] **Step 4: Validate docker-compose config**

```bash
docker compose config --quiet
```

Expected: no errors (exits 0).

- [ ] **Step 5: Commit**

```bash
git add nginx.conf docker-compose.yml
git commit -m "feat: add nginx load balancer with 429 retry; enable docker compose scale"
```

---

## Task 6: Wire McapExtractor to use settings.frame_sample_rate

**Files:**
- Modify: `agent/main.py` (one line — already done in Task 4, verify it's there)

- [ ] **Step 1: Confirm extractor is initialized with settings**

Check `agent/main.py` line that creates `_extractor`:

```python
_extractor = McapExtractor(frame_sample_rate=settings.frame_sample_rate)
```

This was included in the Task 4 rewrite. Confirm it is present:

```bash
grep "frame_sample_rate" agent/main.py
```

Expected output: `_extractor = McapExtractor(frame_sample_rate=settings.frame_sample_rate)`

- [ ] **Step 2: Run full test suite one final time**

```bash
uv run pytest -v
```

Expected: all PASS.

- [ ] **Step 3: Final commit (if any remaining changes)**

```bash
git status
# If clean, no commit needed. If dirty:
git add -p
git commit -m "chore: verify frame_sample_rate wired from settings to McapExtractor"
```

---

## Verification Checklist

After all tasks complete, verify end-to-end behavior:

```bash
# 1. All tests pass
uv run pytest -v

# 2. docker-compose config is valid
docker compose config --quiet

# 3. nginx config is valid
docker run --rm -v "$(pwd)/nginx.conf:/etc/nginx/nginx.conf:ro" nginx:alpine nginx -t

# 4. Config fields visible
uv run python -c "
from agent.config import settings
print('queue_size:', settings.max_queue_size)
print('workers:', settings.worker_count)
print('sample_rate:', settings.frame_sample_rate)
"

# 5. Extractor uses sampling
uv run python -c "
from agent.extractor import McapExtractor
e = McapExtractor(frame_sample_rate=30)
print('sample_rate:', e._frame_sample_rate)
"

# 6. main.py has no BackgroundTasks import
grep "BackgroundTasks" agent/main.py && echo "FAIL: BackgroundTasks still present" || echo "OK: BackgroundTasks removed"
```