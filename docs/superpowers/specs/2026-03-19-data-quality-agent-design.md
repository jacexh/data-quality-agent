# Data Quality Agent — Design Spec

**Date:** 2026-03-19
**Status:** Approved

---

## Overview

An automated data quality assessment agent for robot-collected data. When a ROS bag file is uploaded to a MinIO bucket, the agent is triggered via bucket notification, analyzes the video and sensor data, and outputs a structured JSON quality report to logs.

---

## Requirements

- **Trigger:** MinIO object-created event (webhook)
- **Input:** ROS bag files (`.bag` / `.mcap`) containing camera, audio, and sensor topics
- **Output:** JSON quality report written to structured logs (no database, no file storage)
- **Environment:** CPU-only, offline batch processing acceptable
- **Sensitive info handling:** Detect and report only (no redaction or rejection)

---

## Architecture

### Data Flow

```
MinIO
  │  (bucket notification: s3:ObjectCreated:*)
  ▼
Webhook Server (FastAPI, POST /notify)
  │  async background task: download .bag to temp dir
  ▼
BagExtractor
  ├─ frames: list[np.ndarray]              (camera topic → video frames, may be empty)
  ├─ audio_frames: list[bytes] | None      (audio topic → 30ms PCM frames, 16kHz mono int16)
  └─ sensor_series: dict[str, ndarray]     (IMU/joint topics → time series)
  ▼
AnalysisPipeline  (concurrent.futures.ThreadPoolExecutor)
  ├─ ClarityAnalyzer      → ClarityResult      (name: "clarity")
  ├─ ContinuityAnalyzer   → ContinuityResult   (name: "continuity")
  ├─ FaceDetector         → FaceResult         (name: "face")
  ├─ VoiceDetector        → VoiceResult        (name: "voice")
  └─ GaitDetector         → GaitResult         (name: "gait")
  ▼
ReportBuilder
  └─ merged JSON → loguru structured log
```

### Library Selections

| Module | Library | Reason |
|---|---|---|
| ROS bag parsing | `rosbags` | Pure Python, no ROS installation required |
| Clarity score | OpenCV `Laplacian` variance + Tenengrad | Pure OpenCV/NumPy, no PyTorch dependency |
| Continuity score | OpenCV `calcOpticalFlowFarneback` | Frame-to-frame optical flow, CPU-efficient |
| Face detection | OpenCV DNN + YuNet `.onnx` (~400 KB) | CPU-friendly, uses already-declared OpenCV DNN module |
| Voice detection | `webrtcvad` | Lightweight VAD; requires pre-framed PCM (see below) |
| Gait detection | OpenCV `HOGDescriptor` (people detector) | Already a dependency; binary person-presence flag is sufficient; no extra model file |
| Webhook server | `FastAPI` + `uvicorn` | Async, returns 200 immediately via `BackgroundTasks` |
| Structured logging | `loguru` | Simple JSON output |

**No PyTorch dependency anywhere.** All ML inference uses OpenCV DNN (with bundled `.onnx` model files) or built-in OpenCV algorithms.

**Gait detection rationale:** Full pose estimation (MoveNet) is over-engineered for a binary "human gait present" flag. OpenCV's HOG pedestrian detector identifies walking humans without requiring any additional model download or conversion.

---

## JSON Report Format

### Normal path (all analyzers succeed)

```json
{
  "report_id": "550e8400-e29b-41d4-a716-446655440000",
  "source_file": "robot-data/session_001.bag",
  "minio_bucket": "robot-uploads",
  "analyzed_at": "2026-03-19T14:30:00Z",
  "duration_seconds": 42.5,
  "scores": {
    "clarity": {
      "score": 0.82,
      "method": "laplacian+tenengrad",
      "detail": {"mean_laplacian_variance": 312.4, "mean_tenengrad": 1840.2}
    },
    "continuity": {
      "score": 0.91,
      "method": "optical_flow",
      "detail": {"mean_flow_magnitude": 3.2, "discontinuity_frames": 2}
    }
  },
  "sensitive_info": {
    "has_face": true,
    "face_count": 1,
    "has_human_voice": false,
    "has_human_gait": false
  },
  "analyzer_errors": [],
  "passed": false,
  "failure_reasons": ["has_face"]
}
```

### Error path (MinIO download failure)

If the `.bag` file cannot be downloaded (network error, file deleted, credential failure):
```json
{
  "report_id": "550e8400-e29b-41d4-a716-446655440001",
  "source_file": "robot-data/session_001.bag",
  "minio_bucket": "robot-uploads",
  "analyzed_at": "2026-03-19T14:30:00Z",
  "duration_seconds": null,
  "scores": null,
  "sensitive_info": null,
  "analyzer_errors": ["minio_download"],
  "passed": false,
  "failure_reasons": ["analyzer_error:minio_download"]
}
```

### Error path (one analyzer fails, corrupt bag)

If BagExtractor fails entirely:
```json
{
  "report_id": "...",
  "source_file": "robot-data/bad_file.bag",
  "minio_bucket": "robot-uploads",
  "analyzed_at": "2026-03-19T14:30:00Z",
  "duration_seconds": null,
  "scores": null,
  "sensitive_info": null,
  "analyzer_errors": ["bag_extraction"],
  "passed": false,
  "failure_reasons": ["analyzer_error:bag_extraction"]
}
```

If a single analyzer fails (e.g., gait):
```json
{
  "report_id": "...",
  "source_file": "robot-data/session_001.bag",
  "minio_bucket": "robot-uploads",
  "analyzed_at": "2026-03-19T14:30:00Z",
  "duration_seconds": 42.5,
  "scores": {
    "clarity": {"score": 0.82, "method": "laplacian+tenengrad", "detail": {...}},
    "continuity": {"score": 0.91, "method": "optical_flow", "detail": {...}}
  },
  "sensitive_info": {
    "has_face": false,
    "face_count": 0,
    "has_human_voice": false,
    "has_human_gait": null
  },
  "analyzer_errors": ["gait"],
  "passed": false,
  "failure_reasons": ["analyzer_error:gait"]
}
```

**`detail` fields are structured objects** (not free-form strings) to allow log aggregators to parse individual metrics.

### Pass/Fail Rules

`passed = true` if and only if ALL of:
- `duration_seconds >= MINIMUM_DURATION_SECONDS` (default: 1.0)
- `scores.clarity.score >= CLARITY_THRESHOLD`
- `scores.continuity.score >= CONTINUITY_THRESHOLD`
- `sensitive_info.has_face == false`
- `sensitive_info.has_human_voice == false`
- `sensitive_info.has_human_gait == false`
- `analyzer_errors` is empty

Any `null` score or sensitive info field (from an analyzer error) always counts as a failure. A `null` field **never** silently passes.

---

## Analyzer Interface

```python
# agent/analyzers/base.py

from typing import Protocol, TypedDict
import numpy as np


class ExtractedData(TypedDict):
    frames: list[np.ndarray]           # BGR frames HxWxC uint8; may be empty list
    audio_frames: list[bytes] | None   # 30ms PCM chunks, 16kHz mono int16; None if no audio topic
    sensor_series: dict[str, np.ndarray]  # topic_name → (T, D) float64 array
    duration_seconds: float            # total bag duration


class ClarityDetail(TypedDict):
    mean_laplacian_variance: float     # higher = sharper
    mean_tenengrad: float              # higher = sharper
    frame_count: int                   # 0 when frames is empty


class ContinuityDetail(TypedDict):
    mean_flow_magnitude: float         # average optical flow magnitude across frame pairs
    discontinuity_frames: int          # number of frames where flow exceeds discontinuity threshold
    frame_count: int                   # 0 when frames is empty


class ClarityResult(TypedDict):
    score: float                       # [0.0, 1.0]
    method: str                        # "laplacian+tenengrad"
    detail: ClarityDetail


class ContinuityResult(TypedDict):
    score: float                       # [0.0, 1.0]
    method: str                        # "optical_flow"
    detail: ContinuityDetail


class FaceResult(TypedDict):
    has_face: bool
    face_count: int


class VoiceResult(TypedDict):
    has_human_voice: bool


class GaitResult(TypedDict):
    has_human_gait: bool


class Analyzer(Protocol):
    def name(self) -> str:
        """Canonical key used by ReportBuilder to place this result in the report.
        Required values: "clarity" | "continuity" | "face" | "voice" | "gait"
        """
        ...

    def analyze(self, data: ExtractedData) -> ClarityResult | ContinuityResult | FaceResult | VoiceResult | GaitResult:
        """Analyze the extracted data and return a typed result dict.
        Must handle empty frames list (return score=0.0 or has_*=False with analyzer_error).
        Must not raise — exceptions are caught by AnalysisPipeline.
        """
        ...
```

**PCM framing contract:** `BagExtractor` is responsible for chunking the raw audio bytes from the bag into 30 ms frames (480 samples at 16kHz × 2 bytes = 960 bytes each) before populating `audio_frames`. `VoiceDetector` passes these frames directly to `webrtcvad` without further chunking.

**Empty frames handling:** When `frames` is an empty list, `ClarityAnalyzer` and `ContinuityAnalyzer` return `score=0.0` with `detail.frame_count=0`. `GaitDetector` and `FaceDetector` return `False`. These are valid results, not errors — `passed` will be `false` due to low scores but `analyzer_errors` will be empty.

**No camera topic:** If the bag has no camera topic at all, `BagExtractor` sets `frames=[]`. The bag will fail on `clarity` and `continuity` scores (both 0.0 < threshold). This is by design — the agent is intended for robot data that includes video; bags without a camera topic are expected to fail quality checks. No special `"no_camera_topic"` failure reason is emitted; `failure_reasons` will include `"clarity"` and `"continuity"`.

---

## Project Structure

```
data-quality-agent/
├── agent/
│   ├── __init__.py
│   ├── main.py              # FastAPI app: POST /notify endpoint + BackgroundTasks
│   ├── config.py            # pydantic-settings env config
│   ├── extractor.py         # BagExtractor: parses .bag → ExtractedData (with PCM framing)
│   ├── pipeline.py          # AnalysisPipeline: ThreadPoolExecutor over Analyzer list
│   ├── report.py            # ReportBuilder: merges results → JSON log
│   └── analyzers/
│       ├── __init__.py
│       ├── base.py          # Analyzer Protocol + all TypedDicts
│       ├── clarity.py       # ClarityAnalyzer (name: "clarity")
│       ├── continuity.py    # ContinuityAnalyzer (name: "continuity")
│       ├── face.py          # FaceDetector (name: "face", OpenCV DNN + YuNet)
│       ├── voice.py         # VoiceDetector (name: "voice", webrtcvad)
│       └── gait.py          # GaitDetector (name: "gait", OpenCV HOG)
├── tests/
│   ├── conftest.py          # Shared fixtures: synthetic ExtractedData, frames, PCM bytes
│   ├── test_extractor.py
│   ├── test_pipeline.py
│   ├── test_report.py       # Pass/fail logic, null handling, failure_reasons, report_id uniqueness, analyzed_at format
│   ├── test_main.py         # POST /notify returns 200 immediately; BackgroundTask is enqueued
│   └── analyzers/
│       ├── test_clarity.py
│       ├── test_continuity.py
│       ├── test_face.py
│       ├── test_voice.py
│       └── test_gait.py
├── models/
│   └── yunet.onnx           # YuNet face detection model (~400 KB, bundled in repo)
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

---

## Deployment

### docker-compose.yml Structure

```yaml
services:
  minio:
    image: minio/minio
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
      # Pre-register the webhook endpoint so mc event add can reference the ARN
      MINIO_NOTIFY_WEBHOOK_ENABLE_agent: "on"
      MINIO_NOTIFY_WEBHOOK_ENDPOINT_agent: "http://agent:8000/notify"
    command: server /data --console-address ":9001"
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
      agent:
        condition: service_healthy   # wait for agent to be ready before registering events
    entrypoint: >
      /bin/sh -c "
        mc alias set myminio http://minio:9000 minioadmin minioadmin &&
        mc mb --ignore-existing myminio/robot-uploads &&
        mc event add myminio/robot-uploads arn:minio:sqs::agent:webhook --event s3:ObjectCreated:*
      "

  agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      MINIO_ENDPOINT: minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
      MINIO_BUCKET: robot-uploads
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 5s
      timeout: 3s
      retries: 5
```

**Startup sequencing:** `minio-init` depends on both `minio` and `agent` being healthy before registering the bucket event. This prevents lost events at startup. The agent exposes `GET /health` (returns 200) for the healthcheck.

**ARN format:** `arn:minio:sqs::<alias>:webhook` where `<alias>` is the suffix of `MINIO_NOTIFY_WEBHOOK_ENABLE_<alias>` (`agent` in this case).

**Auth token:** `MINIO_NOTIFY_WEBHOOK_AUTH_TOKEN_agent` is left unset (no auth) for local development. If set, the agent must validate the `Authorization: Bearer <token>` header in `POST /notify` using a corresponding `WEBHOOK_AUTH_TOKEN` env var. The FastAPI endpoint checks this header and returns 401 if it does not match.

### Dockerfile (key lines)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install uv && uv pip install --system .
COPY agent/ ./agent/
COPY models/ ./models/          # bundles yunet.onnx into /app/models/
CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The `models/` directory is copied into `/app/models/` at build time. `MODEL_DIR` defaults to `/app/models` and must match this path.

### Agent Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MINIO_ENDPOINT` | `minio:9000` | MinIO host:port |
| `MINIO_ACCESS_KEY` | `minioadmin` | Access key |
| `MINIO_SECRET_KEY` | `minioadmin` | Secret key |
| `MINIO_BUCKET` | `robot-uploads` | Bucket to watch |
| `CLARITY_THRESHOLD` | `0.6` | Min passing clarity score |
| `CONTINUITY_THRESHOLD` | `0.6` | Min passing continuity score |
| `MINIMUM_DURATION_SECONDS` | `1.0` | Min bag duration to be considered valid |
| `WEBHOOK_AUTH_TOKEN` | `""` | If non-empty, validate Bearer token on /notify |
| `MODEL_DIR` | `/app/models` | Directory containing bundled model files |
| `LOG_LEVEL` | `INFO` | Log verbosity |

---

## Error Handling

| Failure | Behavior |
|---|---|
| Corrupt / unreadable bag | `BagExtractor` raises; a structured error report is emitted to log (`scores: null`, `sensitive_info: null`, `analyzer_errors: ["bag_extraction"]`, `passed: false`). Temp dir cleaned up. |
| MinIO download failure | Caught in `BackgroundTask`. Structured error report emitted (all fields as per "MinIO download failure" JSON example above). `analyzer_errors: ["minio_download"]`, `passed: false`. |
| Single analyzer raises | Caught by `AnalysisPipeline`. Field set to `null`, name appended to `analyzer_errors`, `"analyzer_error:<name>"` in `failure_reasons`. Other analyzers continue. |
| Camera topic missing / empty frames | `frames = []`. Analyzers return `score=0.0` or `has_*=False`. Valid result, not an error — bag fails due to low score. |
| Audio topic missing | `audio_frames = None`. `VoiceDetector` returns `has_human_voice=False`. Valid result. |
| Invalid webhook auth token | `POST /notify` returns HTTP 401. No processing. |

**Temp file cleanup:** `tempfile.TemporaryDirectory` used as context manager in `BackgroundTask`; cleaned up on exit regardless of success or failure.

---

## Testing Strategy

| Test file | What it covers |
|---|---|
| `conftest.py` | Synthetic `ExtractedData` fixture: small BGR frames (4×4 px), synthetic 30ms PCM frame bytes, `sensor_series: {}` (no sensor analyzers in current scope); shared across all test files |
| `test_main.py` | `POST /notify` returns 200 immediately; background task enqueued; `GET /health` returns 200; 401 on bad auth token |
| `test_extractor.py` | BagExtractor with a minimal synthetic bag; PCM chunking produces 30ms frames |
| `test_pipeline.py` | Full pipeline with synthetic data; analyzer errors are caught and do not abort other analyzers |
| `test_report.py` | Pass/fail logic; all null fields produce `passed=false`; `report_id` is a valid UUID4 and unique per call; `analyzed_at` is ISO 8601 UTC; `failure_reasons` populated correctly |
| `analyzers/test_clarity.py` | Score on sharp vs. blurry synthetic frames; empty frames → score=0.0 |
| `analyzers/test_continuity.py` | Score on smooth vs. jumpy frame sequence; empty frames → score=0.0 |
| `analyzers/test_face.py` | No-face frames → `has_face=False`; empty frames → `has_face=False` |
| `analyzers/test_voice.py` | Silent PCM → `False`; `None` audio → `False` |
| `analyzers/test_gait.py` | Empty frames → `has_human_gait=False` |

No external services required for any test.
