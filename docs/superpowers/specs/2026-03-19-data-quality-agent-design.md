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
  ├─ frames: list[np.ndarray]              (camera topic → video frames)
  ├─ audio_wav: bytes | None               (audio topic → wav)
  └─ sensor_series: dict[str, ndarray]     (IMU/joint topics → time series)
  ▼
AnalysisPipeline  (concurrent.futures.ThreadPoolExecutor)
  ├─ ClarityAnalyzer      → ClarityResult
  ├─ ContinuityAnalyzer   → ContinuityResult
  ├─ FaceDetector         → FaceResult
  ├─ VoiceDetector        → VoiceResult
  └─ GaitDetector         → GaitResult
  ▼
ReportBuilder
  └─ merged JSON → loguru structured log
```

### Library Selections

| Module | Library | Reason |
|---|---|---|
| ROS bag parsing | `rosbags` | Pure Python, no ROS installation required |
| Clarity score | OpenCV `Laplacian` variance + Tenengrad gradient | Pure OpenCV/NumPy, no PyTorch dependency |
| Continuity score | OpenCV `calcOpticalFlowFarneback` | Frame-to-frame optical flow, CPU-efficient |
| Face detection | OpenCV DNN + YuNet `.onnx` model | CPU-friendly, uses already-declared OpenCV dependency |
| Voice detection | `webrtcvad` | Lightweight voice activity detection, no ML runtime |
| Gait detection | OpenCV DNN + MoveNet Lightning `.tflite` via `onnxruntime` | Avoids mediapipe's heavyweight install; ONNX export is ~3 MB |
| Webhook server | `FastAPI` + `uvicorn` | Async, returns 200 immediately via BackgroundTasks |
| Structured logging | `loguru` | Simple JSON output |

**Note:** No PyTorch dependency anywhere. All ML inference goes through OpenCV DNN or `onnxruntime` (CPU wheel, ~10 MB).

---

## JSON Report Format

```json
{
  "report_id": "uuid4",
  "source_file": "robot-data/session_001.bag",
  "minio_bucket": "robot-uploads",
  "analyzed_at": "2026-03-19T14:30:00Z",
  "duration_seconds": 42.5,
  "scores": {
    "clarity": {
      "score": 0.82,
      "method": "laplacian+tenengrad",
      "detail": "mean_laplacian_variance=312.4, mean_tenengrad=1840.2"
    },
    "continuity": {
      "score": 0.91,
      "method": "optical_flow",
      "detail": "mean_flow_magnitude=3.2, discontinuity_frames=2"
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

### Pass/Fail Rules

`passed = true` if and only if ALL of:
- `scores.clarity.score >= CLARITY_THRESHOLD`
- `scores.continuity.score >= CONTINUITY_THRESHOLD`
- `sensitive_info.has_face == false`
- `sensitive_info.has_human_voice == false`
- `sensitive_info.has_human_gait == false`

Any analyzer that raises an exception: its output fields are set to `null` in the report, the analyzer name is appended to `analyzer_errors`, and the corresponding failure reason `"analyzer_error:<name>"` is appended to `failure_reasons`. A `null` field **never** silently passes — it always counts as a failure.

Example with a failed gait detector:
```json
{
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

---

## Analyzer Interface

```python
# agent/analyzers/base.py

from typing import Protocol, TypedDict
import numpy as np

class ExtractedData(TypedDict):
    frames: list[np.ndarray]          # BGR frames, HxWxC uint8
    audio_wav: bytes | None           # raw PCM bytes, 16kHz mono int16
    sensor_series: dict[str, np.ndarray]  # topic_name → (T, D) float array

class ClarityResult(TypedDict):
    score: float                      # [0, 1]
    method: str
    detail: str

class ContinuityResult(TypedDict):
    score: float                      # [0, 1]
    method: str
    detail: str

class FaceResult(TypedDict):
    has_face: bool
    face_count: int

class VoiceResult(TypedDict):
    has_human_voice: bool

class GaitResult(TypedDict):
    has_human_gait: bool

class Analyzer(Protocol):
    def name(self) -> str:
        """Returns the key used to store this analyzer's result in the pipeline output dict."""
        ...
    def analyze(self, data: ExtractedData) -> ClarityResult | ContinuityResult | FaceResult | VoiceResult | GaitResult:
        ...
```

`AnalysisPipeline` uses `analyzer.name()` as the key when collecting results:
```python
results = {analyzer.name(): future.result() for analyzer, future in futures.items()}
```

---

## Project Structure

```
data-quality-agent/
├── agent/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, /notify endpoint
│   ├── config.py            # pydantic-settings env config
│   ├── extractor.py         # BagExtractor: parses .bag, yields ExtractedData
│   ├── pipeline.py          # AnalysisPipeline: ThreadPoolExecutor over analyzers
│   ├── report.py            # ReportBuilder: merges results → JSON log
│   └── analyzers/
│       ├── __init__.py
│       ├── base.py          # Analyzer Protocol + all TypedDicts
│       ├── clarity.py       # ClarityAnalyzer (Laplacian + Tenengrad)
│       ├── continuity.py    # ContinuityAnalyzer (optical flow)
│       ├── face.py          # FaceDetector (OpenCV DNN + YuNet)
│       ├── voice.py         # VoiceDetector (webrtcvad)
│       └── gait.py          # GaitDetector (onnxruntime + MoveNet)
├── tests/
│   ├── test_extractor.py
│   ├── test_pipeline.py
│   ├── test_report.py       # Tests pass/fail logic, null handling, failure_reasons
│   └── analyzers/
│       ├── test_clarity.py
│       ├── test_continuity.py
│       ├── test_face.py
│       ├── test_voice.py
│       └── test_gait.py
├── models/                  # Pre-downloaded ONNX/TFLITE model files
│   ├── yunet.onnx
│   └── movenet_lightning.onnx
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

---

## Deployment

**docker-compose.yml** runs two services: `minio` and `agent`.

### MinIO Webhook Configuration

MinIO requires environment variables to pre-register the webhook endpoint before `mc event add` can reference it. Required MinIO env vars:

```env
MINIO_NOTIFY_WEBHOOK_ENABLE_agent=on
MINIO_NOTIFY_WEBHOOK_ENDPOINT_agent=http://agent:8000/notify
MINIO_NOTIFY_WEBHOOK_AUTH_TOKEN_agent=   # empty = no auth, set if needed
```

After MinIO starts, an init container / entrypoint script runs:

```bash
mc alias set myminio http://minio:9000 $MINIO_ROOT_USER $MINIO_ROOT_PASSWORD
mc mb --ignore-existing myminio/$MINIO_BUCKET
mc event add myminio/$MINIO_BUCKET arn:minio:sqs::agent:webhook --event s3:ObjectCreated:*
```

The ARN format is `arn:minio:sqs::<alias>:webhook` where `<alias>` matches the suffix of `MINIO_NOTIFY_WEBHOOK_ENABLE_<alias>`.

### Agent Endpoint Behavior

`POST /notify` immediately returns HTTP 200 and enqueues processing via FastAPI `BackgroundTasks`. This prevents MinIO from retrying on timeout.

### Environment Variables (Agent)

| Variable | Default | Description |
|---|---|---|
| `MINIO_ENDPOINT` | `minio:9000` | MinIO host:port |
| `MINIO_ACCESS_KEY` | `minioadmin` | Access key |
| `MINIO_SECRET_KEY` | `minioadmin` | Secret key |
| `MINIO_BUCKET` | `robot-uploads` | Bucket to watch |
| `CLARITY_THRESHOLD` | `0.6` | Min passing clarity score |
| `CONTINUITY_THRESHOLD` | `0.6` | Min passing continuity score |
| `LOG_LEVEL` | `INFO` | Log verbosity |
| `MODEL_DIR` | `/app/models` | Directory containing ONNX model files |

---

## Error Handling

- **Corrupt / unreadable bag file:** `BagExtractor` logs error with `source_file`, skips analysis entirely. No report emitted.
- **Missing camera/audio topic:** `BagExtractor` sets the corresponding field to `None` / empty list. Analyzers that require the missing data return a null result with `analyzer_error:<name>` in `failure_reasons`.
- **Analyzer exception:** Caught by `AnalysisPipeline`. Result set to `null`, analyzer name added to `analyzer_errors`, `failure_reasons` gets `"analyzer_error:<name>"`. Other analyzers continue.
- **Temp file cleanup:** `tempfile.TemporaryDirectory` used as context manager; cleaned up on exit regardless of success or failure.

---

## Testing Strategy

- **Unit tests per analyzer:** Synthetic data only (small `np.ndarray` frames, synthetic PCM bytes). No MinIO, no real bag files.
- **`test_report.py`:** Tests `ReportBuilder` pass/fail logic, `null` field handling, `failure_reasons` population, and JSON output format correctness.
- **`test_pipeline.py`:** Runs full pipeline with a minimal synthetic bag file; verifies all analyzer results are collected and errors are handled gracefully.
- **`test_extractor.py`:** Tests `BagExtractor` with a minimal real or synthetic `.bag` file.
- No external services required for any test.
