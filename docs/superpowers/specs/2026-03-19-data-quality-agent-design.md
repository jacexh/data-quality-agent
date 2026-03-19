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
  ├─ frames: list[np.ndarray]       (camera topic → video frames)
  ├─ audio_wav: bytes | None        (audio topic → wav)
  └─ sensor_series: dict[str, ndarray]  (IMU/joint topics → time series)
  ▼
AnalysisPipeline  (concurrent.futures.ThreadPoolExecutor)
  ├─ ClarityAnalyzer      → clarity_score: float [0,1]
  ├─ ContinuityAnalyzer   → continuity_score: float [0,1]
  ├─ FaceDetector         → has_face: bool, face_count: int
  ├─ VoiceDetector        → has_voice: bool
  └─ GaitDetector         → has_gait: bool
  ▼
ReportBuilder
  └─ merged JSON → loguru structured log
```

### Library Selections

| Module | Library | Reason |
|---|---|---|
| ROS bag parsing | `rosbags` | Pure Python, no ROS installation required |
| Clarity score | OpenCV `Laplacian` + `piq` BRISQUE | Complementary: sharpness + perceptual quality |
| Continuity score | OpenCV `calcOpticalFlowFarneback` | Frame-to-frame optical flow, CPU-efficient |
| Face detection | OpenCV DNN + YuNet model | CPU-friendly, no heavy dependencies |
| Voice detection | `webrtcvad` | Lightweight voice activity detection |
| Gait detection | `mediapipe` Pose | Skeleton joint time series, CPU-capable |
| Webhook server | `FastAPI` + `uvicorn` | Async, returns 200 immediately via BackgroundTasks |
| Structured logging | `loguru` | Simple JSON output |

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
      "method": "laplacian+brisque",
      "detail": "mean_laplacian_variance=312.4, brisque=28.1"
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
  "passed": false,
  "failure_reasons": ["has_face"]
}
```

**Pass criteria:** `clarity.score >= CLARITY_THRESHOLD` AND `continuity.score >= CONTINUITY_THRESHOLD` AND all sensitive info flags are `false`. Any violation adds to `failure_reasons`.

---

## Analyzer Interface

```python
# agent/analyzers/base.py

class ExtractedData(TypedDict):
    frames: list[np.ndarray]
    audio_wav: bytes | None
    sensor_series: dict[str, np.ndarray]

class Analyzer(Protocol):
    def name(self) -> str: ...
    def analyze(self, data: ExtractedData) -> dict: ...
```

All analyzers implement this Protocol. `AnalysisPipeline` runs them concurrently via `ThreadPoolExecutor` and collects results.

---

## Project Structure

```
data-quality-agent/
├── agent/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, /notify endpoint
│   ├── config.py            # Pydantic-settings env config
│   ├── extractor.py         # BagExtractor
│   ├── pipeline.py          # AnalysisPipeline
│   ├── report.py            # ReportBuilder
│   └── analyzers/
│       ├── __init__.py
│       ├── base.py          # Analyzer Protocol + ExtractedData
│       ├── clarity.py       # ClarityAnalyzer
│       ├── continuity.py    # ContinuityAnalyzer
│       ├── face.py          # FaceDetector
│       ├── voice.py         # VoiceDetector
│       └── gait.py          # GaitDetector
├── tests/
│   ├── test_extractor.py
│   ├── test_pipeline.py
│   └── analyzers/
│       ├── test_clarity.py
│       ├── test_continuity.py
│       ├── test_face.py
│       ├── test_voice.py
│       └── test_gait.py
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

---

## Deployment

**docker-compose.yml** runs two services:

- `minio`: Official MinIO image. On startup, `mc` CLI creates the bucket and registers the webhook event (`mc event add myminio/robot-uploads arn:... --event s3:ObjectCreated:*`).
- `agent`: FastAPI service on port 8000, exposes `POST /notify`.

MinIO sends `POST /notify` on every object creation. The endpoint immediately returns HTTP 200 and processes the bag file in a `BackgroundTask` to avoid blocking MinIO's notification delivery.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MINIO_ENDPOINT` | `minio:9000` | MinIO host:port |
| `MINIO_ACCESS_KEY` | `minioadmin` | Access key |
| `MINIO_SECRET_KEY` | `minioadmin` | Secret key |
| `MINIO_BUCKET` | `robot-uploads` | Bucket to watch |
| `CLARITY_THRESHOLD` | `0.6` | Min passing clarity score |
| `CONTINUITY_THRESHOLD` | `0.6` | Min passing continuity score |
| `LOG_LEVEL` | `INFO` | Log verbosity |

---

## Error Handling

- If `BagExtractor` fails (corrupt file, missing topics): log error with `source_file` and skip analysis.
- If any single `Analyzer` raises: log warning, mark that field as `null` in the report, continue with remaining analyzers.
- Temp files cleaned up via `contextlib.contextmanager` / `tempfile.TemporaryDirectory` regardless of success or failure.

---

## Testing Strategy

- **Unit tests:** Each analyzer tested independently with synthetic data (small numpy arrays, synthetic audio bytes).
- **Integration test:** `test_pipeline.py` runs the full pipeline with a minimal synthetic bag file.
- No external services required in tests (MinIO not needed for unit/integration tests).
