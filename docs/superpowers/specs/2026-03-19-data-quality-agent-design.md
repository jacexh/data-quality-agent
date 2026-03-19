# Data Quality Agent — Design Spec

**Date:** 2026-03-19
**Status:** Approved

---

## Overview

An automated data quality assessment agent for robot-collected data. When a `.mcap` file is uploaded to a MinIO bucket, the agent is triggered via bucket notification. Algorithmic detectors run first (fast, deterministic), then a Claude LLM Agent acts as final judge — resolving ambiguous detections, correlating cross-modal signals, and generating a natural language assessment alongside the structured JSON report. The report is written to structured logs.

---

## Requirements

- **Trigger:** MinIO object-created event (webhook), `.mcap` files only
- **Input:** MCAP files containing camera, audio, and sensor topics
- **Output:** JSON quality report (with LLM narrative) written to structured logs
- **Environment:** CPU-only for algorithmic detectors; LLM calls go to Anthropic API
- **Sensitive info handling:** Detect and report only (no redaction or rejection)

---

## Architecture

### Data Flow

```
MinIO
  │  (bucket notification: s3:ObjectCreated:*, .mcap only)
  ▼
Webhook Server (FastAPI, POST /notify)
  │  async background task: download .mcap to temp dir
  ▼
McapExtractor
  ├─ frames: list[np.ndarray]              (camera topic → video frames, may be empty)
  ├─ audio_frames: list[bytes] | None      (audio topic → 30ms PCM frames, 16kHz mono int16)
  ├─ sensor_series: dict[str, ndarray]     (IMU/joint topics → time series)
  └─ duration_seconds: float
  ▼
AnalysisPipeline  (concurrent.futures.ThreadPoolExecutor)
  ├─ ClarityAnalyzer      → ClarityResult      (name: "clarity")
  ├─ ContinuityAnalyzer   → ContinuityResult   (name: "continuity")
  ├─ FaceDetector         → FaceResult         (name: "face")
  ├─ VoiceDetector        → VoiceResult        (name: "voice")
  └─ GaitDetector         → GaitResult         (name: "gait")
  ▼
LLM Agent (Claude claude-sonnet-4-6, vision-enabled)
  │  Receives detector results + key frames + IMU summary
  │  Uses tool_use to query specific data when needed
  │  Resolves ambiguous detections, correlates cross-modal signals
  │  Makes final pass/fail verdict and generates narrative
  ▼
ReportBuilder
  └─ merged JSON (detector results + LLM assessment) → loguru structured log
```

### Where LLM Adds Value vs. Pure Algorithm

| Scenario | Why LLM is needed |
|---|---|
| Face on screen/poster | YuNet fires, LLM inspects frame and determines it's not a live human |
| Robot arm mimicking gait | HOG fires, LLM correlates with IMU (no bipedal motion pattern) → false positive |
| Voice vs. factory noise | webrtcvad fires, LLM correlates with high IMU vibration → likely noise |
| Motion blur during fast turn | Clarity score low, LLM sees IMU shows high angular velocity → expected, contextual pass |
| Borderline scores (e.g. 0.57 vs threshold 0.6) | LLM reviews key frames holistically before final verdict |
| Natural language report | LLM converts structured results into human-readable explanation |

**Algorithmic detectors are always authoritative for clear-cut cases.** LLM only intervenes when:
1. Any sensitive-info detector returned `True`, OR
2. Any score is within `LLM_REVIEW_MARGIN` of its threshold (default: 0.1), OR
3. There is cross-modal ambiguity (e.g. voice detected but no face/gait)

This keeps LLM API costs low — most clean recordings never trigger LLM review.

### Library Selections

| Module | Library | Reason |
|---|---|---|
| MCAP parsing | `mcap` + `mcap-ros2-support` | Official MCAP Python SDK, no ROS installation required |
| Clarity score | OpenCV `Laplacian` variance + Tenengrad | Pure OpenCV/NumPy, no PyTorch dependency |
| Continuity score | OpenCV `calcOpticalFlowFarneback` | Frame-to-frame optical flow, CPU-efficient |
| Face detection | OpenCV DNN + YuNet `.onnx` (~400 KB) | CPU-friendly, uses already-declared OpenCV DNN module |
| Voice detection | `webrtcvad` | Lightweight VAD; requires pre-framed PCM (see below) |
| Gait detection | OpenCV `HOGDescriptor` (people detector) | Binary person-presence; no extra model file needed |
| LLM Agent | `anthropic` SDK, model `claude-sonnet-4-6` | Multimodal (vision); tool_use for structured data queries |
| Webhook server | `FastAPI` + `uvicorn` | Async, returns 200 immediately via `BackgroundTasks` |
| Structured logging | `loguru` | Simple JSON output |

**No PyTorch dependency anywhere.** All local ML inference uses OpenCV DNN or built-in OpenCV algorithms.

---

## LLM Agent Design

### Role

The LLM Agent is a **final-stage judge**, not a replacement for algorithmic detectors. It receives the full detector output and a set of tools to query additional data from the extracted MCAP, then produces:
1. A final `passed` verdict (may override detector-based verdict for ambiguous cases)
2. A `narrative` string explaining the assessment in natural language

### Tools Available to the Agent

```python
# These are passed as tools in the Claude API call

get_key_frames(frame_indices: list[int]) -> list[str]
# Returns base64-encoded JPEG images for specified frame indices
# Agent uses this to visually inspect frames flagged by detectors

get_imu_summary(window_start: float, window_end: float) -> dict
# Returns {"mean_acceleration": float, "max_angular_velocity": float, "mean_angular_velocity": float}
# for the given time window (seconds). Agent uses this to correlate motion context.

get_detector_results() -> dict
# Returns the full AnalysisPipeline output (always available as context, but tool allows re-querying)
```

### Agent Invocation

The agent is invoked with a system prompt that defines its role and decision criteria:

```
You are a data quality judge for robot-collected MCAP recordings.
You have access to algorithmic detector results and tools to inspect key frames and IMU data.

Your job:
1. Review any flagged sensitive information detections (face/voice/gait). Use get_key_frames
   to verify whether detections are genuine (live human) or false positives (poster, screen,
   robot arm, background noise correlated with high IMU vibration).
2. For borderline quality scores (within 0.1 of threshold), review key frames and IMU context
   to determine if degradation is expected (motion blur during fast maneuver) or problematic.
3. Produce a final passed: true/false verdict and a concise narrative (2-4 sentences) explaining
   your reasoning.

Rules:
- If you cannot inspect a frame (tool error), treat the original detector result as authoritative.
- Do not override a clear failure (e.g. score < 0.3, unambiguous live human face) — only
  reconsider borderline or ambiguous cases.
- Your narrative must be in the same language as this prompt (Chinese or English per config).
```

### LLM Invocation Conditions

```python
should_invoke_llm = (
    any sensitive_info flag is True
    or any score is within LLM_REVIEW_MARGIN of its threshold
    or (voice_detected and not face_detected and not gait_detected)
)
```

If `should_invoke_llm` is False, the LLM step is skipped entirely. The report's `llm_assessment` field is set to `null` and `llm_skipped_reason` explains why.

---

## JSON Report Format

### Normal path — LLM invoked

```json
{
  "report_id": "550e8400-e29b-41d4-a716-446655440000",
  "source_file": "robot-data/session_001.mcap",
  "minio_bucket": "robot-uploads",
  "analyzed_at": "2026-03-19T14:30:00Z",
  "duration_seconds": 42.5,
  "scores": {
    "clarity": {
      "score": 0.62,
      "method": "laplacian+tenengrad",
      "detail": {"mean_laplacian_variance": 180.4, "mean_tenengrad": 920.2, "frame_count": 420}
    },
    "continuity": {
      "score": 0.91,
      "method": "optical_flow",
      "detail": {"mean_flow_magnitude": 3.2, "discontinuity_frames": 2, "frame_count": 420}
    }
  },
  "sensitive_info": {
    "has_face": true,
    "face_count": 1,
    "has_human_voice": false,
    "has_human_gait": false
  },
  "llm_assessment": {
    "passed": false,
    "overrode_detector": false,
    "narrative": "检测到一张人脸（第142帧），经视觉确认为真实人脸，非屏幕或海报。清晰度评分偏低（0.62），但IMU数据显示该时段机器人处于匀速行驶状态，模糊原因需进一步排查。综合判断：数据集因包含人脸信息不通过质量审核。",
    "frames_reviewed": [142, 143, 144],
    "imu_windows_reviewed": []
  },
  "analyzer_errors": [],
  "passed": false,
  "failure_reasons": ["has_face"]
}
```

### Normal path — LLM skipped (clean recording)

```json
{
  "report_id": "...",
  "source_file": "robot-data/session_clean.mcap",
  ...
  "llm_assessment": null,
  "llm_skipped_reason": "all_detectors_clear_no_borderline_scores",
  "analyzer_errors": [],
  "passed": true,
  "failure_reasons": []
}
```

### Normal path — LLM overrides detector (false positive)

```json
{
  "llm_assessment": {
    "passed": true,
    "overrode_detector": true,
    "override_detail": "face detector fired on frame 88; visual inspection confirms face is displayed on a monitor screen in the background, not a live human present in the scene",
    "narrative": "第88帧人脸检测器触发，但视觉检查确认为背景显示器上的人物图像，非真实人员。步态和声音检测均未触发。综合判断：数据集通过质量审核。",
    "frames_reviewed": [86, 87, 88, 89, 90],
    "imu_windows_reviewed": []
  },
  "passed": true,
  "failure_reasons": []
}
```

### Error path — MinIO download failure

```json
{
  "report_id": "550e8400-e29b-41d4-a716-446655440001",
  "source_file": "robot-data/session_001.mcap",
  "minio_bucket": "robot-uploads",
  "analyzed_at": "2026-03-19T14:30:00Z",
  "duration_seconds": null,
  "scores": null,
  "sensitive_info": null,
  "llm_assessment": null,
  "llm_skipped_reason": null,
  "analyzer_errors": ["minio_download"],
  "passed": false,
  "failure_reasons": ["analyzer_error:minio_download"]
}
```

### Error path — corrupt MCAP

```json
{
  "report_id": "...",
  "source_file": "robot-data/bad_file.mcap",
  "minio_bucket": "robot-uploads",
  "analyzed_at": "2026-03-19T14:30:00Z",
  "duration_seconds": null,
  "scores": null,
  "sensitive_info": null,
  "llm_assessment": null,
  "llm_skipped_reason": null,
  "analyzer_errors": ["mcap_extraction"],
  "passed": false,
  "failure_reasons": ["analyzer_error:mcap_extraction"]
}
```

### Error path — single analyzer + LLM failure

```json
{
  "sensitive_info": {
    "has_face": false,
    "face_count": 0,
    "has_human_voice": false,
    "has_human_gait": null
  },
  "llm_assessment": null,
  "llm_skipped_reason": null,
  "analyzer_errors": ["gait", "llm"],
  "passed": false,
  "failure_reasons": ["analyzer_error:gait", "analyzer_error:llm"]
}
```

**LLM failure handling:** If the Anthropic API call fails (timeout, error), `llm_assessment` is set to `null`, `"llm"` is appended to `analyzer_errors`, and the detector-based verdict is used as fallback for `passed` and `failure_reasons`. The recording does not silently pass due to an LLM outage.

### Pass/Fail Rules

`passed` is determined by `llm_assessment.passed` if the LLM was invoked, otherwise by the detector-based rules below.

Detector-based rules (`passed = true` if ALL of):
- `duration_seconds >= MINIMUM_DURATION_SECONDS` (default: 1.0)
- `scores.clarity.score >= CLARITY_THRESHOLD`
- `scores.continuity.score >= CONTINUITY_THRESHOLD`
- `sensitive_info.has_face == false`
- `sensitive_info.has_human_voice == false`
- `sensitive_info.has_human_gait == false`
- `analyzer_errors` is empty

Any `null` field always counts as a failure. A `null` field **never** silently passes.

---

## Analyzer Interface

```python
# agent/analyzers/base.py

from typing import Protocol, TypedDict
import numpy as np


class ExtractedData(TypedDict):
    frames: list[np.ndarray]              # BGR frames HxWxC uint8; may be empty list
    audio_frames: list[bytes] | None      # 30ms PCM chunks, 16kHz mono int16; None if no audio topic
    sensor_series: dict[str, np.ndarray]  # topic_name → (T, D) float64 array; {} if no sensor topics
    duration_seconds: float               # total MCAP duration in seconds


class ClarityDetail(TypedDict):
    mean_laplacian_variance: float        # higher = sharper
    mean_tenengrad: float                 # higher = sharper
    frame_count: int                      # 0 when frames is empty


class ContinuityDetail(TypedDict):
    mean_flow_magnitude: float            # average optical flow magnitude across frame pairs
    discontinuity_frames: int             # frames where flow exceeds discontinuity threshold
    frame_count: int                      # 0 when frames is empty


class ClarityResult(TypedDict):
    score: float                          # [0.0, 1.0]
    method: str                           # "laplacian+tenengrad"
    detail: ClarityDetail


class ContinuityResult(TypedDict):
    score: float                          # [0.0, 1.0]
    method: str                           # "optical_flow"
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
        """Canonical key used by ReportBuilder.
        Required values: "clarity" | "continuity" | "face" | "voice" | "gait"
        """
        ...

    def analyze(self, data: ExtractedData) -> ClarityResult | ContinuityResult | FaceResult | VoiceResult | GaitResult:
        """Must handle empty frames list gracefully (score=0.0 or has_*=False).
        Must not raise — exceptions caught by AnalysisPipeline.
        """
        ...
```

**PCM framing contract:** `McapExtractor` chunks raw audio into 30ms frames (480 samples × 2 bytes = 960 bytes) before populating `audio_frames`. `VoiceDetector` passes these directly to `webrtcvad`.

**Empty frames / missing topics:** `frames=[]` → analyzers return `score=0.0` or `has_*=False` (valid results, not errors). `audio_frames=None` → `VoiceDetector` returns `has_human_voice=False`. Bags without camera topic always fail via low clarity/continuity scores — this is by design.

---

## Project Structure

```
data-quality-agent/
├── agent/
│   ├── __init__.py
│   ├── main.py              # FastAPI app: POST /notify, GET /health
│   ├── config.py            # pydantic-settings env config
│   ├── extractor.py         # McapExtractor: parses .mcap → ExtractedData (with PCM framing)
│   ├── pipeline.py          # AnalysisPipeline: ThreadPoolExecutor over Analyzer list
│   ├── llm_judge.py         # LLM Agent: tool definitions, invocation logic, override handling
│   ├── report.py            # ReportBuilder: merges detector + LLM results → JSON log
│   └── analyzers/
│       ├── __init__.py
│       ├── base.py          # Analyzer Protocol + all TypedDicts
│       ├── clarity.py       # ClarityAnalyzer (name: "clarity")
│       ├── continuity.py    # ContinuityAnalyzer (name: "continuity")
│       ├── face.py          # FaceDetector (name: "face", OpenCV DNN + YuNet)
│       ├── voice.py         # VoiceDetector (name: "voice", webrtcvad)
│       └── gait.py          # GaitDetector (name: "gait", OpenCV HOG)
├── tests/
│   ├── conftest.py          # Synthetic ExtractedData: 4×4 BGR frames, 30ms PCM bytes, sensor_series: {}
│   ├── test_extractor.py    # McapExtractor with minimal synthetic .mcap; PCM chunking
│   ├── test_pipeline.py     # Full detector pipeline; errors caught without aborting others
│   ├── test_llm_judge.py    # LLM invocation conditions; tool call mocking; override logic; API failure fallback
│   ├── test_report.py       # Pass/fail logic; null handling; report_id uniqueness; analyzed_at format
│   ├── test_main.py         # POST /notify → 200 immediately; GET /health → 200; 401 on bad token
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
        condition: service_healthy
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
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 5s
      timeout: 3s
      retries: 5
```

**Startup sequencing:** `minio-init` waits for both `minio` and `agent` to be healthy before registering events.

**ARN format:** `arn:minio:sqs::<alias>:webhook` where `<alias>` matches `MINIO_NOTIFY_WEBHOOK_ENABLE_<alias>`.

### Dockerfile (key lines)

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install uv && uv pip install --system .
COPY agent/ ./agent/
COPY models/ ./models/
CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MINIO_ENDPOINT` | `minio:9000` | MinIO host:port |
| `MINIO_ACCESS_KEY` | `minioadmin` | Access key |
| `MINIO_SECRET_KEY` | `minioadmin` | Secret key |
| `MINIO_BUCKET` | `robot-uploads` | Bucket to watch |
| `ANTHROPIC_API_KEY` | *(required)* | Claude API key |
| `CLARITY_THRESHOLD` | `0.6` | Min passing clarity score |
| `CONTINUITY_THRESHOLD` | `0.6` | Min passing continuity score |
| `LLM_REVIEW_MARGIN` | `0.1` | Score margin within threshold that triggers LLM review |
| `MINIMUM_DURATION_SECONDS` | `1.0` | Min MCAP duration |
| `WEBHOOK_AUTH_TOKEN` | `""` | If non-empty, validate Bearer token on /notify |
| `MODEL_DIR` | `/app/models` | Directory containing bundled model files |
| `LOG_LEVEL` | `INFO` | Log verbosity |

---

## Error Handling

| Failure | Behavior |
|---|---|
| `.mcap` file (not `.mcap` extension) | Webhook ignores event, returns 200, logs skip reason |
| Corrupt / unreadable MCAP | `McapExtractor` raises; error report emitted (`analyzer_errors: ["mcap_extraction"]`, `passed: false`) |
| MinIO download failure | Caught in `BackgroundTask`; error report with `analyzer_errors: ["minio_download"]` |
| Single detector raises | Caught by `AnalysisPipeline`; field → `null`; `analyzer_errors` updated; others continue |
| LLM API failure | `llm_assessment → null`; `"llm"` added to `analyzer_errors`; detector-based verdict used as fallback |
| Camera topic missing / empty frames | `frames=[]`; analyzers return `score=0.0` or `False`; valid result, not an error |
| Audio topic missing | `audio_frames=None`; `VoiceDetector` returns `False` |
| Invalid webhook auth token | `POST /notify` returns 401 immediately |

**Temp file cleanup:** `tempfile.TemporaryDirectory` as context manager; always cleaned up.

---

## Testing Strategy

| Test file | What it covers |
|---|---|
| `conftest.py` | Synthetic `ExtractedData` (4×4 BGR frames, 30ms PCM bytes, `sensor_series: {}`); shared fixtures |
| `test_main.py` | 200 on POST /notify; 200 on GET /health; 401 on bad token; non-mcap events ignored |
| `test_extractor.py` | McapExtractor with synthetic .mcap; PCM chunked to 30ms frames |
| `test_pipeline.py` | Full detector run; one analyzer error doesn't abort others |
| `test_llm_judge.py` | Invocation conditions (skip when clean); tool calls mocked; override logic; API failure → fallback to detector verdict |
| `test_report.py` | Pass/fail; null = failure; `report_id` is UUID4 and unique; `analyzed_at` is ISO 8601 UTC |
| `analyzers/test_clarity.py` | Sharp vs. blurry frames; empty frames → score=0.0 |
| `analyzers/test_continuity.py` | Smooth vs. jumpy sequence; empty frames → score=0.0 |
| `analyzers/test_face.py` | No-face frames → False; empty frames → False |
| `analyzers/test_voice.py` | Silent PCM → False; None → False |
| `analyzers/test_gait.py` | Empty frames → False |

No external services required. Anthropic API calls mocked in `test_llm_judge.py`.
