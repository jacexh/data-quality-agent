# Data Quality Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated MCAP data quality assessment service that triggers on MinIO uploads, runs algorithmic detectors, uses Claude as an LLM judge for ambiguous cases, and logs a structured JSON quality report.

**Architecture:** MinIO webhook triggers FastAPI; `McapExtractor` decodes `.mcap` files into frames/audio/sensor data; five algorithmic analyzers run concurrently via `ThreadPoolExecutor`; `LLMJudge` calls Claude claude-sonnet-4-6 with vision tools only when needed (ambiguous detections or borderline scores); `ReportBuilder` merges everything into a JSON log entry.

**Tech Stack:** Python 3.12, `mcap` + `mcap-ros2-support`, OpenCV, `webrtcvad`, `anthropic` SDK, FastAPI, loguru, pydantic-settings, uv, Docker + MinIO.

**Spec:** `docs/superpowers/specs/2026-03-19-data-quality-agent-design.md`

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Dependencies, project metadata |
| `agent/config.py` | Pydantic-settings env config — all thresholds and secrets |
| `agent/analyzers/base.py` | `Analyzer` Protocol + all TypedDicts (`ExtractedData`, `ClarityResult`, …) |
| `agent/extractor.py` | `McapExtractor` — reads `.mcap`, yields `ExtractedData` with PCM framing |
| `agent/analyzers/clarity.py` | `ClarityAnalyzer` — Laplacian + Tenengrad per frame |
| `agent/analyzers/continuity.py` | `ContinuityAnalyzer` — Farneback optical flow between frames |
| `agent/analyzers/face.py` | `FaceDetector` — OpenCV DNN + YuNet ONNX |
| `agent/analyzers/voice.py` | `VoiceDetector` — webrtcvad on pre-framed PCM |
| `agent/analyzers/gait.py` | `GaitDetector` — OpenCV HOG people detector |
| `agent/pipeline.py` | `AnalysisPipeline` — concurrent executor, error isolation |
| `agent/llm_judge.py` | `LLMJudge` — Claude tool_use loop, invocation gating, fallback |
| `agent/report.py` | `ReportBuilder` — merge detector + LLM results → final dict |
| `agent/main.py` | FastAPI app — `/notify`, `/health`, auth, `BackgroundTasks` |
| `tests/conftest.py` | Synthetic `ExtractedData` fixtures shared by all tests |
| `tests/test_extractor.py` | McapExtractor unit tests |
| `tests/test_pipeline.py` | AnalysisPipeline error isolation tests |
| `tests/test_llm_judge.py` | LLMJudge invocation conditions, tool mocking, fallback |
| `tests/test_report.py` | ReportBuilder pass/fail logic, null handling, UUID/ISO checks |
| `tests/test_main.py` | FastAPI endpoint tests |
| `tests/analyzers/test_clarity.py` | ClarityAnalyzer |
| `tests/analyzers/test_continuity.py` | ContinuityAnalyzer |
| `tests/analyzers/test_face.py` | FaceDetector |
| `tests/analyzers/test_voice.py` | VoiceDetector |
| `tests/analyzers/test_gait.py` | GaitDetector |
| `models/yunet.onnx` | YuNet face detection model (downloaded once) |
| `Dockerfile` | Container image |
| `docker-compose.yml` | MinIO + agent + init service |

---

## Task 1: Project Setup

**Files:**
- Create: `pyproject.toml`
- Create: `agent/__init__.py`, `agent/analyzers/__init__.py`
- Create: `tests/__init__.py`, `tests/analyzers/__init__.py`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p agent/analyzers tests/analyzers models
touch agent/__init__.py agent/analyzers/__init__.py
touch tests/__init__.py tests/analyzers/__init__.py
```

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[project]
name = "data-quality-agent"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115",
    "uvicorn>=0.30",
    "pydantic-settings>=2.0",
    "mcap>=1.1",
    "mcap-ros2-support>=0.5",
    "opencv-python-headless>=4.10",
    "numpy>=1.26",
    "webrtcvad>=2.0.10",
    "anthropic>=0.40",
    "loguru>=0.7",
    "python-multipart>=0.0.9",
    "boto3>=1.34",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "httpx>=0.27",
    "pytest-mock>=3.14",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 3: Install dependencies with uv**

```bash
pip install uv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Expected: all packages install without error.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml agent/ tests/ models/
git commit -m "chore: project structure and dependencies"
```

---

## Task 2: Base TypedDicts, Protocol, and Test Fixtures

**Files:**
- Create: `agent/analyzers/base.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `agent/analyzers/base.py`**

```python
from typing import Protocol, TypedDict
import numpy as np


class ExtractedData(TypedDict):
    frames: list[np.ndarray]              # BGR HxWxC uint8; may be []
    audio_frames: list[bytes] | None      # 30ms PCM chunks 16kHz mono int16; None if absent
    sensor_series: dict[str, np.ndarray]  # topic → (T,D) float64; {} if absent
    duration_seconds: float


class ClarityDetail(TypedDict):
    mean_laplacian_variance: float
    mean_tenengrad: float
    frame_count: int


class ContinuityDetail(TypedDict):
    mean_flow_magnitude: float
    discontinuity_frames: int
    frame_count: int


class ClarityResult(TypedDict):
    score: float       # [0.0, 1.0]
    method: str        # "laplacian+tenengrad"
    detail: ClarityDetail


class ContinuityResult(TypedDict):
    score: float       # [0.0, 1.0]
    method: str        # "optical_flow"
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
        """One of: "clarity" | "continuity" | "face" | "voice" | "gait" """
        ...

    def analyze(self, data: ExtractedData) -> ClarityResult | ContinuityResult | FaceResult | VoiceResult | GaitResult:
        """Must not raise. Handle empty frames gracefully."""
        ...
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
import numpy as np
import pytest
from agent.analyzers.base import ExtractedData


def _make_sharp_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Checkerboard pattern — high Laplacian variance."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[::4, :] = 255
    return frame


def _make_blurry_frame(h: int = 64, w: int = 64) -> np.ndarray:
    """Uniform grey — near-zero Laplacian variance."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _make_silent_pcm_frame() -> bytes:
    """960 bytes of zero PCM = 30ms silence at 16kHz mono int16."""
    return b"\x00" * 960


@pytest.fixture
def sharp_data() -> ExtractedData:
    return ExtractedData(
        frames=[_make_sharp_frame() for _ in range(10)],
        audio_frames=[_make_silent_pcm_frame() for _ in range(5)],
        sensor_series={},
        duration_seconds=5.0,
    )


@pytest.fixture
def blurry_data() -> ExtractedData:
    return ExtractedData(
        frames=[_make_blurry_frame() for _ in range(10)],
        audio_frames=None,
        sensor_series={},
        duration_seconds=5.0,
    )


@pytest.fixture
def empty_data() -> ExtractedData:
    return ExtractedData(
        frames=[],
        audio_frames=None,
        sensor_series={},
        duration_seconds=0.0,
    )
```

- [ ] **Step 3: Verify imports work**

```bash
python -c "from agent.analyzers.base import Analyzer, ExtractedData; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add agent/analyzers/base.py tests/conftest.py
git commit -m "feat: base TypedDicts, Analyzer Protocol, and test fixtures"
```

---

## Task 3: Config

**Files:**
- Create: `agent/config.py`

- [ ] **Step 1: Write `agent/config.py`**

```python
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
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

    class Config:
        env_file = ".env"


settings = Settings()
```

- [ ] **Step 2: Smoke-test**

```bash
python -c "from agent.config import settings; print(settings.clarity_threshold)"
```

Expected: `0.6`

- [ ] **Step 3: Commit**

```bash
git add agent/config.py
git commit -m "feat: pydantic-settings config"
```

---

## Task 4: McapExtractor

**Files:**
- Create: `agent/extractor.py`
- Create: `tests/test_extractor.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_extractor.py
import struct
import numpy as np
import pytest
from agent.extractor import McapExtractor


def test_empty_frames_on_missing_camera_topic(tmp_path):
    """An MCAP with no camera topic → frames=[]."""
    # Create a minimal valid MCAP with no messages
    # Use a real empty .mcap file from mcap library
    import mcap.writer as mw
    mcap_path = tmp_path / "empty.mcap"
    with open(mcap_path, "wb") as f:
        writer = mw.Writer(f)
        writer.start()
        writer.finish()

    extractor = McapExtractor(camera_topic="/camera/image_raw")
    data = extractor.extract(str(mcap_path))
    assert data["frames"] == []
    assert data["audio_frames"] is None
    assert data["duration_seconds"] == 0.0


def test_pcm_frames_are_960_bytes():
    """PCM frames must be exactly 960 bytes (30ms at 16kHz mono int16)."""
    # 480 samples × 2 bytes = 960
    assert 480 * 2 == 960


def test_pcm_chunking():
    """BagExtractor chunks raw PCM into 960-byte frames."""
    from agent.extractor import chunk_pcm
    raw = b"\x01\x02" * 480 * 3  # exactly 3 frames
    frames = chunk_pcm(raw)
    assert len(frames) == 3
    assert all(len(f) == 960 for f in frames)


def test_pcm_chunking_drops_remainder():
    """Incomplete trailing bytes are dropped."""
    from agent.extractor import chunk_pcm
    raw = b"\x00" * (960 * 2 + 100)  # 2 full frames + 100 leftover bytes
    frames = chunk_pcm(raw)
    assert len(frames) == 2
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_extractor.py -v
```

Expected: `ImportError` or `ModuleNotFoundError`.

- [ ] **Step 3: Write `agent/extractor.py`**

```python
from __future__ import annotations
import struct
import numpy as np
from mcap.reader import make_reader
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
    ) -> None:
        self._camera_topic = camera_topic
        self._audio_topic = audio_topic
        self._imu_topic = imu_topic

    def extract(self, mcap_path: str) -> ExtractedData:
        """Parse an MCAP file and return ExtractedData.

        Frames are decoded from sensor_msgs/Image messages.
        Audio is decoded from audio_common_msgs/AudioData messages and chunked to 30ms PCM frames.
        IMU is accumulated from sensor_msgs/Imu messages.
        """
        frames: list[np.ndarray] = []
        raw_audio = b""
        imu_rows: list[np.ndarray] = []
        timestamps: list[float] = []

        try:
            from mcap_ros2.reader import read_ros2_messages
        except ImportError:
            raise RuntimeError("mcap-ros2-support not installed")

        for msg in read_ros2_messages(mcap_path, topics=[
            self._camera_topic, self._audio_topic, self._imu_topic
        ]):
            t = msg.log_time / 1e9  # nanoseconds → seconds
            timestamps.append(t)
            topic = msg.channel.topic

            if topic == self._camera_topic:
                frame = self._decode_image(msg.ros_msg)
                if frame is not None:
                    frames.append(frame)

            elif topic == self._audio_topic:
                chunk = self._decode_audio(msg.ros_msg)
                if chunk:
                    raw_audio += chunk

            elif topic == self._imu_topic:
                row = self._decode_imu(msg.ros_msg)
                if row is not None:
                    imu_rows.append(row)

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

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_extractor.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/extractor.py tests/test_extractor.py
git commit -m "feat: McapExtractor with PCM framing"
```

---

## Task 5: ClarityAnalyzer

**Files:**
- Create: `agent/analyzers/clarity.py`
- Create: `tests/analyzers/test_clarity.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/analyzers/test_clarity.py
import numpy as np
import pytest
from agent.analyzers.clarity import ClarityAnalyzer


def test_sharp_frames_score_higher_than_blurry(sharp_data, blurry_data):
    analyzer = ClarityAnalyzer()
    sharp_result = analyzer.analyze(sharp_data)
    blurry_result = analyzer.analyze(blurry_data)
    assert sharp_result["score"] > blurry_result["score"]


def test_score_is_normalized(sharp_data):
    analyzer = ClarityAnalyzer()
    result = analyzer.analyze(sharp_data)
    assert 0.0 <= result["score"] <= 1.0


def test_empty_frames_returns_zero(empty_data):
    analyzer = ClarityAnalyzer()
    result = analyzer.analyze(empty_data)
    assert result["score"] == 0.0
    assert result["detail"]["frame_count"] == 0


def test_name():
    assert ClarityAnalyzer().name() == "clarity"


def test_method_field(sharp_data):
    result = ClarityAnalyzer().analyze(sharp_data)
    assert result["method"] == "laplacian+tenengrad"


def test_detail_keys(sharp_data):
    result = ClarityAnalyzer().analyze(sharp_data)
    d = result["detail"]
    assert "mean_laplacian_variance" in d
    assert "mean_tenengrad" in d
    assert "frame_count" in d
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/analyzers/test_clarity.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write `agent/analyzers/clarity.py`**

```python
import cv2
import numpy as np
from agent.analyzers.base import ExtractedData, ClarityResult, ClarityDetail

# Normalisation caps: values above these map to score=1.0
_LAP_CAP = 500.0
_TEN_CAP = 3000.0


class ClarityAnalyzer:
    def name(self) -> str:
        return "clarity"

    def analyze(self, data: ExtractedData) -> ClarityResult:
        frames = data["frames"]
        if not frames:
            return ClarityResult(
                score=0.0,
                method="laplacian+tenengrad",
                detail=ClarityDetail(mean_laplacian_variance=0.0, mean_tenengrad=0.0, frame_count=0),
            )

        lap_vars, tenegrads = [], []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_vars.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            tenegrads.append(float((sx**2 + sy**2).mean()))

        mean_lap = float(np.mean(lap_vars))
        mean_ten = float(np.mean(tenegrads))

        score_lap = min(mean_lap / _LAP_CAP, 1.0)
        score_ten = min(mean_ten / _TEN_CAP, 1.0)
        score = (score_lap + score_ten) / 2.0

        return ClarityResult(
            score=round(score, 4),
            method="laplacian+tenengrad",
            detail=ClarityDetail(
                mean_laplacian_variance=round(mean_lap, 4),
                mean_tenengrad=round(mean_ten, 4),
                frame_count=len(frames),
            ),
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/analyzers/test_clarity.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/analyzers/clarity.py tests/analyzers/test_clarity.py
git commit -m "feat: ClarityAnalyzer (Laplacian + Tenengrad)"
```

---

## Task 6: ContinuityAnalyzer

**Files:**
- Create: `agent/analyzers/continuity.py`
- Create: `tests/analyzers/test_continuity.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/analyzers/test_continuity.py
import numpy as np
import pytest
from agent.analyzers.continuity import ContinuityAnalyzer
from agent.analyzers.base import ExtractedData


def _jumpy_data() -> ExtractedData:
    """Alternates black and white frames — maximal optical flow."""
    frames = []
    for i in range(10):
        val = 255 if i % 2 == 0 else 0
        frames.append(np.full((64, 64, 3), val, dtype=np.uint8))
    return ExtractedData(frames=frames, audio_frames=None, sensor_series={}, duration_seconds=5.0)


def test_smooth_scores_higher_than_jumpy(sharp_data):
    """Identical frames → zero flow → high continuity score."""
    analyzer = ContinuityAnalyzer()
    smooth = analyzer.analyze(sharp_data)       # identical frames
    jumpy = analyzer.analyze(_jumpy_data())
    assert smooth["score"] > jumpy["score"]


def test_score_normalized(sharp_data):
    result = ContinuityAnalyzer().analyze(sharp_data)
    assert 0.0 <= result["score"] <= 1.0


def test_empty_frames_returns_zero(empty_data):
    result = ContinuityAnalyzer().analyze(empty_data)
    assert result["score"] == 0.0
    assert result["detail"]["frame_count"] == 0


def test_single_frame_returns_perfect(sharp_data):
    """Single frame → no pairs → nothing to be discontinuous → score 1.0."""
    data = ExtractedData(frames=[sharp_data["frames"][0]], audio_frames=None, sensor_series={}, duration_seconds=1.0)
    result = ContinuityAnalyzer().analyze(data)
    assert result["score"] == 1.0


def test_name():
    assert ContinuityAnalyzer().name() == "continuity"


def test_detail_keys(sharp_data):
    d = ContinuityAnalyzer().analyze(sharp_data)["detail"]
    assert "mean_flow_magnitude" in d
    assert "discontinuity_frames" in d
    assert "frame_count" in d
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/analyzers/test_continuity.py -v
```

- [ ] **Step 3: Write `agent/analyzers/continuity.py`**

```python
import cv2
import numpy as np
from agent.analyzers.base import ExtractedData, ContinuityResult, ContinuityDetail

_DISCONTINUITY_THRESHOLD = 15.0  # pixels/frame — above this counts as a jump


class ContinuityAnalyzer:
    def name(self) -> str:
        return "continuity"

    def analyze(self, data: ExtractedData) -> ContinuityResult:
        frames = data["frames"]
        if not frames:
            return ContinuityResult(
                score=0.0,
                method="optical_flow",
                detail=ContinuityDetail(mean_flow_magnitude=0.0, discontinuity_frames=0, frame_count=0),
            )
        if len(frames) == 1:
            return ContinuityResult(
                score=1.0,
                method="optical_flow",
                detail=ContinuityDetail(mean_flow_magnitude=0.0, discontinuity_frames=0, frame_count=1),
            )

        magnitudes, discontinuities = [], 0
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = float(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).mean())
            magnitudes.append(mag)
            if mag > _DISCONTINUITY_THRESHOLD:
                discontinuities += 1
            prev_gray = curr_gray

        mean_mag = float(np.mean(magnitudes))
        score = 1.0 - discontinuities / len(magnitudes)

        return ContinuityResult(
            score=round(score, 4),
            method="optical_flow",
            detail=ContinuityDetail(
                mean_flow_magnitude=round(mean_mag, 4),
                discontinuity_frames=discontinuities,
                frame_count=len(frames),
            ),
        )
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/analyzers/test_continuity.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/analyzers/continuity.py tests/analyzers/test_continuity.py
git commit -m "feat: ContinuityAnalyzer (Farneback optical flow)"
```

---

## Task 7: FaceDetector + Download YuNet Model

**Files:**
- Create: `agent/analyzers/face.py`
- Create: `tests/analyzers/test_face.py`
- Create: `models/yunet.onnx` (downloaded)

- [ ] **Step 1: Download YuNet model**

```bash
curl -L -o models/yunet.onnx \
  "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
```

Verify: `ls -lh models/yunet.onnx` → approximately 400 KB.

- [ ] **Step 2: Write failing tests**

```python
# tests/analyzers/test_face.py
import numpy as np
import pytest
from agent.analyzers.face import FaceDetector
from agent.analyzers.base import ExtractedData


def test_empty_frames_no_face(empty_data):
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(empty_data)
    assert result["has_face"] is False
    assert result["face_count"] == 0


def test_uniform_frame_no_face(blurry_data):
    """Uniform grey frame contains no face."""
    detector = FaceDetector(model_path="models/yunet.onnx")
    result = detector.analyze(blurry_data)
    assert result["has_face"] is False


def test_name():
    assert FaceDetector(model_path="models/yunet.onnx").name() == "face"
```

- [ ] **Step 3: Run to confirm failure**

```bash
pytest tests/analyzers/test_face.py -v
```

- [ ] **Step 4: Write `agent/analyzers/face.py`**

```python
import cv2
import numpy as np
from agent.analyzers.base import ExtractedData, FaceResult


class FaceDetector:
    def __init__(self, model_path: str = "models/yunet.onnx", conf_threshold: float = 0.6) -> None:
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._detector = None  # lazy init — avoid loading on import

    def _get_detector(self, width: int, height: int):
        detector = cv2.FaceDetectorYN.create(
            self._model_path,
            "",
            (width, height),
            score_threshold=self._conf_threshold,
            nms_threshold=0.3,
        )
        return detector

    def name(self) -> str:
        return "face"

    def analyze(self, data: ExtractedData) -> FaceResult:
        frames = data["frames"]
        if not frames:
            return FaceResult(has_face=False, face_count=0)

        max_faces = 0
        for frame in frames:
            h, w = frame.shape[:2]
            detector = self._get_detector(w, h)
            _, detections = detector.detect(frame)
            count = len(detections) if detections is not None else 0
            max_faces = max(max_faces, count)

        return FaceResult(has_face=max_faces > 0, face_count=max_faces)
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/analyzers/test_face.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add agent/analyzers/face.py tests/analyzers/test_face.py models/yunet.onnx
git commit -m "feat: FaceDetector (YuNet ONNX)"
```

---

## Task 8: VoiceDetector

**Files:**
- Create: `agent/analyzers/voice.py`
- Create: `tests/analyzers/test_voice.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/analyzers/test_voice.py
from agent.analyzers.voice import VoiceDetector
from agent.analyzers.base import ExtractedData


def test_silent_pcm_no_voice(sharp_data):
    """All-zero PCM frames are silence → no voice."""
    detector = VoiceDetector()
    result = detector.analyze(sharp_data)
    assert result["has_human_voice"] is False


def test_none_audio_no_voice(empty_data):
    detector = VoiceDetector()
    result = detector.analyze(empty_data)
    assert result["has_human_voice"] is False


def test_name():
    assert VoiceDetector().name() == "voice"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/analyzers/test_voice.py -v
```

- [ ] **Step 3: Write `agent/analyzers/voice.py`**

```python
import webrtcvad
from agent.analyzers.base import ExtractedData, VoiceResult

_SAMPLE_RATE = 16000
_VAD_MODE = 2  # aggressiveness 0-3; 2 = balanced


class VoiceDetector:
    def __init__(self, mode: int = _VAD_MODE) -> None:
        self._vad = webrtcvad.Vad(mode)

    def name(self) -> str:
        return "voice"

    def analyze(self, data: ExtractedData) -> VoiceResult:
        audio_frames = data["audio_frames"]
        if not audio_frames:
            return VoiceResult(has_human_voice=False)

        for frame in audio_frames:
            try:
                if self._vad.is_speech(frame, _SAMPLE_RATE):
                    return VoiceResult(has_human_voice=True)
            except Exception:
                continue

        return VoiceResult(has_human_voice=False)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/analyzers/test_voice.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/analyzers/voice.py tests/analyzers/test_voice.py
git commit -m "feat: VoiceDetector (webrtcvad)"
```

---

## Task 9: GaitDetector

**Files:**
- Create: `agent/analyzers/gait.py`
- Create: `tests/analyzers/test_gait.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/analyzers/test_gait.py
from agent.analyzers.gait import GaitDetector


def test_empty_frames_no_gait(empty_data):
    detector = GaitDetector()
    result = detector.analyze(empty_data)
    assert result["has_human_gait"] is False


def test_small_uniform_frames_no_gait(blurry_data):
    """64×64 uniform frames contain no walking person."""
    detector = GaitDetector()
    result = detector.analyze(blurry_data)
    assert result["has_human_gait"] is False


def test_name():
    assert GaitDetector().name() == "gait"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/analyzers/test_gait.py -v
```

- [ ] **Step 3: Write `agent/analyzers/gait.py`**

```python
import cv2
from agent.analyzers.base import ExtractedData, GaitResult


class GaitDetector:
    def __init__(self) -> None:
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def name(self) -> str:
        return "gait"

    def analyze(self, data: ExtractedData) -> GaitResult:
        frames = data["frames"]
        if not frames:
            return GaitResult(has_human_gait=False)

        for frame in frames:
            # HOG requires minimum size of ~64×128; skip tiny frames
            h, w = frame.shape[:2]
            if h < 128 or w < 64:
                continue
            rects, _ = self._hog.detectMultiScale(
                frame, winStride=(8, 8), padding=(4, 4), scale=1.05
            )
            if len(rects) > 0:
                return GaitResult(has_human_gait=True)

        return GaitResult(has_human_gait=False)
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/analyzers/test_gait.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/analyzers/gait.py tests/analyzers/test_gait.py
git commit -m "feat: GaitDetector (OpenCV HOG)"
```

---

## Task 10: AnalysisPipeline

**Files:**
- Create: `agent/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_pipeline.py
import pytest
from agent.pipeline import AnalysisPipeline
from agent.analyzers.base import ExtractedData


class _OkAnalyzer:
    def name(self) -> str:
        return "ok"
    def analyze(self, data):
        return {"value": 42}


class _BrokenAnalyzer:
    def name(self) -> str:
        return "broken"
    def analyze(self, data):
        raise RuntimeError("oops")


def test_all_analyzers_run(sharp_data):
    pipeline = AnalysisPipeline(analyzers=[_OkAnalyzer()])
    results, errors = pipeline.run(sharp_data)
    assert results["ok"] == {"value": 42}
    assert errors == []


def test_broken_analyzer_does_not_abort_others(sharp_data):
    pipeline = AnalysisPipeline(analyzers=[_OkAnalyzer(), _BrokenAnalyzer()])
    results, errors = pipeline.run(sharp_data)
    assert results["ok"] == {"value": 42}
    assert results["broken"] is None
    assert "broken" in errors


def test_all_broken_returns_all_none(sharp_data):
    pipeline = AnalysisPipeline(analyzers=[_BrokenAnalyzer()])
    results, errors = pipeline.run(sharp_data)
    assert results["broken"] is None
    assert errors == ["broken"]
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_pipeline.py -v
```

- [ ] **Step 3: Write `agent/pipeline.py`**

```python
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from agent.analyzers.base import Analyzer, ExtractedData


class AnalysisPipeline:
    def __init__(self, analyzers: list[Analyzer], max_workers: int = 5) -> None:
        self._analyzers = analyzers
        self._max_workers = max_workers

    def run(self, data: ExtractedData) -> tuple[dict, list[str]]:
        """Run all analyzers concurrently.

        Returns:
            results: dict mapping analyzer.name() → result dict (or None on error)
            errors: list of analyzer names that raised
        """
        results: dict = {}
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(a.analyze, data): a for a in self._analyzers}
            for future, analyzer in futures.items():
                name = analyzer.name()
                try:
                    results[name] = future.result()
                except Exception:
                    results[name] = None
                    errors.append(name)

        return results, errors
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_pipeline.py -v
```

Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/pipeline.py tests/test_pipeline.py
git commit -m "feat: AnalysisPipeline with concurrent execution and error isolation"
```

---

## Task 11: LLM Judge

**Files:**
- Create: `agent/llm_judge.py`
- Create: `tests/test_llm_judge.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_llm_judge.py
import base64
import pytest
from unittest.mock import MagicMock, patch
from agent.llm_judge import LLMJudge, should_invoke_llm
from agent.analyzers.base import ExtractedData
import numpy as np


def _make_detector_results(
    clarity_score=0.9, continuity_score=0.9,
    has_face=False, has_voice=False, has_gait=False
):
    return {
        "clarity": {"score": clarity_score, "method": "laplacian+tenengrad", "detail": {}},
        "continuity": {"score": continuity_score, "method": "optical_flow", "detail": {}},
        "face": {"has_face": has_face, "face_count": 1 if has_face else 0},
        "voice": {"has_human_voice": has_voice},
        "gait": {"has_human_gait": has_gait},
    }


def test_should_skip_when_all_clear():
    results = _make_detector_results()
    assert should_invoke_llm(results, clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1) is False


def test_should_invoke_when_face_detected():
    results = _make_detector_results(has_face=True)
    assert should_invoke_llm(results, clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1) is True


def test_should_invoke_when_score_borderline():
    results = _make_detector_results(clarity_score=0.62)  # within 0.1 of 0.6 threshold
    assert should_invoke_llm(results, clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1) is True


def test_should_invoke_on_cross_modal_ambiguity():
    """Voice detected but no face and no gait → ambiguous."""
    results = _make_detector_results(has_voice=True)
    assert should_invoke_llm(results, clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1) is True


def test_llm_failure_falls_back_to_detector_verdict(sharp_data):
    """If Anthropic API raises, LLMJudge returns None assessment and 'llm' error."""
    judge = LLMJudge(api_key="fake", model="claude-sonnet-4-6", clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1)
    results = _make_detector_results(has_face=True)

    with patch("agent.llm_judge.anthropic.Anthropic") as mock_client_cls:
        mock_client_cls.return_value.messages.create.side_effect = RuntimeError("API down")
        assessment, error = judge.judge(results, sharp_data)

    assert assessment is None
    assert error == "llm"


def test_llm_skipped_returns_none_assessment_and_no_error(sharp_data):
    judge = LLMJudge(api_key="fake", model="claude-sonnet-4-6", clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1)
    results = _make_detector_results()  # all clear → should skip
    assessment, error = judge.judge(results, sharp_data)
    assert assessment is None
    assert error is None
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_llm_judge.py -v
```

- [ ] **Step 3: Write `agent/llm_judge.py`**

```python
from __future__ import annotations
import base64
import json
import cv2
import numpy as np
import anthropic
from agent.analyzers.base import ExtractedData

_SYSTEM_PROMPT = """You are a data quality judge for robot-collected MCAP recordings.

You have access to algorithmic detector results and tools to inspect key frames and IMU data.

Your job:
1. Review any flagged sensitive information (face/voice/gait). Use get_key_frames to verify
   whether detections are genuine (live human) or false positives (poster, screen, background noise).
2. For borderline quality scores (within the review margin of threshold), review key frames and
   IMU context to determine if degradation is expected (e.g. motion blur during fast maneuver).
3. Produce a final verdict and a concise narrative (2-4 sentences in Chinese).

Rules:
- If a tool call fails, treat the original detector result as authoritative.
- Do not override clear failures (score < 0.3, unambiguous live face). Only reconsider ambiguous cases.
- Respond with valid JSON: {"passed": bool, "overrode_detector": bool, "override_detail": str | null, "narrative": str, "frames_reviewed": [int], "imu_windows_reviewed": [[float, float]]}
"""


def should_invoke_llm(
    detector_results: dict,
    clarity_threshold: float,
    continuity_threshold: float,
    margin: float,
) -> bool:
    """Return True if LLM review is warranted."""
    face = detector_results.get("face") or {}
    voice = detector_results.get("voice") or {}
    gait = detector_results.get("gait") or {}
    clarity = detector_results.get("clarity") or {}
    continuity = detector_results.get("continuity") or {}

    if face.get("has_face"):
        return True
    if gait.get("has_human_gait"):
        return True
    if voice.get("has_human_voice") and not face.get("has_face") and not gait.get("has_human_gait"):
        return True  # cross-modal ambiguity

    clarity_score = clarity.get("score")
    if clarity_score is not None and abs(clarity_score - clarity_threshold) <= margin:
        return True
    continuity_score = continuity.get("score")
    if continuity_score is not None and abs(continuity_score - continuity_threshold) <= margin:
        return True

    return False


class LLMJudge:
    def __init__(
        self,
        api_key: str,
        model: str,
        clarity_threshold: float,
        continuity_threshold: float,
        margin: float,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._clarity_threshold = clarity_threshold
        self._continuity_threshold = continuity_threshold
        self._margin = margin

    def judge(
        self,
        detector_results: dict,
        data: ExtractedData,
    ) -> tuple[dict | None, str | None]:
        """Run LLM judgment if warranted.

        Returns (assessment_dict, error_name):
          - (dict, None)  → LLM ran successfully
          - (None, None)  → LLM skipped (not warranted)
          - (None, "llm") → LLM failed; caller uses detector fallback
        """
        if not should_invoke_llm(
            detector_results, self._clarity_threshold, self._continuity_threshold, self._margin
        ):
            return None, None

        try:
            return self._run_agent(detector_results, data), None
        except Exception:
            return None, "llm"

    def _run_agent(self, detector_results: dict, data: ExtractedData) -> dict:
        client = anthropic.Anthropic(api_key=self._api_key)
        frames = data["frames"]
        imu = data["sensor_series"]

        tools = [
            {
                "name": "get_key_frames",
                "description": "Returns base64-encoded JPEG images for the specified frame indices.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "frame_indices": {"type": "array", "items": {"type": "integer"}}
                    },
                    "required": ["frame_indices"],
                },
            },
            {
                "name": "get_imu_summary",
                "description": "Returns IMU summary stats for a time window (seconds).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "window_start": {"type": "number"},
                        "window_end": {"type": "number"},
                    },
                    "required": ["window_start", "window_end"],
                },
            },
        ]

        user_message = (
            f"Detector results:\n{json.dumps(detector_results, ensure_ascii=False, indent=2)}\n\n"
            "Please review the flagged detections and/or borderline scores, "
            "then respond with your JSON verdict."
        )
        messages = [{"role": "user", "content": user_message}]

        for _ in range(5):  # max tool-use rounds
            response = client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )

            if response.stop_reason != "tool_use":
                # Extract text response
                text = next(
                    (b.text for b in response.content if hasattr(b, "text")), "{}"
                )
                return json.loads(text)

            # Process tool calls and append results
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_content = self._dispatch_tool(block.name, block.input, frames, imu)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                })

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError("LLM agent exceeded max tool-use rounds")

    def _dispatch_tool(self, name: str, inputs: dict, frames: list, imu: dict) -> str:
        if name == "get_key_frames":
            indices = inputs.get("frame_indices", [])
            result = []
            for idx in indices:
                if 0 <= idx < len(frames):
                    _, buf = cv2.imencode(".jpg", frames[idx], [cv2.IMWRITE_JPEG_QUALITY, 70])
                    result.append({
                        "frame_index": idx,
                        "image_b64": base64.b64encode(buf.tobytes()).decode(),
                    })
            return json.dumps(result)

        elif name == "get_imu_summary":
            # Simplified: return overall stats regardless of window
            for key, arr in imu.items():
                if arr.shape[1] >= 3:
                    acc = arr[:, :3]
                    return json.dumps({
                        "mean_acceleration": float(np.linalg.norm(acc, axis=1).mean()),
                        "max_angular_velocity": float(np.abs(arr[:, 3:6]).max()) if arr.shape[1] >= 6 else 0.0,
                        "mean_angular_velocity": float(np.abs(arr[:, 3:6]).mean()) if arr.shape[1] >= 6 else 0.0,
                    })
            return json.dumps({"mean_acceleration": 0.0, "max_angular_velocity": 0.0, "mean_angular_velocity": 0.0})

        return json.dumps({"error": f"unknown tool: {name}"})
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_llm_judge.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/llm_judge.py tests/test_llm_judge.py
git commit -m "feat: LLMJudge with tool_use loop, invocation gating, and API failure fallback"
```

---

## Task 12: ReportBuilder

**Files:**
- Create: `agent/report.py`
- Create: `tests/test_report.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_report.py
import uuid
from datetime import datetime, timezone
import pytest
from agent.report import ReportBuilder
from agent.config import Settings


def _settings(**kwargs):
    return Settings(
        clarity_threshold=kwargs.get("clarity_threshold", 0.6),
        continuity_threshold=kwargs.get("continuity_threshold", 0.6),
        minimum_duration_seconds=kwargs.get("minimum_duration_seconds", 1.0),
        anthropic_api_key="fake",
    )


def _good_results():
    return {
        "clarity": {"score": 0.9, "method": "laplacian+tenengrad", "detail": {}},
        "continuity": {"score": 0.9, "method": "optical_flow", "detail": {}},
        "face": {"has_face": False, "face_count": 0},
        "voice": {"has_human_voice": False},
        "gait": {"has_human_gait": False},
    }


def test_passed_true_on_clean_data():
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap", bucket="bucket",
        detector_results=_good_results(), detector_errors=[],
        llm_assessment=None, llm_error=None,
        duration_seconds=5.0,
    )
    assert report["passed"] is True
    assert report["failure_reasons"] == []


def test_failed_on_face():
    results = _good_results()
    results["face"] = {"has_face": True, "face_count": 1}
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap", bucket="bucket",
        detector_results=results, detector_errors=[],
        llm_assessment=None, llm_error=None,
        duration_seconds=5.0,
    )
    assert report["passed"] is False
    assert "has_face" in report["failure_reasons"]


def test_null_score_causes_failure():
    results = _good_results()
    results["clarity"] = None
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap", bucket="bucket",
        detector_results=results, detector_errors=["clarity"],
        llm_assessment=None, llm_error=None,
        duration_seconds=5.0,
    )
    assert report["passed"] is False
    assert "analyzer_error:clarity" in report["failure_reasons"]


def test_report_id_is_valid_uuid4():
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap", bucket="bucket",
        detector_results=_good_results(), detector_errors=[],
        llm_assessment=None, llm_error=None, duration_seconds=5.0,
    )
    parsed = uuid.UUID(report["report_id"])
    assert parsed.version == 4


def test_report_id_is_unique():
    builder = ReportBuilder(_settings())
    kwargs = dict(
        source_file="test.mcap", bucket="b",
        detector_results=_good_results(), detector_errors=[],
        llm_assessment=None, llm_error=None, duration_seconds=5.0,
    )
    r1 = builder.build(**kwargs)
    r2 = builder.build(**kwargs)
    assert r1["report_id"] != r2["report_id"]


def test_analyzed_at_is_iso8601_utc():
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap", bucket="b",
        detector_results=_good_results(), detector_errors=[],
        llm_assessment=None, llm_error=None, duration_seconds=5.0,
    )
    dt = datetime.fromisoformat(report["analyzed_at"].replace("Z", "+00:00"))
    assert dt.tzinfo is not None


def test_llm_assessment_overrides_verdict():
    results = _good_results()
    results["face"] = {"has_face": True, "face_count": 1}
    llm = {"passed": True, "overrode_detector": True, "override_detail": "face is on screen", "narrative": "ok", "frames_reviewed": [], "imu_windows_reviewed": []}
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap", bucket="b",
        detector_results=results, detector_errors=[],
        llm_assessment=llm, llm_error=None,
        duration_seconds=5.0,
    )
    assert report["passed"] is True
    assert report["failure_reasons"] == []


def test_short_duration_fails():
    builder = ReportBuilder(_settings(minimum_duration_seconds=2.0))
    report = builder.build(
        source_file="test.mcap", bucket="b",
        detector_results=_good_results(), detector_errors=[],
        llm_assessment=None, llm_error=None,
        duration_seconds=0.5,
    )
    assert report["passed"] is False
    assert "duration_too_short" in report["failure_reasons"]
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_report.py -v
```

- [ ] **Step 3: Write `agent/report.py`**

```python
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from agent.config import Settings


class ReportBuilder:
    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def build(
        self,
        source_file: str,
        bucket: str,
        detector_results: dict,
        detector_errors: list[str],
        llm_assessment: dict | None,
        llm_error: str | None,
        duration_seconds: float | None,
    ) -> dict:
        analyzer_errors = list(detector_errors)
        if llm_error:
            analyzer_errors.append(llm_error)

        failure_reasons: list[str] = []

        # Duration check
        if duration_seconds is None or duration_seconds < self._s.minimum_duration_seconds:
            failure_reasons.append("duration_too_short")

        # Build scores section
        clarity = detector_results.get("clarity")
        continuity = detector_results.get("continuity")
        scores = None
        if clarity is not None or continuity is not None:
            scores = {}
            if clarity is not None:
                scores["clarity"] = clarity
            if continuity is not None:
                scores["continuity"] = continuity

        # Build sensitive_info section
        face = detector_results.get("face")
        voice = detector_results.get("voice")
        gait = detector_results.get("gait")
        sensitive_info = None
        if any(x is not None for x in [face, voice, gait]):
            sensitive_info = {
                "has_face": face["has_face"] if face else None,
                "face_count": face["face_count"] if face else None,
                "has_human_voice": voice["has_human_voice"] if voice else None,
                "has_human_gait": gait["has_human_gait"] if gait else None,
            }

        # Collect detector-based failure reasons
        for name in detector_errors:
            failure_reasons.append(f"analyzer_error:{name}")

        if clarity is None and "clarity" not in detector_errors:
            pass  # not run
        elif clarity is not None:
            if clarity["score"] < self._s.clarity_threshold:
                failure_reasons.append("clarity")

        if continuity is not None:
            if continuity["score"] < self._s.continuity_threshold:
                failure_reasons.append("continuity")

        if face is not None and face["has_face"]:
            failure_reasons.append("has_face")
        elif face is None and "face" not in detector_errors:
            pass
        elif face is None:
            pass  # already in analyzer_errors

        if voice is not None and voice["has_human_voice"]:
            failure_reasons.append("has_human_voice")

        if gait is not None and gait["has_human_gait"]:
            failure_reasons.append("has_human_gait")

        # LLM overrides verdict if it ran successfully
        if llm_assessment is not None:
            passed = llm_assessment["passed"]
            failure_reasons = [] if passed else failure_reasons
        else:
            passed = len(failure_reasons) == 0 and not analyzer_errors

        llm_skipped_reason = None
        if llm_assessment is None and llm_error is None:
            llm_skipped_reason = "all_detectors_clear_no_borderline_scores"

        return {
            "report_id": str(uuid.uuid4()),
            "source_file": source_file,
            "minio_bucket": bucket,
            "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_seconds": duration_seconds,
            "scores": scores,
            "sensitive_info": sensitive_info,
            "llm_assessment": llm_assessment,
            "llm_skipped_reason": llm_skipped_reason,
            "analyzer_errors": analyzer_errors,
            "passed": passed,
            "failure_reasons": sorted(set(failure_reasons)),
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_report.py -v
```

Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add agent/report.py tests/test_report.py
git commit -m "feat: ReportBuilder with pass/fail logic and LLM override support"
```

---

## Task 13: FastAPI App

**Files:**
- Create: `agent/main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_main.py
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from agent.main import app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_health_returns_200(client):
    response = await client.get("/health")
    assert response.status_code == 200


async def test_notify_returns_200_immediately(client):
    payload = {
        "EventName": "s3:ObjectCreated:Put",
        "Key": "robot-uploads/session.mcap",
        "Records": [{"s3": {"bucket": {"name": "robot-uploads"}, "object": {"key": "session.mcap"}}}],
    }
    with patch("agent.main._process_mcap") as mock_process:
        response = await client.post("/notify", json=payload)
    assert response.status_code == 200


async def test_notify_ignores_non_mcap(client):
    payload = {
        "EventName": "s3:ObjectCreated:Put",
        "Key": "robot-uploads/session.bag",
        "Records": [{"s3": {"bucket": {"name": "robot-uploads"}, "object": {"key": "session.bag"}}}],
    }
    with patch("agent.main._process_mcap") as mock_process:
        response = await client.post("/notify", json=payload)
    assert response.status_code == 200
    mock_process.assert_not_called()


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
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_main.py -v
```

- [ ] **Step 3: Write `agent/main.py`**

```python
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
_pipeline = AnalysisPipeline(analyzers=[
    ClarityAnalyzer(),
    ContinuityAnalyzer(),
    FaceDetector(model_path=os.path.join(settings.model_dir, "yunet.onnx")),
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
    report_base = {
        "source_file": source_file,
        "minio_bucket": bucket,
    }

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
```

- [ ] **Step 4: Verify boto3 is already in dependencies**

`boto3` was added to `pyproject.toml` in Task 1. Confirm it is present:

```bash
grep boto3 pyproject.toml
```

Expected: `"boto3>=1.34"` appears.

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_main.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add agent/main.py tests/test_main.py pyproject.toml
git commit -m "feat: FastAPI app with /notify, /health, auth, and BackgroundTasks"
```

---

## Task 14: Run Full Test Suite

- [ ] **Step 1: Run all tests**

```bash
pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Fix any failures**

If any test fails, read the error and fix the root cause. Do not skip or mock away real failures.

- [ ] **Step 3: Commit any fixes**

```bash
git add -p
git commit -m "fix: <describe what was wrong>"
```

---

## Task 15: Dockerfile and docker-compose

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.env.example`

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml .
RUN pip install uv && uv pip install --system .
COPY agent/ ./agent/
COPY models/ ./models/
CMD ["uvicorn", "agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write `docker-compose.yml`**

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
      agent:
        condition: service_healthy
    entrypoint: >
      /bin/sh -c "
        mc alias set myminio http://minio:9000 minioadmin minioadmin &&
        mc mb --ignore-existing myminio/robot-uploads &&
        mc event add myminio/robot-uploads arn:minio:sqs::agent:webhook --event s3:ObjectCreated:* &&
        echo 'MinIO setup complete'
      "

  agent:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      MINIO_ENDPOINT: minio:9000
      MINIO_BUCKET: robot-uploads
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 5s
      timeout: 3s
      retries: 5
```

- [ ] **Step 3: Write `.env.example`**

```bash
ANTHROPIC_API_KEY=your-key-here
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
CLARITY_THRESHOLD=0.6
CONTINUITY_THRESHOLD=0.6
LLM_REVIEW_MARGIN=0.1
MINIMUM_DURATION_SECONDS=1.0
LOG_LEVEL=INFO
```

- [ ] **Step 4: Verify Docker build**

```bash
cp .env.example .env
# Fill in ANTHROPIC_API_KEY in .env
docker build -t data-quality-agent .
```

Expected: image builds without error.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml .env.example
git commit -m "chore: Dockerfile and docker-compose with MinIO webhook setup"
```

---

## Task 16: Smoke Test End-to-End

- [ ] **Step 1: Start the stack**

```bash
docker compose up --build -d
docker compose logs -f minio-init
```

Expected: `MinIO setup complete` in logs.

- [ ] **Step 2: Create a minimal synthetic MCAP and upload it**

```python
# create_test_mcap.py — run once to generate a test file
from mcap.writer import Writer
with open("/tmp/test.mcap", "wb") as f:
    writer = Writer(f)
    writer.start(profile="", library="test")
    schema_id = writer.register_schema(name="std_msgs/String", encoding="ros2msg", data=b"string data\n")
    chan_id = writer.register_channel(topic="/test", message_encoding="cdr", schema_id=schema_id)
    import time; t = int(time.time() * 1e9)
    writer.add_message(channel_id=chan_id, log_time=t, data=b"\x00\x00\x00\x00\x05\x00\x00\x00hello", publish_time=t)
    writer.finish()
print("Created /tmp/test.mcap")
```

```bash
python create_test_mcap.py
docker compose cp /tmp/test.mcap minio:/tmp/test.mcap
docker compose exec minio mc cp /tmp/test.mcap myminio/robot-uploads/test.mcap
```

- [ ] **Step 3: Watch agent logs**

```bash
docker compose logs -f agent
```

Expected: a JSON quality report log line appears within a few seconds of upload.

- [ ] **Step 4: Final commit**

```bash
git add .
git commit -m "chore: end-to-end smoke test verified"
```
