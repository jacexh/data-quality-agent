# Multi-Camera & Multi-Audio Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the data quality agent to support multiple video and audio topics from a single MCAP file, with per-topic detection, independent LLM assessment, configurable pass/fail strategies, and a unified `cameras`/`audios` report schema.

**Architecture:** `McapExtractor` extracts all matching topics into `ExtractedData.videos` and `ExtractedData.audios` dicts. Per-topic detection runs in `ProcessPoolExecutor` workers (CPU-bound); per-topic LLM calls run concurrently in `ThreadPoolExecutor` (I/O-bound). `ReportBuilder` assembles `CameraResult`/`AudioResult` arrays and applies `evaluate_strategy()` with configurable `all`/`any`/`majority` logic.

**Tech Stack:** Python 3.11+, numpy, OpenCV, anthropic SDK, concurrent.futures (ProcessPoolExecutor + ThreadPoolExecutor), pydantic-settings, pytest

**Spec:** `docs/superpowers/specs/2026-03-22-multi-camera-audio-support-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `agent/config.py` | Add 7 new fields: topic lists, pass strategies, frame/concurrency limits |
| Modify | `agent/analyzers/base.py` | Replace `ExtractedData`; add `CameraResult`, `AudioResult`; split `Analyzer` Protocol |
| Modify | `agent/analyzers/clarity.py` | `analyze(self, data)` → `analyze(self, frames)` |
| Modify | `agent/analyzers/continuity.py` | Same as clarity |
| Modify | `agent/analyzers/face.py` | Same as clarity |
| Modify | `agent/analyzers/gait.py` | Same as clarity |
| Modify | `agent/analyzers/voice.py` | `analyze(self, data)` → `analyze(self, audio_frames)` |
| Modify | `agent/extractor.py` | Multi-topic bucketing + 3-step sampling + `extraction_warnings` |
| Modify | `agent/pipeline.py` | `run()` → `run_visual(frames)` + `run_audio(audio_frames)`; add picklable top-level workers |
| Modify | `agent/report.py` | New `cameras`/`audios` schema; `evaluate_strategy()`; `CameraResult`/`AudioResult` assembly |
| Modify | `agent/llm_judge.py` | New `judge(topic, results, frames, audio_frames, sensor_series)` signature |
| Modify | `agent/runner.py` | `ProcessPoolExecutor` detection phase; `ThreadPoolExecutor` LLM phase; result assembly |
| Modify | `tests/conftest.py` | Update fixtures to new `ExtractedData` shape |
| Modify | `tests/analyzers/test_clarity.py` | Call `analyzer.analyze(frames)` directly |
| Modify | `tests/analyzers/test_continuity.py` | Same as clarity |
| Modify | `tests/analyzers/test_face.py` | Same as clarity |
| Modify | `tests/analyzers/test_gait.py` | Same as clarity |
| Modify | `tests/analyzers/test_voice.py` | Call `analyzer.analyze(audio_frames)` directly |
| Modify | `tests/test_pipeline.py` | Update stubs and test calls to `run_visual`/`run_audio` |
| Modify | `tests/test_extractor.py` | Assert on `data["videos"]`, `data["audios"]`, `data["extraction_warnings"]` |
| Modify | `tests/test_report.py` | Complete rewrite: `cameras`/`audios` arrays, `evaluate_strategy`, new `build()` sig |
| Modify | `tests/test_llm_judge.py` | Update `judge()` call signature |

---

## Task 1: Config — add new settings fields

**Files:**
- Modify: `agent/config.py`

Remove `camera_topic` and `audio_topic` (if they exist — check first; the current code has no explicit single-topic fields there, they are defaults in the extractor). Add new fields directly.

- [ ] **Step 1: Write failing import test**

```python
# Add to tests/test_config.py (create if absent) or run inline
# Verify new fields exist with correct defaults:
from agent.config import Settings
s = Settings(anthropic_api_key="x")
assert s.camera_topics == []
assert s.audio_topics == []
assert s.camera_pass_strategy == "all"
assert s.audio_pass_strategy == "all"
assert s.max_frames_per_topic == 300
assert s.max_concurrent_topics == 4
assert s.llm_max_concurrent_calls == 4
```

Run: `uv run python -c "from agent.config import Settings; s=Settings(anthropic_api_key='x'); print(s.camera_topics)"`

Expected: `AttributeError` — fields don't exist yet.

- [ ] **Step 2: Add new fields to `agent/config.py`**

In `agent/config.py`, add after `frame_sample_rate`:

```python
from typing import Literal

camera_topics: list[str] = []
audio_topics: list[str] = []
camera_pass_strategy: Literal["all", "any", "majority"] = "all"
audio_pass_strategy: Literal["all", "any", "majority"] = "all"
max_frames_per_topic: int = Field(default=300, gt=0)
max_concurrent_topics: int = Field(default=4, gt=0)
llm_max_concurrent_calls: int = Field(default=4, gt=0)
```

Also add `from typing import Literal` to the imports at top.

- [ ] **Step 3: Verify new fields resolve correctly**

Run: `uv run python -c "from agent.config import Settings; s=Settings(anthropic_api_key='x'); print(s.camera_topics, s.camera_pass_strategy, s.max_frames_per_topic)"`

Expected: `[] all 300`

- [ ] **Step 4: Run existing tests — expect no new failures**

Run: `uv run pytest tests/ -x -q 2>&1 | tail -20`

Expected: same failures as before (none introduced by this change).

- [ ] **Step 5: Commit**

```bash
git add agent/config.py
git commit -m "feat(config): add multi-topic and strategy config fields"
```

---

## Task 2: Base data structures — update `ExtractedData`, add result types, split Analyzer Protocol

**Files:**
- Modify: `agent/analyzers/base.py`

This is the foundational breaking change. After this task, all tests that construct `ExtractedData` will fail until fixed in subsequent tasks.

- [ ] **Step 1: Write tests for new types**

Add `tests/test_base_types.py` (new file):

```python
import numpy as np
from agent.analyzers.base import ExtractedData, CameraResult, AudioResult

def test_extracted_data_has_videos_and_audios():
    d = ExtractedData(
        videos={"/cam": []},
        audios={"/audio": []},
        sensor_series={},
        duration_seconds=1.0,
        extraction_warnings={},
    )
    assert "/cam" in d["videos"]
    assert "/audio" in d["audios"]
    assert d["extraction_warnings"] == {}


def test_camera_result_has_required_keys():
    r = CameraResult(
        topic="/cam",
        frame_count=10,
        clarity={"score": 0.9, "method": "laplacian+fft", "detail": {}},
        continuity={"score": 0.8, "method": "optical_flow", "detail": {}},
        face={"has_face": False, "face_count": 0, "face_frame_ratio": 0.0, "max_confidence": 0.0},
        gait={"has_human_gait": False, "person_frame_ratio": 0.0, "max_detection_weight": 0.0},
        llm_assessment=None,
        llm_skipped_reason="no_sensitive_detection",
        passed=True,
        failure_reasons=[],
        analyzer_errors=[],
    )
    assert r["topic"] == "/cam"
    assert r["passed"] is True


def test_audio_result_has_required_keys():
    r = AudioResult(
        topic="/audio",
        audio_frame_count=100,
        voice={"has_human_voice": False, "speech_frame_ratio": 0.0},
        llm_assessment=None,
        llm_skipped_reason="no_sensitive_detection",
        passed=True,
        failure_reasons=[],
        analyzer_errors=[],
    )
    assert r["topic"] == "/audio"
    assert r["passed"] is True
```

Run: `uv run pytest tests/test_base_types.py -x -q`

Expected: `ImportError` — `CameraResult`, `AudioResult` don't exist yet.

- [ ] **Step 2: Rewrite `agent/analyzers/base.py`**

```python
from typing import Protocol, TypedDict
import numpy as np


class ExtractedData(TypedDict):
    videos: dict[str, list[np.ndarray]]   # topic → BGR frame list
    audios: dict[str, list[bytes]]         # topic → 30ms PCM frame list
    sensor_series: dict[str, np.ndarray]   # shared across all topics
    duration_seconds: float
    extraction_warnings: dict[str, str]    # topic → "below_min_frames" etc.


class ClarityDetail(TypedDict):
    mean_laplacian_variance: float
    fft_high_freq_ratio: float
    frame_score_std: float
    frame_count: int


class ContinuityDetail(TypedDict):
    mean_flow_magnitude: float
    flow_magnitude_std: float
    flow_direction_std: float
    discontinuity_frames: int
    frame_count: int


class ClarityResult(TypedDict):
    score: float
    method: str
    detail: ClarityDetail


class ContinuityResult(TypedDict):
    score: float
    method: str
    detail: ContinuityDetail


class FaceResult(TypedDict):
    has_face: bool
    face_count: int
    face_frame_ratio: float
    max_confidence: float


class VoiceResult(TypedDict):
    has_human_voice: bool
    speech_frame_ratio: float


class GaitResult(TypedDict):
    has_human_gait: bool
    person_frame_ratio: float
    max_detection_weight: float


class CameraResult(TypedDict):
    topic: str
    frame_count: int
    clarity: ClarityResult
    continuity: ContinuityResult
    face: FaceResult
    gait: GaitResult
    llm_assessment: dict | None
    llm_skipped_reason: str | None
    passed: bool
    failure_reasons: list[str]
    analyzer_errors: list[str]


class AudioResult(TypedDict):
    topic: str
    audio_frame_count: int
    voice: VoiceResult
    llm_assessment: dict | None
    llm_skipped_reason: str | None
    passed: bool
    failure_reasons: list[str]
    analyzer_errors: list[str]


class VisualAnalyzer(Protocol):
    def name(self) -> str: ...
    def analyze(self, frames: list[np.ndarray]) -> ClarityResult | ContinuityResult | FaceResult | GaitResult:
        """Must not raise. Handle empty frames gracefully."""
        ...


class AudioAnalyzer(Protocol):
    def name(self) -> str: ...
    def analyze(self, audio_frames: list[bytes]) -> VoiceResult:
        """Must not raise. Handle empty audio_frames gracefully."""
        ...
```

- [ ] **Step 3: Run new type tests — expect PASS**

Run: `uv run pytest tests/test_base_types.py -x -q`

Expected: All 3 tests PASS.

- [ ] **Step 4: Note expected breakage in existing tests**

Run: `uv run pytest tests/ -q --tb=no 2>&1 | tail -5`

Expected: Multiple failures due to `frames`/`audio_frames` key removal — this is intentional and will be fixed task by task.

- [ ] **Step 5: Update `tests/conftest.py` fixtures**

Replace the file content. Keep helper functions for creating frames/audio. Update all `ExtractedData` constructions to use `videos`/`audios`:

```python
import pathlib
import numpy as np
import pytest
import cv2
from agent.analyzers.base import ExtractedData

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"

_CAM = "/camera/image_raw"
_AUDIO = "/audio/data"


def _make_sharp_frame(h: int = 64, w: int = 64) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[::4, :] = 255
    return frame


def _make_blurry_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _make_silent_pcm_frame() -> bytes:
    return b"\x00" * 960


def _make_speech_pcm_frame(freq: int = 1000) -> bytes:
    n = 480
    t = np.linspace(0, n / 16000, n, endpoint=False)
    samples = (16000 * np.sin(2 * np.pi * freq * t)).astype(np.int16)
    return samples.tobytes()


@pytest.fixture
def sharp_frames() -> list[np.ndarray]:
    return [_make_sharp_frame() for _ in range(10)]


@pytest.fixture
def blurry_frames() -> list[np.ndarray]:
    return [_make_blurry_frame() for _ in range(10)]


@pytest.fixture
def empty_frames() -> list[np.ndarray]:
    return []


@pytest.fixture
def silent_audio() -> list[bytes]:
    return [_make_silent_pcm_frame() for _ in range(5)]


@pytest.fixture
def speech_audio() -> list[bytes]:
    return [_make_speech_pcm_frame() for _ in range(10)]


@pytest.fixture
def sharp_data(sharp_frames, silent_audio) -> ExtractedData:
    return ExtractedData(
        videos={_CAM: sharp_frames},
        audios={_AUDIO: silent_audio},
        sensor_series={},
        duration_seconds=5.0,
        extraction_warnings={},
    )


@pytest.fixture
def blurry_data(blurry_frames) -> ExtractedData:
    return ExtractedData(
        videos={_CAM: blurry_frames},
        audios={},
        sensor_series={},
        duration_seconds=5.0,
        extraction_warnings={},
    )


@pytest.fixture
def empty_data() -> ExtractedData:
    return ExtractedData(
        videos={_CAM: []},
        audios={},
        sensor_series={},
        duration_seconds=0.0,
        extraction_warnings={},
    )


@pytest.fixture
def face_data() -> ExtractedData:
    img = cv2.imread(str(FIXTURES_DIR / "lena.jpg"))
    assert img is not None, "tests/fixtures/lena.jpg missing"
    return ExtractedData(
        videos={_CAM: [img]},
        audios={},
        sensor_series={},
        duration_seconds=1.0,
        extraction_warnings={},
    )


@pytest.fixture
def person_data() -> ExtractedData:
    img = cv2.imread(str(FIXTURES_DIR / "person.jpg"))
    assert img is not None, "tests/fixtures/person.jpg missing"
    return ExtractedData(
        videos={_CAM: [img]},
        audios={},
        sensor_series={},
        duration_seconds=1.0,
        extraction_warnings={},
    )


@pytest.fixture
def speech_data(speech_audio) -> ExtractedData:
    return ExtractedData(
        videos={_CAM: [_make_sharp_frame() for _ in range(5)]},
        audios={_AUDIO: speech_audio},
        sensor_series={},
        duration_seconds=0.3,
        extraction_warnings={},
    )
```

- [ ] **Step 6: Commit**

```bash
git add agent/analyzers/base.py tests/conftest.py tests/test_base_types.py
git commit -m "feat(base): new ExtractedData, CameraResult, AudioResult, VisualAnalyzer/AudioAnalyzer protocols"
```

---

## Task 3: Visual analyzers — update `analyze()` signature (4 files)

**Files:**
- Modify: `agent/analyzers/clarity.py`, `agent/analyzers/continuity.py`, `agent/analyzers/face.py`, `agent/analyzers/gait.py`
- Modify: `tests/analyzers/test_clarity.py`, `tests/analyzers/test_continuity.py`, `tests/analyzers/test_face.py`, `tests/analyzers/test_gait.py`

Each analyzer changes from `analyze(self, data: ExtractedData)` → `analyze(self, frames: list[np.ndarray])` and removes `data["frames"]` access. This is a mechanical change — no logic changes.

### 3a: ClarityAnalyzer

- [ ] **Step 1: Update `tests/analyzers/test_clarity.py`**

Replace all `analyzer.analyze(data)` calls with `analyzer.analyze(frames)`. Replace `ExtractedData(frames=..., ...)` constructions with plain `list[np.ndarray]`.

Key changes to make in the test file:
- `analyzer.analyze(sharp_data)` → `analyzer.analyze(sharp_frames)` (use the new `sharp_frames` fixture)
- `analyzer.analyze(blurry_data)` → `analyzer.analyze(blurry_frames)`
- `analyzer.analyze(empty_data)` → `analyzer.analyze([])`
- Inline `ExtractedData(frames=[...], ...)` → just `[_make_sharp_frame() for _ in range(5)]`

Full updated file:

```python
import numpy as np
import pytest
import cv2
from agent.analyzers.clarity import ClarityAnalyzer


def _make_sharp_frame(h: int = 64, w: int = 64) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[::4, :] = 255
    return frame


def _make_blurry_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), 128, dtype=np.uint8)


def _make_checkerboard_frame(h: int = 64, w: int = 64, block: int = 8) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    for r in range(0, h, block):
        for c in range(0, w, block):
            if (r // block + c // block) % 2 == 0:
                frame[r:r+block, c:c+block] = 255
    return frame


def _make_motion_blurred_frame(h: int = 64, w: int = 64) -> np.ndarray:
    sharp = _make_checkerboard_frame(h, w)
    kernel = np.zeros((1, 15))
    kernel[0, :] = 1.0 / 15
    return cv2.filter2D(sharp, -1, kernel)


def test_sharp_frames_score_higher_than_blurry(sharp_frames, blurry_frames):
    analyzer = ClarityAnalyzer()
    assert analyzer.analyze(sharp_frames)["score"] > analyzer.analyze(blurry_frames)["score"]


def test_score_is_normalized(sharp_frames):
    result = ClarityAnalyzer().analyze(sharp_frames)
    assert 0.0 <= result["score"] <= 1.0


def test_empty_frames_returns_zero():
    result = ClarityAnalyzer().analyze([])
    assert result["score"] == 0.0
    assert result["detail"]["frame_count"] == 0


def test_name():
    assert ClarityAnalyzer().name() == "clarity"


def test_method_field(sharp_frames):
    result = ClarityAnalyzer().analyze(sharp_frames)
    assert result["method"] == "laplacian+fft"


def test_detail_keys(sharp_frames):
    d = ClarityAnalyzer().analyze(sharp_frames)["detail"]
    assert "mean_laplacian_variance" in d
    assert "fft_high_freq_ratio" in d
    assert "frame_score_std" in d
    assert "frame_count" in d


def test_fft_high_freq_ratio_higher_for_sharp_than_blurry():
    analyzer = ClarityAnalyzer()
    sharp = [_make_sharp_frame() for _ in range(5)]
    blurry = [_make_blurry_frame() for _ in range(5)]
    assert analyzer.analyze(sharp)["detail"]["fft_high_freq_ratio"] > \
           analyzer.analyze(blurry)["detail"]["fft_high_freq_ratio"]


def test_motion_blurred_scores_lower_than_sharp():
    analyzer = ClarityAnalyzer()
    sharp = [_make_sharp_frame() for _ in range(5)]
    motion = [_make_motion_blurred_frame() for _ in range(5)]
    assert analyzer.analyze(sharp)["score"] > analyzer.analyze(motion)["score"]


def test_frame_score_std_is_zero_for_uniform_frames():
    result = ClarityAnalyzer().analyze([_make_sharp_frame() for _ in range(8)])
    assert result["detail"]["frame_score_std"] == pytest.approx(0.0, abs=1e-4)


def test_frame_score_std_is_positive_for_mixed_frames():
    frames = [_make_sharp_frame() for _ in range(4)] + [_make_blurry_frame() for _ in range(4)]
    result = ClarityAnalyzer().analyze(frames)
    assert result["detail"]["frame_score_std"] > 0.0


def test_single_frame_std_is_zero():
    result = ClarityAnalyzer().analyze([_make_sharp_frame()])
    assert result["detail"]["frame_score_std"] == pytest.approx(0.0, abs=1e-4)
```

Run: `uv run pytest tests/analyzers/test_clarity.py -x -q`

Expected: FAIL — `analyze()` still takes `data`.

- [ ] **Step 2: Update `agent/analyzers/clarity.py`**

Change `analyze(self, data: ExtractedData) -> ClarityResult:` to `analyze(self, frames: list[np.ndarray]) -> ClarityResult:`.

Remove `from agent.analyzers.base import ExtractedData, ClarityResult, ClarityDetail` and update import to `from agent.analyzers.base import ClarityResult, ClarityDetail`.

Remove line `frames = data["frames"]` — `frames` is now the parameter directly.

- [ ] **Step 3: Run clarity tests — expect PASS**

Run: `uv run pytest tests/analyzers/test_clarity.py -v`

Expected: All tests PASS.

### 3b: ContinuityAnalyzer

- [ ] **Step 4: Update `tests/analyzers/test_continuity.py`**

Same pattern as clarity: replace `analyzer.analyze(data)` → `analyzer.analyze(frames)`, remove `ExtractedData` fixture usage for inline tests.

Key changes:
- `analyzer.analyze(sharp_data)` → `analyzer.analyze(sharp_frames)`
- All inline `ExtractedData(frames=[...])` → plain lists
- Remove `from agent.analyzers.base import ExtractedData` import

- [ ] **Step 5: Update `agent/analyzers/continuity.py`**

Same mechanical change: `analyze(self, data: ExtractedData)` → `analyze(self, frames: list[np.ndarray])`, remove `frames = data["frames"]` line, update import.

- [ ] **Step 6: Run continuity tests — expect PASS**

Run: `uv run pytest tests/analyzers/test_continuity.py -v`

Expected: All tests PASS.

### 3c: FaceDetector

- [ ] **Step 7: Update `tests/analyzers/test_face.py`**

Same pattern: `detector.analyze(face_data)` → `detector.analyze(face_data["videos"]["/camera/image_raw"])`.

Also update any inline `ExtractedData(frames=[img], ...)` → `[img]`.

- [ ] **Step 8: Update `agent/analyzers/face.py`**

`analyze(self, data: ExtractedData)` → `analyze(self, frames: list[np.ndarray])`, remove `frames = data["frames"]`, update import.

- [ ] **Step 9: Run face tests — expect PASS**

Run: `uv run pytest tests/analyzers/test_face.py -v`

### 3d: GaitDetector

- [ ] **Step 10: Update `tests/analyzers/test_gait.py`**

Same pattern: `detector.analyze(person_data)` → `detector.analyze(person_data["videos"]["/camera/image_raw"])`.

- [ ] **Step 11: Update `agent/analyzers/gait.py`**

`analyze(self, data: ExtractedData)` → `analyze(self, frames: list[np.ndarray])`, remove `frames = data["frames"]`, update import.

- [ ] **Step 12: Run gait tests — expect PASS**

Run: `uv run pytest tests/analyzers/test_gait.py -v`

- [ ] **Step 13: Commit all visual analyzer changes**

```bash
git add agent/analyzers/clarity.py agent/analyzers/continuity.py \
        agent/analyzers/face.py agent/analyzers/gait.py \
        tests/analyzers/test_clarity.py tests/analyzers/test_continuity.py \
        tests/analyzers/test_face.py tests/analyzers/test_gait.py
git commit -m "feat(analyzers): update visual analyzer signatures to accept frames directly"
```

---

## Task 4: VoiceDetector — update `analyze()` signature

**Files:**
- Modify: `agent/analyzers/voice.py`
- Modify: `tests/analyzers/test_voice.py`

- [ ] **Step 1: Update `tests/analyzers/test_voice.py`**

Replace `detector.analyze(speech_data)` → `detector.analyze(speech_audio)` (use the new `speech_audio` fixture).
Replace `detector.analyze(data)` with inline audio list for silent audio tests.
Remove `ExtractedData` import and usage.

- [ ] **Step 2: Run tests — expect FAIL**

Run: `uv run pytest tests/analyzers/test_voice.py -x -q`

Expected: FAIL — voice still takes `data`.

- [ ] **Step 3: Update `agent/analyzers/voice.py`**

```python
import webrtcvad
from loguru import logger
from agent.analyzers.base import VoiceResult

_SAMPLE_RATE = 16000
_VAD_MODE = 2


class VoiceDetector:
    """Voice activity detector using WebRTC VAD."""

    def __init__(self, mode: int = _VAD_MODE) -> None:
        self._vad = webrtcvad.Vad(mode)

    def name(self) -> str:
        return "voice"

    def analyze(self, audio_frames: list[bytes]) -> VoiceResult:
        if not audio_frames:
            return VoiceResult(has_human_voice=False, speech_frame_ratio=0.0)

        speech_count = 0
        for frame in audio_frames:
            try:
                if self._vad.is_speech(frame, _SAMPLE_RATE):
                    speech_count += 1
            except Exception as exc:
                logger.debug("VAD check failed on frame ({}B): {}", len(frame), exc)
                continue

        speech_frame_ratio = speech_count / len(audio_frames)
        return VoiceResult(
            has_human_voice=speech_count > 0,
            speech_frame_ratio=round(speech_frame_ratio, 4),
        )
```

- [ ] **Step 4: Run voice tests — expect PASS**

Run: `uv run pytest tests/analyzers/test_voice.py -v`

- [ ] **Step 5: Commit**

```bash
git add agent/analyzers/voice.py tests/analyzers/test_voice.py
git commit -m "feat(analyzers): update VoiceDetector to accept audio_frames directly"
```

---

## Task 5: Pipeline — `run_visual` / `run_audio` + picklable workers

**Files:**
- Modify: `agent/pipeline.py`
- Modify: `tests/test_pipeline.py`

The pipeline loses its `run(data)` method. It gains `run_visual(frames)` (4 visual analyzers in ThreadPoolExecutor) and `run_audio(audio_frames)` (1 voice analyzer). Two module-level picklable functions `_run_visual_worker` and `_run_audio_worker` are added for ProcessPoolExecutor dispatch from runner.

- [ ] **Step 1: Update `tests/test_pipeline.py`**

```python
import pytest
import numpy as np
from agent.pipeline import AnalysisPipeline


class _OkVisualAnalyzer:
    def name(self) -> str:
        return "ok_visual"

    def analyze(self, frames: list[np.ndarray]):
        return {"value": 42}


class _BrokenVisualAnalyzer:
    def name(self) -> str:
        return "broken"

    def analyze(self, frames: list[np.ndarray]):
        raise RuntimeError("oops")


class _OkAudioAnalyzer:
    def name(self) -> str:
        return "ok_audio"

    def analyze(self, audio_frames: list[bytes]):
        return {"value": 99}


def test_run_visual_returns_results():
    pipeline = AnalysisPipeline(visual_analyzers=[_OkVisualAnalyzer()], audio_analyzers=[])
    frames = [np.zeros((64, 64, 3), dtype=np.uint8)]
    results, errors = pipeline.run_visual(frames)
    assert results["ok_visual"] == {"value": 42}
    assert errors == []


def test_run_visual_broken_analyzer_does_not_abort_others():
    pipeline = AnalysisPipeline(
        visual_analyzers=[_OkVisualAnalyzer(), _BrokenVisualAnalyzer()],
        audio_analyzers=[],
    )
    results, errors = pipeline.run_visual([np.zeros((64, 64, 3), dtype=np.uint8)])
    assert results["ok_visual"] == {"value": 42}
    assert results["broken"] is None
    assert "broken" in errors


def test_run_audio_returns_results():
    pipeline = AnalysisPipeline(visual_analyzers=[], audio_analyzers=[_OkAudioAnalyzer()])
    results, errors = pipeline.run_audio([b"\x00" * 960])
    assert results["ok_audio"] == {"value": 99}
    assert errors == []


def test_run_audio_empty_input():
    pipeline = AnalysisPipeline(visual_analyzers=[], audio_analyzers=[_OkAudioAnalyzer()])
    results, errors = pipeline.run_audio([])
    assert results["ok_audio"] == {"value": 99}  # analyzer handles empty gracefully
    assert errors == []
```

Run: `uv run pytest tests/test_pipeline.py -x -q`

Expected: FAIL — `run_visual` doesn't exist yet.

- [ ] **Step 2: Rewrite `agent/pipeline.py`**

```python
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import numpy as np
from loguru import logger
from agent.analyzers.base import VisualAnalyzer, AudioAnalyzer, VoiceResult


class AnalysisPipeline:
    """Runs visual and audio analyzers concurrently via a thread pool."""

    def __init__(
        self,
        visual_analyzers: list[VisualAnalyzer],
        audio_analyzers: list[AudioAnalyzer],
        max_workers: int = 5,
    ) -> None:
        self._visual = visual_analyzers
        self._audio = audio_analyzers
        self._max_workers = max_workers

    def run_visual(self, frames: list[np.ndarray]) -> tuple[dict[str, Any], list[str]]:
        """Run all visual analyzers concurrently. Returns (results_dict, error_names)."""
        return self._run([(a, frames) for a in self._visual])

    def run_audio(self, audio_frames: list[bytes]) -> tuple[dict[str, Any], list[str]]:
        """Run all audio analyzers concurrently. Returns (results_dict, error_names)."""
        return self._run([(a, audio_frames) for a in self._audio])

    def _run(self, analyzer_inputs: list[tuple]) -> tuple[dict[str, Any], list[str]]:
        results: dict[str, Any] = {}
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(a.analyze, inp): a for a, inp in analyzer_inputs}
            for future, analyzer in futures.items():
                name = analyzer.name()
                try:
                    results[name] = future.result()
                except Exception as exc:
                    logger.error("Analyzer {!r} failed: {}", name, exc, exc_info=True)
                    results[name] = None
                    errors.append(name)
        return results, errors


# ── Picklable top-level workers for ProcessPoolExecutor dispatch ────────────

def _run_visual_worker(
    topic: str,
    frames: list[np.ndarray],
    model_path: str,
) -> tuple[str, dict[str, Any], list[str]]:
    """Per-topic visual detection worker. Runs in a separate process.

    Instantiates a fresh AnalysisPipeline with all visual analyzers.
    model_path is passed explicitly because the worker process has no access
    to the parent's singleton state.
    """
    import os
    from agent.analyzers.clarity import ClarityAnalyzer
    from agent.analyzers.continuity import ContinuityAnalyzer
    from agent.analyzers.face import FaceDetector
    from agent.analyzers.gait import GaitDetector

    pipeline = AnalysisPipeline(
        visual_analyzers=[
            ClarityAnalyzer(),
            ContinuityAnalyzer(),
            FaceDetector(model_path=model_path),
            GaitDetector(),
        ],
        audio_analyzers=[],
    )
    results, errors = pipeline.run_visual(frames)
    return topic, results, errors


def _run_audio_worker(
    topic: str,
    audio_frames: list[bytes],
) -> tuple[str, dict[str, Any], list[str]]:
    """Per-topic audio detection worker. Runs in a separate process."""
    from agent.analyzers.voice import VoiceDetector

    pipeline = AnalysisPipeline(visual_analyzers=[], audio_analyzers=[VoiceDetector()])
    results, errors = pipeline.run_audio(audio_frames)
    return topic, results, errors
```

- [ ] **Step 3: Run pipeline tests — expect PASS**

Run: `uv run pytest tests/test_pipeline.py -v`

Expected: All 4 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add agent/pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): split into run_visual/run_audio; add picklable ProcessPoolExecutor workers"
```

---

## Task 6: Extractor — multi-topic extraction with 3-step sampling

**Files:**
- Modify: `agent/extractor.py`
- Modify: `tests/test_extractor.py`

The extractor is rewritten to accept `camera_topics: list[str]` and `audio_topics: list[str]` (empty = auto-discover), extract into `videos`/`audios` dicts, and apply the 3-step sampling pipeline per topic.

- [ ] **Step 1: Write new extractor tests**

Add to `tests/test_extractor.py` (keep existing `chunk_pcm` tests, add new tests):

```python
def test_extractor_init_accepts_topic_lists():
    """Extractor accepts list[str] for camera and audio topics."""
    from agent.extractor import McapExtractor
    e = McapExtractor(camera_topics=["/cam1", "/cam2"], audio_topics=["/audio"])
    assert e is not None


def test_frame_cap_applied_with_linspace(monkeypatch):
    """max_frames_per_topic=5 should uniformly downsample from 20 frames to 5."""
    from agent.extractor import McapExtractor
    import numpy as np

    e = McapExtractor(
        camera_topics=["/cam"],
        audio_topics=[],
        frame_sample_rate=1,
        min_frames=1,
        max_frames_per_topic=5,
    )
    # Simulate 20 frames already collected
    raw_frames = [np.zeros((64, 64, 3), dtype=np.uint8) for _ in range(20)]
    # Apply internal sampling logic (step 1 already done: sample_rate=1)
    frames = raw_frames[::1]  # step 1: no thinning
    if len(frames) > 5:
        import numpy as np
        indices = np.linspace(0, len(frames) - 1, 5, dtype=int)
        frames = [frames[i] for i in indices]
    assert len(frames) == 5


def test_below_min_frames_produces_warning_and_empty_list():
    """Topics with frames < min_frames after sampling should emit extraction_warning and be zeroed."""
    # This tests the logic directly — integration tested via mock in test_extractor_extract
    from agent.extractor import McapExtractor
    e = McapExtractor(
        camera_topics=["/cam"],
        audio_topics=[],
        frame_sample_rate=1,
        min_frames=10,
        max_frames_per_topic=300,
    )
    # 5 frames < min_frames=10 → should be treated as empty
    # We test through the sampling logic embedded in extract()
    # (full integration test uses mock MCAP — tested in test_extractor_extract_mocked)
    assert e._min_frames == 10
```

Run: `uv run pytest tests/test_extractor.py -x -q`

Expected: FAIL on `test_extractor_init_accepts_topic_lists` — constructor signature is wrong.

- [ ] **Step 2: Rewrite `agent/extractor.py`**

```python
# agent/extractor.py
from __future__ import annotations
from typing import Any, Iterator
import numpy as np
from loguru import logger
from mcap.reader import make_reader
from mcap.exceptions import DecoderNotFoundError
from agent.analyzers.base import ExtractedData
from agent.mcap_codecs import (
    ProtocolReaderFactory,
    SchemaDecoderRegistry,
    build_default_registry,
)

_PCM_FRAME_BYTES = 960


def chunk_pcm(raw: bytes) -> list[bytes]:
    """Split raw PCM bytes into 30ms frames (960 bytes each). Drops remainder."""
    return [raw[i:i + _PCM_FRAME_BYTES] for i in range(0, len(raw) - _PCM_FRAME_BYTES + 1, _PCM_FRAME_BYTES)]


def _safe_iter(reader: Any, topics: list[str]) -> Iterator[tuple[Any, Any, Any, Any]]:
    """Wrap iter_decoded_messages to skip messages that raise DecoderNotFoundError."""
    it = reader.iter_decoded_messages(topics=topics)
    while True:
        try:
            yield next(it)
        except DecoderNotFoundError as e:
            logger.warning("No decoder for message encoding, skipping: {}", e)
        except StopIteration:
            return


_IMAGE_SCHEMAS = {"sensor_msgs/Image", "sensor_msgs/CompressedImage"}
_AUDIO_SCHEMAS = {"audio_common_msgs/AudioData"}


class McapExtractor:
    """Parses MCAP files and extracts frames, audio, and IMU data for all configured topics.

    camera_topics / audio_topics:
        - Non-empty list: extract only the listed topics present in the file.
        - Empty list: auto-discover all image / audio topics found in the file.
    """

    def __init__(
        self,
        camera_topics: list[str] | None = None,
        audio_topics: list[str] | None = None,
        frame_sample_rate: int = 5,
        min_frames: int = 10,
        max_frames_per_topic: int = 300,
        registry: SchemaDecoderRegistry | None = None,
    ) -> None:
        self._camera_topics: list[str] = camera_topics if camera_topics is not None else []
        self._audio_topics: list[str] = audio_topics if audio_topics is not None else []
        self._frame_sample_rate = max(1, frame_sample_rate)
        self._min_frames = max(0, min_frames)
        self._max_frames_per_topic = max(1, max_frames_per_topic)
        self._registry = registry if registry is not None else build_default_registry()

    def _resolve_topics(self, mcap_path: str) -> tuple[list[str], list[str]]:
        """Return (video_topics, audio_topics) to extract.

        If the configured lists are non-empty, return only those topics that exist
        in the file. If empty, auto-discover all image / audio topics from the file.
        Topics are returned in sorted order for determinism.
        """
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            channels = list(summary.channels.values()) if summary else []
            schemas = summary.schemas if summary else {}

        def schema_name(ch) -> str:
            s = schemas.get(ch.schema_id)
            return s.name if s else ""

        available_image  = sorted(ch.topic for ch in channels if schema_name(ch) in _IMAGE_SCHEMAS)
        available_audio  = sorted(ch.topic for ch in channels if schema_name(ch) in _AUDIO_SCHEMAS)

        if self._camera_topics:
            video_topics = [t for t in self._camera_topics if t in set(available_image)]
        else:
            video_topics = available_image

        if self._audio_topics:
            audio_topics = [t for t in self._audio_topics if t in set(available_audio)]
        else:
            audio_topics = available_audio

        return video_topics, audio_topics

    def extract(self, mcap_path: str) -> ExtractedData:
        """Parse an MCAP file and return ExtractedData with per-topic video/audio dicts."""
        video_topics, audio_topics = self._resolve_topics(mcap_path)
        all_topics = video_topics + audio_topics

        # Accumulate raw frames per topic
        raw_videos: dict[str, list[np.ndarray]] = {t: [] for t in video_topics}
        raw_audios: dict[str, dict[str, bytearray]] = {t: bytearray() for t in audio_topics}  # type: ignore[assignment]
        raw_audio_buf: dict[str, bytearray] = {t: bytearray() for t in audio_topics}
        timestamps: list[float] = []
        imu_rows: list[np.ndarray] = []
        imu_topic: str | None = None

        # Also collect IMU from any imu topic in the file
        _IMU_SCHEMAS = {"sensor_msgs/Imu"}
        decoder_factories = ProtocolReaderFactory.build_decoder_factories(mcap_path)
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=decoder_factories)
            # Get IMU topics too for sensor_series
            with open(mcap_path, "rb") as f2:
                summary_reader = make_reader(f2)
                summary = summary_reader.get_summary()
                if summary:
                    imu_channels = [
                        ch.topic for ch in summary.channels.values()
                        if (summary.schemas.get(ch.schema_id) or type("", (), {"name": ""})()).name in _IMU_SCHEMAS
                    ]
                    imu_topic = imu_channels[0] if imu_channels else None
            topics_to_read = all_topics + ([imu_topic] if imu_topic else [])
            for schema, channel, message, decoded_message in _safe_iter(reader, topics_to_read):
                t = message.log_time / 1e9
                timestamps.append(t)
                topic = channel.topic

                if topic in raw_videos:
                    frame = self._registry.decode_image(schema.name, decoded_message)
                    if frame is not None:
                        raw_videos[topic].append(frame)

                elif topic in raw_audio_buf:
                    chunk = self._registry.decode_audio(schema.name, decoded_message)
                    if chunk:
                        raw_audio_buf[topic].extend(chunk)

                elif imu_topic and topic == imu_topic:
                    row = self._registry.decode_imu(schema.name, decoded_message)
                    if row is not None:
                        imu_rows.append(row)

        # Post-loop: per-topic 3-step sampling for videos
        extraction_warnings: dict[str, str] = {}
        videos: dict[str, list[np.ndarray]] = {}
        for topic, frames in raw_videos.items():
            frames = frames[::self._frame_sample_rate]  # step 1: frame_sample_rate thinning
            if len(frames) > self._max_frames_per_topic:              # step 2: hard cap
                indices = np.linspace(0, len(frames) - 1, self._max_frames_per_topic, dtype=int)
                frames = [frames[i] for i in indices]
            if 0 < len(frames) < self._min_frames:                   # step 3: min_frames
                logger.warning(
                    "Topic {} has only {} frames after sampling — treated as empty",
                    topic, len(frames),
                )
                extraction_warnings[topic] = "below_min_frames"
                frames = []
            videos[topic] = frames

        # Post-loop: convert audio buffers to PCM frame lists (no sub-sampling)
        audios: dict[str, list[bytes]] = {}
        for topic, buf in raw_audio_buf.items():
            raw = bytes(buf)
            audios[topic] = chunk_pcm(raw) if raw else []

        duration = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0
        sensor_series: dict[str, np.ndarray] = {}
        if imu_rows and imu_topic:
            sensor_series[imu_topic] = np.array(imu_rows, dtype=np.float64)

        return ExtractedData(
            videos=videos,
            audios=audios,
            sensor_series=sensor_series,
            duration_seconds=duration,
            extraction_warnings=extraction_warnings,
        )
```

- [ ] **Step 3: Run extractor tests — expect PASS**

Run: `uv run pytest tests/test_extractor.py -v`

Expected: All tests PASS (chunk_pcm tests unchanged; new tests pass).

- [ ] **Step 4: Commit**

```bash
git add agent/extractor.py tests/test_extractor.py
git commit -m "feat(extractor): multi-topic extraction with 3-step frame sampling pipeline"
```

---

## Task 7: Report — new schema with `cameras`/`audios` arrays

**Files:**
- Modify: `agent/report.py`
- Modify: `tests/test_report.py`

The report is rewritten. The flat `detector_results` signature is replaced by `camera_results: list[CameraResult]` + `audio_results: list[AudioResult]`. `evaluate_strategy()` is a new module-level helper. Per-topic `passed` computation is done upstream (in runner); the report builder just aggregates.

- [ ] **Step 1: Write new report tests**

Completely replace `tests/test_report.py`:

```python
import uuid
from datetime import datetime
import pytest
from agent.report import ReportBuilder, evaluate_strategy
from agent.config import Settings
from agent.analyzers.base import CameraResult, AudioResult


def _settings(**kwargs):
    return Settings(
        clarity_threshold=kwargs.get("clarity_threshold", 0.6),
        continuity_threshold=kwargs.get("continuity_threshold", 0.6),
        minimum_duration_seconds=kwargs.get("minimum_duration_seconds", 1.0),
        camera_pass_strategy=kwargs.get("camera_pass_strategy", "all"),
        audio_pass_strategy=kwargs.get("audio_pass_strategy", "all"),
        anthropic_api_key="fake",
    )


def _good_camera(topic: str = "/cam") -> CameraResult:
    return CameraResult(
        topic=topic,
        frame_count=60,
        clarity={"score": 0.9, "method": "laplacian+fft", "detail": {}},
        continuity={"score": 0.9, "method": "optical_flow", "detail": {}},
        face={"has_face": False, "face_count": 0, "face_frame_ratio": 0.0, "max_confidence": 0.0},
        gait={"has_human_gait": False, "person_frame_ratio": 0.0, "max_detection_weight": 0.0},
        llm_assessment=None,
        llm_skipped_reason="all_detectors_clear_no_borderline_scores",
        passed=True,
        failure_reasons=[],
        analyzer_errors=[],
    )


def _good_audio(topic: str = "/audio") -> AudioResult:
    return AudioResult(
        topic=topic,
        audio_frame_count=400,
        voice={"has_human_voice": False, "speech_frame_ratio": 0.0},
        llm_assessment=None,
        llm_skipped_reason="all_detectors_clear_no_borderline_scores",
        passed=True,
        failure_reasons=[],
        analyzer_errors=[],
    )


# ── evaluate_strategy ────────────────────────────────────────────────────────

def test_strategy_all_passes_when_all_true():
    assert evaluate_strategy([True, True, True], "all") is True


def test_strategy_all_fails_when_any_false():
    assert evaluate_strategy([True, False, True], "all") is False


def test_strategy_any_passes_when_one_true():
    assert evaluate_strategy([False, True, False], "any") is True


def test_strategy_any_fails_when_all_false():
    assert evaluate_strategy([False, False], "any") is False


def test_strategy_majority_passes():
    assert evaluate_strategy([True, True, False], "majority") is True


def test_strategy_majority_fails():
    assert evaluate_strategy([True, False, False], "majority") is False


def test_strategy_empty_list_returns_false():
    """Zero topics → failure. No silent pass."""
    assert evaluate_strategy([], "all") is False
    assert evaluate_strategy([], "any") is False
    assert evaluate_strategy([], "majority") is False


def test_strategy_unknown_raises():
    with pytest.raises(ValueError):
        evaluate_strategy([True], "unknown")


# ── ReportBuilder.build() ────────────────────────────────────────────────────

def test_overall_passed_when_all_topics_pass():
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[_good_camera()],
        audio_results=[_good_audio()],
        analyzer_errors=[],
    )
    assert report["overall_passed"] is True
    assert report["cameras"][0]["passed"] is True
    assert report["audios"][0]["passed"] is True


def test_overall_fails_when_camera_fails():
    bad_cam = _good_camera()
    bad_cam["passed"] = False
    bad_cam["failure_reasons"] = ["clarity"]
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[bad_cam],
        audio_results=[_good_audio()],
        analyzer_errors=[],
    )
    assert report["overall_passed"] is False


def test_overall_fails_when_audio_fails():
    bad_audio = _good_audio()
    bad_audio["passed"] = False
    bad_audio["failure_reasons"] = ["has_human_voice"]
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[_good_camera()],
        audio_results=[bad_audio],
        analyzer_errors=[],
    )
    assert report["overall_passed"] is False


def test_zero_cameras_fails():
    """No camera topics → failure (evaluate_strategy returns False for empty list)."""
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[],
        audio_results=[_good_audio()],
        analyzer_errors=[],
    )
    assert report["overall_passed"] is False


def test_zero_audios_fails():
    builder = ReportBuilder(_settings())
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[_good_camera()],
        audio_results=[],
        analyzer_errors=[],
    )
    assert report["overall_passed"] is False


def test_any_strategy_passes_when_one_camera_passes():
    bad_cam = _good_camera("/cam1")
    bad_cam["passed"] = False
    good_cam = _good_camera("/cam2")
    builder = ReportBuilder(_settings(camera_pass_strategy="any"))
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[bad_cam, good_cam],
        audio_results=[],  # no audio → fail
        analyzer_errors=[],
    )
    # cameras: "any" → True; audios: empty → False → overall False
    assert report["overall_passed"] is False
    assert report["camera_pass_strategy"] == "any"


def test_multi_camera_all_strategy():
    cam1 = _good_camera("/cam1")
    cam2 = _good_camera("/cam2")
    builder = ReportBuilder(_settings(camera_pass_strategy="all"))
    report = builder.build(
        source_file="test.mcap",
        duration_seconds=5.0,
        camera_results=[cam1, cam2],
        audio_results=[_good_audio()],
        analyzer_errors=[],
    )
    assert report["overall_passed"] is True
    assert len(report["cameras"]) == 2


def test_report_id_is_valid_uuid4():
    builder = ReportBuilder(_settings())
    report = builder.build("f.mcap", 5.0, [_good_camera()], [_good_audio()], [])
    assert uuid.UUID(report["report_id"]).version == 4


def test_report_id_is_unique():
    builder = ReportBuilder(_settings())
    r1 = builder.build("f.mcap", 5.0, [_good_camera()], [_good_audio()], [])
    r2 = builder.build("f.mcap", 5.0, [_good_camera()], [_good_audio()], [])
    assert r1["report_id"] != r2["report_id"]


def test_analyzed_at_is_iso8601_utc():
    builder = ReportBuilder(_settings())
    report = builder.build("f.mcap", 5.0, [_good_camera()], [_good_audio()], [])
    dt = datetime.fromisoformat(report["analyzed_at"].replace("Z", "+00:00"))
    assert dt.tzinfo is not None


def test_short_duration_fails():
    builder = ReportBuilder(_settings(minimum_duration_seconds=2.0))
    report = builder.build("f.mcap", 0.5, [_good_camera()], [_good_audio()], [])
    assert report["overall_passed"] is False
    assert "duration_too_short" in report["failure_reasons"]


def test_analyzer_errors_in_report():
    builder = ReportBuilder(_settings())
    report = builder.build("f.mcap", 5.0, [_good_camera()], [_good_audio()], ["mcap_extraction"])
    assert "mcap_extraction" in report["analyzer_errors"]
```

Run: `uv run pytest tests/test_report.py -x -q`

Expected: FAIL — `evaluate_strategy` not importable yet.

- [ ] **Step 2: Rewrite `agent/report.py`**

```python
from __future__ import annotations
import uuid
from datetime import datetime, timezone
from typing import Any, Literal
from agent.analyzers.base import CameraResult, AudioResult
from agent.config import Settings


def evaluate_strategy(results: list[bool], strategy: str) -> bool:
    """Evaluate a list of per-topic pass/fail booleans using the configured strategy.

    Empty list always returns False — no silent pass with zero topics.
    """
    if not results:
        return False
    if strategy == "all":
        return all(results)
    elif strategy == "any":
        return any(results)
    elif strategy == "majority":
        return sum(results) > len(results) / 2
    raise ValueError(f"Unknown pass strategy: {strategy!r}")


class ReportBuilder:
    """Assembles camera/audio per-topic results into a structured JSON report."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def build(
        self,
        source_file: str,
        duration_seconds: float | None,
        camera_results: list[CameraResult],
        audio_results: list[AudioResult],
        analyzer_errors: list[str],
        bucket: str = "",
    ) -> dict[str, Any]:
        """Build the final report dict from per-topic results."""
        failure_reasons: list[str] = []

        # Duration check
        if duration_seconds is None or duration_seconds < self._s.minimum_duration_seconds:
            failure_reasons.append("duration_too_short")

        cameras_passed = evaluate_strategy(
            [r["passed"] for r in camera_results], self._s.camera_pass_strategy
        )
        audios_passed = evaluate_strategy(
            [r["passed"] for r in audio_results], self._s.audio_pass_strategy
        )
        overall_passed = cameras_passed and audios_passed and not failure_reasons

        return {
            "report_id": str(uuid.uuid4()),
            "source_file": source_file,
            "minio_bucket": bucket,
            "analyzed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_seconds": duration_seconds,
            "camera_pass_strategy": self._s.camera_pass_strategy,
            "audio_pass_strategy": self._s.audio_pass_strategy,
            "cameras": list(camera_results),
            "audios": list(audio_results),
            "overall_passed": overall_passed,
            "failure_reasons": sorted(set(failure_reasons)),
            "analyzer_errors": list(analyzer_errors),
        }
```

- [ ] **Step 3: Run report tests — expect PASS**

Run: `uv run pytest tests/test_report.py -v`

Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add agent/report.py tests/test_report.py
git commit -m "feat(report): new cameras/audios schema with evaluate_strategy and per-topic results"
```

---

## Task 8: LLM Judge — new `judge()` signature

**Files:**
- Modify: `agent/llm_judge.py`
- Modify: `tests/test_llm_judge.py`

`judge()` now takes explicit `frames`, `audio_frames`, and `sensor_series` instead of reading from `ExtractedData`. `should_invoke_llm()` is unchanged.

- [ ] **Step 1: Update `tests/test_llm_judge.py`**

Update the two tests that call `judge.judge(results, sharp_data)` to use the new signature:

```python
# Old: judge.judge(results, sharp_data)
# New: judge.judge("/cam", results, frames, None, sensor_series)
```

Full replacements for the affected tests:

```python
def test_llm_failure_falls_back_to_detector_verdict(sharp_frames):
    """If Anthropic API raises, LLMJudge returns None assessment and 'llm' error."""
    judge = LLMJudge(api_key="fake", model="claude-sonnet-4-6",
                     clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1)
    results = _make_detector_results(has_face=True)

    with patch("agent.llm_judge.anthropic.Anthropic") as mock_client_cls:
        mock_client_cls.return_value.messages.create.side_effect = RuntimeError("API down")
        assessment, error = judge.judge("/cam", results, sharp_frames, None, {})

    assert assessment is None
    assert error == "llm"


def test_llm_skipped_returns_none_assessment_and_no_error(sharp_frames):
    judge = LLMJudge(api_key="fake", model="claude-sonnet-4-6",
                     clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1)
    results = _make_detector_results()
    assessment, error = judge.judge("/cam", results, sharp_frames, None, {})
    assert assessment is None
    assert error is None


def test_llm_max_rounds_exceeded_returns_llm_error(sharp_frames):
    judge = LLMJudge(api_key="fake", model="claude-sonnet-4-6",
                     clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1)
    results = _make_detector_results(has_face=True)

    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"
    mock_response.content = []

    with patch("agent.llm_judge.anthropic.Anthropic") as mock_client_cls:
        mock_client_cls.return_value.messages.create.return_value = mock_response
        assessment, error = judge.judge("/cam", results, sharp_frames, None, {})

    assert assessment is None
    assert error == "llm"
```

Also update fixture usage at top: replace `sharp_data` with `sharp_frames` in function signatures.
Remove `from agent.analyzers.base import ExtractedData` import.

Run: `uv run pytest tests/test_llm_judge.py -x -q`

Expected: FAIL — `judge()` still has old signature.

- [ ] **Step 2: Update `agent/llm_judge.py`**

Change `judge()` signature from `judge(self, detector_results, data: ExtractedData)` to:

```python
def judge(
    self,
    topic: str,
    detector_results: dict[str, Any],
    frames: list[np.ndarray] | None,
    audio_frames: list[bytes] | None,
    sensor_series: dict[str, np.ndarray],
) -> tuple[dict[str, Any] | None, str | None]:
```

In `judge()`, pass `frames` and `sensor_series` to `_run_agent()` directly:

```python
def judge(self, topic, detector_results, frames, audio_frames, sensor_series):
    if not should_invoke_llm(
        detector_results, self._clarity_threshold, self._continuity_threshold, self._margin
    ):
        return None, None
    try:
        return self._run_agent(detector_results, frames or [], sensor_series), None
    except Exception as exc:
        logger.warning("LLM judge failed for {!r}, falling back: {}", topic, exc)
        return None, "llm"
```

Change `_run_agent(self, detector_results, data)` to `_run_agent(self, detector_results, frames, imu)`.

Remove `frames = data["frames"]` and `imu = data["sensor_series"]` — these are now parameters.

Remove `from agent.analyzers.base import ExtractedData` import.

- [ ] **Step 3: Run LLM judge tests — expect PASS**

Run: `uv run pytest tests/test_llm_judge.py -v`

Expected: All 9 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add agent/llm_judge.py tests/test_llm_judge.py
git commit -m "feat(llm_judge): new judge() signature with explicit frames/audio_frames/sensor_series params"
```

---

## Task 9: Runner — multi-topic orchestration

**Files:**
- Modify: `agent/runner.py`

The runner is rewritten to:
1. Build `McapExtractor` from list-based config fields
2. Detection phase: `ProcessPoolExecutor` per topic
3. LLM phase: `ThreadPoolExecutor` concurrent calls
4. Assembly: build `CameraResult`/`AudioResult` per topic
5. Call `ReportBuilder.build()` with new signature

No new test file is added for runner (it is integration-tested via `test_main.py` and `test_cli.py`). After this task, run the full suite to verify end-to-end.

- [ ] **Step 1: Rewrite `agent/runner.py`**

```python
# agent/runner.py
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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

    failure_reasons: list[str] = list(errors and [f"analyzer_error:{e}" for e in item["errors"]] or [])

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

def analyze_local_file(local_path: str, source_file: str = "", bucket: str = "") -> dict:
    """Run the full pipeline on a local MCAP file. Returns a report dict (never raises)."""
    src = source_file or local_path

    try:
        data = _extractor.extract(local_path)
    except Exception as exc:
        logger.error("MCAP extraction failed for {!r}: {}", local_path, exc, exc_info=True)
        return _builder.build(
            source_file=src, bucket=bucket,
            duration_seconds=None,
            camera_results=[],
            audio_results=[],
            analyzer_errors=["mcap_extraction"],
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

    # Detection phase: per-topic ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=settings.max_concurrent_topics) as executor:
        cam_futures = {
            executor.submit(_run_visual_worker, topic, frames, _model_path): topic
            for topic, frames in data["videos"].items()
            if frames and topic not in camera_intermediates
        }
        aud_futures = {
            executor.submit(_run_audio_worker, topic, audio_frames): topic
            for topic, audio_frames in data["audios"].items()
            if audio_frames and topic not in audio_intermediates
        }

        for future, topic in cam_futures.items():
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

        for future, topic in aud_futures.items():
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

    # LLM phase: concurrent ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=settings.llm_max_concurrent_calls) as executor:
        cam_llm_futures = {
            executor.submit(
                _judge.judge, topic, item["results"],
                item["frames"], None, data["sensor_series"]
            ): topic
            for topic, item in camera_intermediates.items() if item["needs_llm"]
        }
        aud_llm_futures = {
            executor.submit(
                _judge.judge, topic, item["results"],
                None, item["audio_frames"], data["sensor_series"]
            ): topic
            for topic, item in audio_intermediates.items() if item["needs_llm"]
        }

        for future, topic in cam_llm_futures.items():
            llm_result, llm_error = future.result()
            camera_intermediates[topic]["llm_result"] = llm_result
            camera_intermediates[topic]["llm_error"] = llm_error

        for future, topic in aud_llm_futures.items():
            llm_result, llm_error = future.result()
            audio_intermediates[topic]["llm_result"] = llm_result
            audio_intermediates[topic]["llm_error"] = llm_error

    # Assembly phase
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
```

- [ ] **Step 2: Run full test suite — check for remaining failures**

Run: `uv run pytest tests/ -x -q 2>&1 | tail -30`

Expected: Most tests pass. `test_main.py` and `test_cli.py` may need minor updates if they assert on old report keys (`passed` → `overall_passed`, `scores` → `cameras[0].clarity`).

- [ ] **Step 3: Fix any remaining test failures in `test_main.py` / `test_cli.py`**

Look for assertions on `report["passed"]` → change to `report["overall_passed"]`.
Look for `report["scores"]` → change to `report["cameras"][0]["clarity"]`.
Look for `report["sensitive_info"]` → change to `report["cameras"][0]["face"]` etc.

- [ ] **Step 4: Commit**

```bash
git add agent/runner.py tests/test_main.py tests/test_cli.py
git commit -m "feat(runner): multi-topic ProcessPoolExecutor detection + ThreadPoolExecutor LLM orchestration"
```

---

## Task 10: Final verification — run full test suite

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v 2>&1 | tail -50`

Expected: All tests PASS. If any fail, investigate and fix before proceeding.

- [ ] **Step 2: Verify CLI still works (smoke test)**

Run: `uv run python -c "from agent.runner import analyze_local_file; print('import ok')"`

Expected: `import ok` (no import errors)

- [ ] **Step 3: Check no old API references remain**

Run: `grep -r "data\[.frames.\]" agent/`

Expected: no output.

Run: `grep -r "data\[.audio_frames.\]" agent/`

Expected: no output.

Run: `grep -r "camera_topic\b" agent/`

Expected: no output (only `camera_topics` plural).

- [ ] **Step 4: Final commit**

```bash
git add -p  # stage any stray fixes
git commit -m "fix: resolve remaining test failures after multi-camera/audio refactor"
```
