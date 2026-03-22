# Multi-Camera & Multi-Audio Support Design

**Date:** 2026-03-22
**Status:** Approved
**Scope:** Extend the data quality agent pipeline to support multiple video and audio streams from a single MCAP file, with per-topic detection results, independent LLM assessment, and configurable pass/fail strategy.

---

## Background

Robot sensor data (MCAP files) commonly contains multiple image topics (e.g., front camera, rear camera, fisheye) and multiple audio topics (e.g., front mic, rear mic). The current pipeline extracts only a single camera topic and a single audio topic, discarding all others. Detection results and the final report provide no per-topic breakdown, making it impossible to identify which camera or microphone has quality issues.

---

## Goals

1. Extract **all** video and audio topics from an MCAP file.
2. Run each detector independently per topic.
3. LLM assessment invoked independently per topic (concurrent, bounded).
4. Report structured as `cameras: [...]` and `audios: [...]` arrays, unified format for both single- and multi-stream files.
5. Overall `passed` determined by a configurable strategy (`all` / `any` / `majority`) per media type.
6. Zero-topic scenario treated as failure (no silent pass).

---

## Non-Goals

- Cross-camera comparison or synchronization analysis.
- Merging frames across cameras before analysis.

---

## Architecture

### Layer 1: Data Structures (`analyzers/base.py`)

Replace flat `frames` and `audio_frames` fields with topic-keyed dicts:

```python
class ExtractedData(TypedDict):
    videos: dict[str, list[np.ndarray]]   # topic → BGR frame list (replaces `frames`)
    audios: dict[str, list[bytes]]         # topic → 30ms PCM frame list (replaces `audio_frames`)
    sensor_series: dict[str, np.ndarray]   # unchanged (shared across all topics)
    duration_seconds: float
    extraction_warnings: dict[str, str]    # topic → warning reason ("below_min_frames", etc.)
```

**Analyzer Protocol change:** The `Analyzer` Protocol's `analyze(self, data: ExtractedData)` signature changes to accept the stream directly, eliminating the need for `ExtractedData` indirection inside analyzers:

```python
# Visual analyzers: clarity, continuity, face, gait
def analyze(self, frames: list[np.ndarray]) -> <ResultType>:
    ...

# Audio analyzer: voice
def analyze(self, audio_frames: list[bytes]) -> VoiceResult:
    ...
```

All five analyzer files (`clarity.py`, `continuity.py`, `face.py`, `gait.py`, `voice.py`) must be updated: replace `data["frames"]` / `data["audio_frames"]` access with the direct parameter. This is a mechanical change — no analysis logic changes.

Add result containers for per-topic report output:

```python
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
```

### Layer 2: Extractor (`extractor.py`)

**`McapExtractor.__init__`** signature changes:

```python
def __init__(
    self,
    camera_topics: list[str],          # replaces camera_topic: str
    audio_topics: list[str],           # replaces audio_topic: str
    frame_sample_rate: int = 5,
    min_frames: int = 10,
    max_frames_per_topic: int = 300,   # hard upper bound per topic after sampling
):
```

Empty list means "auto-discover all topics of this type." `runner.py` passes `settings.camera_topics` and `settings.audio_topics` directly.

**`_resolve_topics()`** returns all discovered image and audio topics:

```python
def _resolve_topics(self) -> tuple[list[str], list[str]]:
    # Returns (video_topics, audio_topics)
    # If camera_topics / audio_topics were specified, filter to those present in file
    # Otherwise return all discovered image / audio topics, sorted by channel name
```

Frame extraction buckets messages by topic, then applies sampling and `min_frames` validation per topic:

```python
videos: dict[str, list[np.ndarray]] = {t: [] for t in video_topics}
audios: dict[str, list[bytes]] = {t: [] for t in audio_topics}

for schema, channel, message in reader.iter_messages(topics=all_topics):
    if channel.topic in videos:
        videos[channel.topic].append(decode_image(message))
    elif channel.topic in audios:
        audios[channel.topic].append(decode_audio(message))

# Post-loop: per-topic sampling pipeline
# Step 1: frame_sample_rate — keep every Nth frame
# Step 2: max_frames_per_topic — uniform downsample if still over limit
# Step 3: min_frames — treat as empty if below threshold
for topic in list(videos):
    frames = videos[topic][::self._frame_sample_rate]
    if len(frames) > self._max_frames_per_topic:
        # Uniform downsample: pick max_frames_per_topic evenly spaced indices
        indices = np.linspace(0, len(frames) - 1, self._max_frames_per_topic, dtype=int)
        frames = [frames[i] for i in indices]
    if 0 < len(frames) < self._min_frames:
        logger.warning(
            "Topic {} has only {} frames after sampling — treated as empty", topic, len(frames)
        )
        extraction_warnings[topic] = "below_min_frames"
        frames = []
    videos[topic] = frames
```

Audio topics are not sub-sampled (PCM frame rate is already fixed at 30ms).

**Error string distinction in runner.py:**
- `"zero_frames"` — topic existed but had 0 messages at extraction time
- `"below_min_frames"` — topic had messages but fewer than `min_frames` after sampling; extractor sets the list to `[]` and the runner detects an empty list — runner should check whether the extractor logged a below-min-frames warning for this topic to emit the correct error string. In practice, the extractor can signal this by returning a sentinel (e.g., a special empty subclass) or by the runner checking an `extraction_warnings: dict[str, str]` field in `ExtractedData`. The simpler implementation: add `extraction_warnings: dict[str, str]` to `ExtractedData` where the extractor writes `{topic: "below_min_frames"}` for affected topics, and the runner reads it when building intermediates.

**Config migration:**

| Old field | New field | Notes |
|-----------|-----------|-------|
| `camera_topic: str` | `camera_topics: list[str]` | Empty = auto-discover all |
| `audio_topic: str` | `audio_topics: list[str]` | Empty = auto-discover all |

### Layer 3: Pipeline (`pipeline.py` + `runner.py`)

**Per-topic detection runs in separate processes** via `ProcessPoolExecutor`. Each worker process receives the frames for one topic, instantiates a fresh `AnalysisPipeline`, runs the detectors, and returns serializable results. Within each worker, the 4 visual analyzers still run concurrently via `ThreadPoolExecutor` (unchanged).

**Worker functions** (must be module-level for pickle compatibility):

```python
# pipeline.py — top-level functions, picklable
def _run_visual_worker(topic: str, frames: list[np.ndarray]) -> tuple[str, dict[str, Any], list[str]]:
    """Entry point for per-topic visual detection in a worker process."""
    pipeline = AnalysisPipeline()
    results, errors = pipeline.run_visual(frames)
    return topic, results, errors

def _run_audio_worker(topic: str, audio_frames: list[bytes]) -> tuple[str, VoiceResult, list[str]]:
    """Entry point for per-topic audio detection in a worker process."""
    pipeline = AnalysisPipeline()
    voice_result, errors = pipeline.run_audio(audio_frames)
    return topic, voice_result, errors
```

`AnalysisPipeline` gains two instance methods called from within workers:

```python
def run_visual(self, frames: list[np.ndarray]) -> tuple[dict[str, Any], list[str]]:
    # Runs clarity, continuity, face, gait in ThreadPoolExecutor
    # Returns (results_dict, errors_list)

def run_audio(self, audio_frames: list[bytes]) -> tuple[VoiceResult, list[str]]:
    # Runs voice detector; returns (VoiceResult, errors_list)
```

**Runner detection phase** uses `ProcessPoolExecutor`:

```python
# runner.py — parallel per-topic detection
camera_intermediates: dict[str, dict] = {}
audio_intermediates: dict[str, dict] = {}

# Pre-populate empty-topic entries (no process needed)
for topic, frames in data["videos"].items():
    error = data["extraction_warnings"].get(topic) or ("zero_frames" if not frames else None)
    if error:
        camera_intermediates[topic] = {"frames": [], "results": {}, "errors": [error], "needs_llm": False}

for topic, audio_frames in data["audios"].items():
    error = data["extraction_warnings"].get(topic) or ("zero_frames" if not audio_frames else None)
    if error:
        audio_intermediates[topic] = {"audio_frames": [], "results": {}, "errors": [error], "needs_llm": False}

# Dispatch non-empty topics to worker processes
with ProcessPoolExecutor(max_workers=settings.max_concurrent_topics) as executor:
    cam_futures = {
        executor.submit(_run_visual_worker, topic, frames): topic
        for topic, frames in data["videos"].items() if frames and topic not in camera_intermediates
    }
    aud_futures = {
        executor.submit(_run_audio_worker, topic, audio_frames): topic
        for topic, audio_frames in data["audios"].items() if audio_frames and topic not in audio_intermediates
    }
    for future, topic in cam_futures.items():
        _, results, errors = future.result()
        camera_intermediates[topic] = {
            "frames": data["videos"][topic], "results": results, "errors": errors,
            "needs_llm": should_invoke_llm(results, settings.clarity_threshold,
                                           settings.continuity_threshold, settings.llm_review_margin),
            "llm_result": None, "llm_error": None,
        }
    for future, topic in aud_futures.items():
        _, voice_result, errors = future.result()
        audio_intermediates[topic] = {
            "audio_frames": data["audios"][topic], "results": voice_result, "errors": errors,
            "needs_llm": voice_result.get("has_human_voice", False),
            "llm_result": None, "llm_error": None,
        }
```

**Serialization note:** numpy arrays are pickle-serializable. Frames are sent to worker processes via pickle. With `max_frames_per_topic=300` and a typical 1080p BGR frame (~6 MB), each topic's payload is ≤ 1.8 GB — stay within reasonable bounds by tuning `max_frames_per_topic` and `max_concurrent_topics` together. Workers are forked after extraction, so no shared memory issues.

### Layer 4: LLM Judge (`llm_judge.py`)

**Signature change:**

```python
class LLMJudge:
    def judge(
        self,
        topic: str,
        detector_results: dict[str, Any],
        frames: list[np.ndarray] | None,          # None for audio-only topics
        audio_frames: list[bytes] | None,         # None for camera-only topics
        sensor_series: dict[str, np.ndarray],     # shared IMU/sensor data (same for all topics)
    ) -> tuple[dict[str, Any] | None, str | None]:
        # Returns (llm_result, llm_error_name) — same tuple shape as current code
        ...
```

`_run_agent` receives frames and sensor_series as parameters (no longer reads `data["frames"]` or `data["sensor_series"]`). `get_key_frames` tool samples from the passed `frames` list; `get_imu_summary` tool reads from `sensor_series`.

**Concurrency:** Per-topic LLM calls are dispatched via `ThreadPoolExecutor` with a cap of `settings.llm_max_concurrent_calls` (new config field, default 4). One shared `LLMJudge` instance is reused across topics (it is stateless).

The runner uses an intermediate dict during the detection phase to carry frames alongside results, then runs LLM concurrently:

```python
# runner.py — concurrent LLM phase (after ProcessPoolExecutor detection phase completes)
# camera_intermediates: dict[topic, intermediate_dict] — built by detection phase above
with ThreadPoolExecutor(max_workers=settings.llm_max_concurrent_calls) as executor:
    cam_llm_futures = {
        executor.submit(
            judge.judge, topic, item["results"], item["frames"], None, data["sensor_series"]
        ): topic
        for topic, item in camera_intermediates.items() if item["needs_llm"]
    }
    aud_llm_futures = {
        executor.submit(
            judge.judge, topic, item["results"], None, item["audio_frames"], data["sensor_series"]
        ): topic
        for topic, item in audio_intermediates.items() if item["needs_llm"]
    }
    for future, topic in {**cam_llm_futures, **aud_llm_futures}.items():
        llm_result, llm_error = future.result()
        target = camera_intermediates if topic in camera_intermediates else audio_intermediates
        target[topic]["llm_result"] = llm_result
        target[topic]["llm_error"] = llm_error

# runner.py — build final result lists
camera_results = [_build_camera_result(topic, item) for topic, item in camera_intermediates.items()]
audio_results  = [_build_audio_result(topic, item)  for topic, item in audio_intermediates.items()]
```

`should_invoke_llm()` is the existing function from `llm_judge.py` — signature and logic unchanged.

### Layer 5: Report (`report.py`)

**`ReportBuilder.build()` new signature:**

```python
def build(
    self,
    source_file: str,
    duration_seconds: float,
    camera_results: list[CameraResult],
    audio_results: list[AudioResult],
    analyzer_errors: list[str],           # top-level errors (e.g., extraction failures)
) -> dict[str, Any]:
    cameras_passed = evaluate_strategy(
        [r["passed"] for r in camera_results], self._settings.camera_pass_strategy
    )
    audios_passed = evaluate_strategy(
        [r["passed"] for r in audio_results], self._settings.audio_pass_strategy
    )
    overall_passed = cameras_passed and audios_passed
    ...
```

`evaluate_strategy()` is a module-level helper in `report.py`.

**Final report schema:**

```json
{
  "report_id": "...",
  "source_file": "...",
  "minio_bucket": "...",
  "analyzed_at": "...",
  "duration_seconds": 12.3,
  "camera_pass_strategy": "all",
  "audio_pass_strategy": "all",
  "cameras": [
    {
      "topic": "/camera/front/image_raw",
      "frame_count": 60,
      "clarity": { "score": 0.85, "detail": {} },
      "continuity": { "score": 0.91, "detail": {} },
      "face": { "has_face": false },
      "gait": { "has_human_gait": false },
      "llm_assessment": null,
      "llm_skipped_reason": "no_sensitive_detection",
      "passed": true,
      "failure_reasons": [],
      "analyzer_errors": []
    }
  ],
  "audios": [
    {
      "topic": "/audio/front",
      "audio_frame_count": 400,
      "voice": { "has_human_voice": false },
      "llm_assessment": null,
      "llm_skipped_reason": "no_sensitive_detection",
      "passed": true,
      "failure_reasons": [],
      "analyzer_errors": []
    }
  ],
  "overall_passed": true,
  "failure_reasons": [],
  "analyzer_errors": []
}
```

### Layer 6: Config (`config.py`)

```python
camera_topics: list[str] = []                        # empty = auto-discover all image topics
audio_topics: list[str] = []                         # empty = auto-discover all audio topics
camera_pass_strategy: Literal["all", "any", "majority"] = "all"
audio_pass_strategy: Literal["all", "any", "majority"] = "all"
max_frames_per_topic: int = 300                      # hard upper bound per topic after frame_sample_rate
max_concurrent_topics: int = 4                       # ProcessPoolExecutor worker count for detection
llm_max_concurrent_calls: int = 4                    # ThreadPoolExecutor worker count for LLM
```

**Memory budget guidance:** peak detection-phase memory ≈ `max_concurrent_topics × max_frames_per_topic × frame_size`. For 1080p BGR (6 MB/frame): 4 × 300 × 6 MB ≈ 7.2 GB. Tune these two settings together to fit available RAM.

---

## Pass/Fail Strategy Logic

```python
def evaluate_strategy(results: list[bool], strategy: str) -> bool:
    if not results:
        # Zero topics of this type → failure (no silent pass)
        return False
    if strategy == "all":
        return all(results)
    elif strategy == "any":
        return any(results)
    elif strategy == "majority":
        return sum(results) > len(results) / 2
    raise ValueError(f"Unknown strategy: {strategy}")
```

`overall_passed` combines both media types:

```python
cameras_passed = evaluate_strategy([r["passed"] for r in camera_results], camera_pass_strategy)
audios_passed = evaluate_strategy([r["passed"] for r in audio_results], audio_pass_strategy)
overall_passed = cameras_passed and audios_passed
```

**Zero-topic behavior:** If an MCAP file has no image topics, `camera_results` is empty and `cameras_passed = False`. This is intentional — a recording with no usable camera data fails quality checks. The same applies to audio. If a specific media type is legitimately absent (e.g., camera-only robot), the user should configure `audio_pass_strategy` accordingly or the pipeline can be extended in a follow-up to support explicit "media type not expected" config.

---

## Error Handling

- Zero frames for a topic: skipped, topic marked `passed: false`, error recorded in that topic's `analyzer_errors`.
- Analyzer failure for a specific topic: topic marked `passed: false`, other topics unaffected.
- LLM failure for a specific topic: falls back to detector-only verdict (existing degradation behavior).
- All detector failures for a topic: topic marked `passed: false` (existing "null result = failure" rule applies per-topic).

---

## Migration Notes

- `ExtractedData.frames` and `ExtractedData.audio_frames` are **removed**. All analyzer unit tests constructing `ExtractedData` directly must be updated.
- Config fields `camera_topic` and `audio_topic` are **removed**. Existing `.env` files are silently ignored by Pydantic — users must update to list fields.
- `McapExtractor.__init__` now takes `camera_topics: list[str]` and `audio_topics: list[str]`.

---

## Files to Modify

| File | Change |
|------|--------|
| `agent/analyzers/base.py` | Replace `frames`/`audio_frames` with `videos`/`audios`; update `Analyzer` Protocol signatures; add `CameraResult`, `AudioResult` |
| `agent/analyzers/clarity.py` | `analyze(self, data: ExtractedData)` → `analyze(self, frames: list[np.ndarray])` |
| `agent/analyzers/continuity.py` | Same as clarity |
| `agent/analyzers/face.py` | Same as clarity |
| `agent/analyzers/gait.py` | Same as clarity |
| `agent/analyzers/voice.py` | `analyze(self, data: ExtractedData)` → `analyze(self, audio_frames: list[bytes])` |
| `agent/extractor.py` | Updated `__init__` signature; `_resolve_topics()` returns all topics; per-topic bucketing + sampling + min_frames |
| `agent/config.py` | Replace single-topic fields with list fields; add strategy fields and `llm_max_concurrent_calls` |
| `agent/pipeline.py` | Replace `run()` with `run_visual(frames)` and `run_audio(audio_frames)`; add top-level `_run_visual_worker` and `_run_audio_worker` picklable functions |
| `agent/runner.py` | Per-topic `ProcessPoolExecutor` detection phase; `ThreadPoolExecutor` LLM phase; final result assembly |
| `agent/llm_judge.py` | New `judge()` signature taking explicit `frames`/`audio_frames`; remove `data["frames"]` access |
| `agent/report.py` | New schema with `cameras`/`audios` arrays; `evaluate_strategy()`; strategy-based `overall_passed` |
| `tests/conftest.py` | Update `ExtractedData` fixtures to use `videos`/`audios` keys |
| `tests/` (all) | Update fixture construction, `build()` call assertions, per-topic result assertions |

---

## Test Migration

All tests constructing `ExtractedData` must change from:

```python
ExtractedData(frames=[...], audio_frames=[...], ...)
```

to:

```python
ExtractedData(videos={"/camera/image_raw": [...]}, audios={"/audio": [...]}, ...)
```

Tests for `ReportBuilder.build()` must assert on `report["cameras"][0]["clarity"]` instead of `report["scores"]["clarity"]`.

Analyzer unit tests (e.g., `test_clarity_analyzer`) become **simpler** — they now call `analyzer.analyze(frames)` directly with a plain list instead of constructing an `ExtractedData` wrapper. Any existing test that built `ExtractedData(frames=...)` to pass to an analyzer must be updated to pass the list directly.
