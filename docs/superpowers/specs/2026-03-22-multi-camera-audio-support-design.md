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
    sensor_series: dict[str, np.ndarray]   # unchanged
    duration_seconds: float
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

# Post-loop: apply frame_sample_rate and min_frames per topic
for topic in list(videos):
    videos[topic] = videos[topic][::self._frame_sample_rate]
    if len(videos[topic]) < self._min_frames:
        logger.warning(f"Topic {topic} has only {len(videos[topic])} frames after sampling — treated as empty")
        videos[topic] = []  # triggers zero-frames failure path in runner
```

Audio topics are not sub-sampled (PCM frame rate is already fixed at 30ms).

**Config migration:**

| Old field | New field | Notes |
|-----------|-----------|-------|
| `camera_topic: str` | `camera_topics: list[str]` | Empty = auto-discover all |
| `audio_topic: str` | `audio_topics: list[str]` | Empty = auto-discover all |

### Layer 3: Pipeline (`pipeline.py` + `runner.py`)

Analyzer signatures are **unchanged** — each analyzer receives a plain `list[np.ndarray]` (for visual) or `list[bytes]` (for audio). The routing happens in `runner.py`:

```python
# runner.py (analyze_local_file)
camera_results: list[CameraResult] = []
for topic, frames in data["videos"].items():
    if not frames:
        # Log and skip; topic recorded as failed with error "zero frames"
        camera_results.append(_empty_camera_result(topic, "zero_frames"))
        continue
    results, errors = pipeline.run_visual(frames)   # pipeline extracts frames list
    camera_results.append(_build_camera_result(topic, frames, results, errors))

audio_results: list[AudioResult] = []
for topic, audio_frames in data["audios"].items():
    if not audio_frames:
        audio_results.append(_empty_audio_result(topic, "zero_frames"))
        continue
    voice_result, errors = pipeline.run_audio(audio_frames)
    audio_results.append(_build_audio_result(topic, audio_frames, voice_result, errors))
```

`AnalysisPipeline` gains two methods:

```python
def run_visual(self, frames: list[np.ndarray]) -> tuple[dict[str, Any], list[str]]:
    # Runs clarity, continuity, face, gait in ThreadPoolExecutor
    # Each analyzer receives frames directly (not ExtractedData)

def run_audio(self, audio_frames: list[bytes]) -> tuple[VoiceResult, list[str]]:
    # Runs voice detector with audio_frames directly
```

This keeps `ThreadPoolExecutor`-based error isolation for both media types.

### Layer 4: LLM Judge (`llm_judge.py`)

**Signature change:**

```python
class LLMJudge:
    def judge(
        self,
        topic: str,
        detector_results: dict[str, Any],
        frames: list[np.ndarray] | None,   # None for audio-only topics
        audio_frames: list[bytes] | None,  # None for camera-only topics
    ) -> dict[str, Any]:
        ...
```

`_run_agent` accesses frames directly from the parameter (no longer reads `data["frames"]`). `get_key_frames` tool samples from the passed `frames` list.

**Concurrency:** Per-topic LLM calls are dispatched via `ThreadPoolExecutor` with a cap of `settings.llm_max_concurrent_calls` (new config field, default 4). One shared `LLMJudge` instance is reused across topics (it is stateless).

The runner uses an intermediate dict during the detection phase to carry frames alongside results, then runs LLM concurrently:

```python
# runner.py — detection phase builds intermediate list (not CameraResult yet)
camera_intermediates: list[dict] = []
for topic, frames in data["videos"].items():
    if not frames:
        camera_intermediates.append({"topic": topic, "frames": [], "results": {}, "errors": ["zero_frames"], "needs_llm": False})
        continue
    results, errors = pipeline.run_visual(frames)
    camera_intermediates.append({
        "topic": topic,
        "frames": frames,
        "results": results,
        "errors": errors,
        "needs_llm": should_invoke_llm(results),  # existing function from llm_judge.py
        "llm_result": None,
        "llm_skipped_reason": None,
    })

# runner.py — concurrent LLM phase
with ThreadPoolExecutor(max_workers=settings.llm_max_concurrent_calls) as executor:
    futures = {
        executor.submit(judge.judge, item["topic"], item["results"], item["frames"], None): item
        for item in camera_intermediates if item["needs_llm"]
    }
    for future, item in futures.items():
        item["llm_result"] = future.result()

# runner.py — build final CameraResult list
camera_results = [_build_camera_result(item) for item in camera_intermediates]
# similarly for audio_intermediates / audio_results
```

`should_invoke_llm()` is the existing function from `llm_judge.py` — no new helper needed.

### Layer 5: Report (`report.py`)

Final report schema:

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
llm_max_concurrent_calls: int = 4                    # cap on concurrent LLM API calls
```

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
| `agent/pipeline.py` | Replace `run()` with `run_visual(frames)` and `run_audio(audio_frames)` |
| `agent/runner.py` | Per-topic intermediate loop; concurrent LLM phase with bounded thread pool; final result assembly |
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
