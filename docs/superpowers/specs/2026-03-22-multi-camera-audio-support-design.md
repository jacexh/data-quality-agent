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
3. LLM assessment invoked independently per topic (concurrent).
4. Report structured as `cameras: [...]` and `audios: [...]` arrays, unified format for both single- and multi-stream files.
5. Overall `passed` determined by a configurable strategy (`all` / `any` / `majority`), independently configurable for cameras and audios.

---

## Non-Goals

- Cross-camera comparison or synchronization analysis.
- Merging frames across cameras before analysis.
- Changing the Analyzer Protocol signatures (they remain single-stream).

---

## Architecture

### Layer 1: Data Structures (`analyzers/base.py`)

Replace flat `frames` and `audio_frames` fields with topic-keyed dicts:

```python
class ExtractedData(TypedDict):
    videos: dict[str, list[np.ndarray]]   # topic → BGR frame list
    audios: dict[str, list[bytes]]         # topic → 30ms PCM frame list
    sensor_series: dict[str, np.ndarray]   # unchanged
    duration_seconds: float
```

Add result containers for per-topic report output:

```python
class CameraResult(TypedDict):
    topic: str
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
    voice: VoiceResult
    llm_assessment: dict | None
    llm_skipped_reason: str | None
    passed: bool
    failure_reasons: list[str]
    analyzer_errors: list[str]
```

### Layer 2: Extractor (`extractor.py`)

**`_resolve_topics()`** returns all discovered image and audio topics:

```python
def _resolve_topics(self) -> tuple[list[str], list[str]]:
    # Returns (video_topics, audio_topics)
    # Config-specified topics are prioritized (placed first) if present in file
    # Remaining auto-discovered topics are appended
```

Frame extraction buckets messages by topic:

```python
videos: dict[str, list[np.ndarray]] = {t: [] for t in video_topics}
audios: dict[str, list[bytes]] = {t: [] for t in audio_topics}

for schema, channel, message in reader.iter_messages(topics=all_topics):
    if channel.topic in videos:
        videos[channel.topic].append(decode_image(message))
    elif channel.topic in audios:
        audios[channel.topic].append(decode_audio(message))
```

**Config migration:**

| Old field | New field | Notes |
|-----------|-----------|-------|
| `camera_topic: str` | `camera_topics: list[str]` | Empty = auto-discover all |
| `audio_topic: str` | `audio_topics: list[str]` | Empty = auto-discover all |

Old single-value fields are removed. Users with a single configured topic simply provide a one-element list.

### Layer 3: Pipeline (`pipeline.py` + `runner.py`)

Analyzer signatures remain unchanged — each analyzer receives a single stream's frames and returns a single result. The pipeline routes per-topic in the runner:

```python
# runner.py (analyze_local_file)
camera_results = {}
for topic, frames in data["videos"].items():
    single_extracted = ExtractedData(videos={topic: frames}, ...)
    results, errors = pipeline.run(single_extracted)
    camera_results[topic] = (results, errors)

audio_results = {}
for topic, audio_frames in data["audios"].items():
    voice_result = voice_analyzer.analyze(audio_frames)
    audio_results[topic] = voice_result
```

`AnalysisPipeline.run()` internally reads `data["videos"]` — since it receives a single-topic dict, existing per-frame logic is unchanged.

### Layer 4: LLM Judge (`llm_judge.py`)

Each topic that meets the LLM trigger condition (sensitive detection or score within threshold ±0.1) gets an independent `LLMJudge` call. Camera topics run concurrently via `ThreadPoolExecutor` (reusing the existing thread pool pattern). Audio topics run similarly.

- `get_key_frames` tool scopes frame sampling to the specific topic's frame list.
- LLM verdict is per-topic: `{ "passed": bool, "reason": str }`.

### Layer 5: Report (`report.py`)

Final report schema:

```json
{
  "report_id": "...",
  "source_file": "...",
  "minio_bucket": "...",
  "analyzed_at": "...",
  "duration_seconds": 12.3,
  "cameras": [
    {
      "topic": "/camera/front/image_raw",
      "clarity": { "score": 0.85, "detail": {...} },
      "continuity": { "score": 0.91, "detail": {...} },
      "face": { "has_face": false, ... },
      "gait": { "has_human_gait": false, ... },
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
      "voice": { "has_human_voice": false, ... },
      "llm_assessment": null,
      "llm_skipped_reason": "no_sensitive_detection",
      "passed": true,
      "failure_reasons": [],
      "analyzer_errors": []
    }
  ],
  "overall_passed": true,
  "failure_reasons": [],
  "analyzer_errors": [],
  "camera_pass_strategy": "all",
  "audio_pass_strategy": "all"
}
```

`overall_passed` is `true` only when both all cameras and all audios pass (each evaluated by their respective strategy).

### Layer 6: Config (`config.py`)

```python
camera_topics: list[str] = []          # empty = auto-discover all image topics
audio_topics: list[str] = []           # empty = auto-discover all audio topics
camera_pass_strategy: Literal["all", "any", "majority"] = "all"
audio_pass_strategy: Literal["all", "any", "majority"] = "all"
```

---

## Pass/Fail Strategy Logic

```python
def evaluate_strategy(results: list[bool], strategy: str) -> bool:
    if strategy == "all":
        return all(results)
    elif strategy == "any":
        return any(results)
    elif strategy == "majority":
        return sum(results) > len(results) / 2
```

Applied independently to camera results and audio results. `overall_passed = cameras_passed and audios_passed`.

---

## Error Handling

- If a topic has zero frames after extraction, it is skipped and logged to `analyzer_errors` at the topic level.
- If an analyzer fails for a specific topic, that topic is marked `passed: false` with the error in its `analyzer_errors`; other topics are unaffected.
- If LLM fails for a specific topic, that topic falls back to detector-only verdict (existing degradation behavior).

---

## Migration Notes

- `ExtractedData.frames` and `ExtractedData.audio_frames` are removed. Any code referencing them will fail at import time (no silent regression).
- Config fields `camera_topic` and `audio_topic` are removed. Existing `.env` files referencing these will be ignored by Pydantic (unknown fields), so no crash — but users should update to `camera_topics` / `audio_topics`.
- All tests referencing `data["frames"]` must be updated to `data["videos"]["<topic>"]`.

---

## Files to Modify

| File | Change |
|------|--------|
| `agent/analyzers/base.py` | Replace `frames`/`audio_frames` with `videos`/`audios`; add `CameraResult`, `AudioResult` |
| `agent/extractor.py` | `_resolve_topics()` returns all topics; frame bucketing by topic |
| `agent/config.py` | Replace single-topic fields with list fields + strategy fields |
| `agent/pipeline.py` | Internal read updated to `data["videos"]` single-topic dict |
| `agent/runner.py` | Per-topic loop over cameras and audios |
| `agent/llm_judge.py` | Per-topic LLM calls, scoped frame sampling |
| `agent/report.py` | New schema with `cameras`/`audios` arrays, strategy-based `overall_passed` |
| `agent/analyzers/*.py` | Update `data["frames"]` → `data["videos"][topic]` references if any |
| `tests/` | Update fixtures and assertions to new schema |
