# MCAP Protocol Factory Design

**Date:** 2026-03-20
**Status:** Approved
**Scope:** `agent/extractor.py` + new `agent/mcap_codecs.py`

---

## Problem

`McapExtractor` hardcodes `mcap_ros2.reader.read_ros2_messages`, causing silent zero-score reports when given ROS1-encoded MCAP files. The current design also couples protocol decoding (wire format) with schema decoding (message structure) inside private methods, making both axes hard to extend.

Confirmed with a real ROS1 file (`20241203_demo_Office_PickPlace_ljw_152145.mcap`):
- 8 channels, 76 422 messages, ~6 min of data
- All `ros1` encoding, all `sensor_msgs/CompressedImage`
- None of the expected topics (`/camera/image_raw` etc.) exist
- Result: `frame_count=0`, all scores 0, report unreliable with no error surfaced

---

## Goals

- Support ROS1 and ROS2 MCAP files out of the box
- Handle both `sensor_msgs/Image` (raw) and `sensor_msgs/CompressedImage`
- Auto-detect encoding from file metadata; no caller changes required
- Leave clean extension points for future protocols and schemas
- Keep `McapExtractor.extract(path) → ExtractedData` interface unchanged

---

## Architecture: Two-Layer Factory

```
mcap_path
    │
    ▼
ProtocolReaderFactory.from_file(path)
    │  Reads file summary (index only, O(1))
    │  Selects DecoderFactory instances by encoding
    │  Returns mcap.McapReader (unified iterator)
    │
    ▼
reader.iter_decoded_messages(topics=[...])
    │  Yields (schema, channel, message, ros_msg)
    │  Protocol differences transparent to caller
    │
    ▼
SchemaDecoderRegistry
    │  Dispatches by schema.name
    │  sensor_msgs/Image           → _decode_raw_image
    │  sensor_msgs/CompressedImage → _decode_compressed_image
    │  sensor_msgs/Imu             → _decode_imu
    │  audio_common_msgs/AudioData → _decode_audio
    │
    ▼
ExtractedData (unchanged)
```

**Extending protocols:** add a `DecoderFactory` in `ProtocolReaderFactory._build_decoder_factories`
**Extending schemas:** call `registry.register_image("my_msgs/MyImage", fn)` before extraction

---

## File Structure

```
agent/
├── extractor.py      # McapExtractor — public API unchanged; delegates to mcap_codecs
└── mcap_codecs.py    # New: ProtocolReaderFactory + SchemaDecoderRegistry + decode fns

tests/
├── test_extractor.py      # Updated: mock ProtocolReaderFactory.from_file
└── test_mcap_codecs.py    # New: unit tests for registry and decode functions
```

---

## Component Specifications

### `ProtocolReaderFactory`

```python
class ProtocolReaderFactory:
    _SUPPORTED = {"ros1", "ros2", "cdr"}

    @classmethod
    def from_file(cls, path: str) -> McapReader:
        """Detect encodings, build decoder factories, return configured reader."""

    @classmethod
    def _detect_encodings(cls, path: str) -> set[str]:
        """Read summary index only (no message scan). Log warning for unsupported encodings."""

    @classmethod
    def _build_decoder_factories(cls, encodings: set[str]) -> list:
        """Lazily import mcap_ros1/mcap_ros2 DecoderFactory based on detected encodings."""
```

- `_detect_encodings` reads `summary.channels` only — O(1) index read, not a full message scan
- Unknown encodings emit `logger.warning` and are skipped (partial reads allowed)
- `from mcap_ros1.decoder import DecoderFactory` imported lazily to avoid hard dependency when not needed

### `SchemaDecoderRegistry`

```python
ImageDecodeFn = Callable[[Any], np.ndarray | None]
AudioDecodeFn = Callable[[Any], bytes]
ImuDecodeFn   = Callable[[Any], np.ndarray | None]

class SchemaDecoderRegistry:
    def register_image(self, schema: str, fn: ImageDecodeFn) -> None: ...
    def register_audio(self, schema: str, fn: AudioDecodeFn) -> None: ...
    def register_imu(self,   schema: str, fn: ImuDecodeFn)   -> None: ...

    def decode_image(self, schema: str, msg: Any) -> np.ndarray | None: ...
    def decode_audio(self, schema: str, msg: Any) -> bytes: ...
    def decode_imu(self,   schema: str, msg: Any) -> np.ndarray | None: ...
```

- Returns `None` / `b""` for unregistered schemas (no exception)
- `build_default_registry() -> SchemaDecoderRegistry` pre-registers all built-in decoders

### Built-in Decode Functions (module-level, pure)

| Function | Schema | Notes |
|---|---|---|
| `_decode_raw_image` | `sensor_msgs/Image` | Handles rgb/bgr encoding, reshapes buffer |
| `_decode_compressed_image` | `sensor_msgs/CompressedImage` | `cv2.imdecode` → BGR |
| `_decode_audio` | `audio_common_msgs/AudioData` | Returns raw bytes |
| `_decode_imu` | `sensor_msgs/Imu` | 6-element float64 array |

All catch `(AttributeError, ValueError)` and emit `logger.debug` on failure — consistent with Python robustness patterns.

### `McapExtractor` (refactored)

```python
class McapExtractor:
    def __init__(
        self,
        camera_topic: str = "/camera/image_raw",
        audio_topic:  str = "/audio/data",
        imu_topic:    str = "/imu/data",
        frame_sample_rate: int = 1,
        registry: SchemaDecoderRegistry | None = None,  # injection point
    ) -> None: ...

    def extract(self, mcap_path: str) -> ExtractedData: ...
```

- `registry` parameter enables test injection of a minimal registry without real MCAP files
- Internal `_decode_image/_decode_audio/_decode_imu` private methods removed; logic lives in `mcap_codecs.py`
- Frame sampling, PCM chunking, duration calculation logic unchanged

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Unknown encoding in file | `logger.warning`, skip those channels, continue |
| Registered schema but malformed message | `logger.debug`, return `None`/`b""`, skip message |
| Unregistered schema on matched topic | Return `None`/`b""`, no log (expected for unknown schemas) |
| File not found / corrupt header | Exception propagates to `runner.analyze_local_file`, logged there |

---

## Testing

### `test_mcap_codecs.py` (new)

- `_decode_compressed_image`: encode a synthetic JPEG, verify BGR shape returned
- `_decode_raw_image`: construct minimal `SimpleNamespace` with height/width/data/encoding
- `SchemaDecoderRegistry`: unknown schema returns `None`; custom decoder registered and called
- `ProtocolReaderFactory._detect_encodings`: mock `make_reader` summary with known encodings

### `test_extractor.py` (updated)

- Replace `read_ros2_messages` mock with `ProtocolReaderFactory.from_file` mock
- Mock reader yields `(schema, channel, message, ros_msg)` 4-tuples
- Frame sampling, PCM chunking, duration tests: update mock layer only, logic assertions unchanged
- Inject minimal `SchemaDecoderRegistry` with only needed decoders per test

---

## Dependencies

No new packages required. `mcap_ros1` decoder is already present via `mcap-ros2-support` (the package bundles both). Lazy import avoids errors if ever run without it.
