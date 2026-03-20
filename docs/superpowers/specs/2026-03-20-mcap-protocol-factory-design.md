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
    │  Yields (schema, channel, message, decoded_message)
    │  Protocol differences transparent to caller
    │  DecoderNotFoundError caught per-message → logger.warning + skip
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
    _SUPPORTED = {"ros1", "ros2"}

    @classmethod
    def build_decoder_factories(cls, path: str) -> list:
        """Detect encodings from file metadata, return list of DecoderFactory instances."""

    @classmethod
    def _detect_encodings(cls, path: str) -> set[str]:
        """Read summary index only (no message scan). Log warning for unsupported encodings.
        Falls back to scanning Channel records if get_summary() returns None (unindexed files)."""

    @classmethod
    def _build_factories_for(cls, encodings: set[str]) -> list:
        """Lazily import mcap_ros1/mcap_ros2 DecoderFactory based on detected encodings."""
```

- `"cdr"` removed from `_SUPPORTED` — CDR is not a decodable protocol in this version; files
  with CDR-encoded channels will emit `logger.warning` like any other unsupported encoding
- `_detect_encodings` reads `summary.channels` — O(1) index read, not a full message scan
- If `get_summary()` returns `None` (unindexed / streaming-written file), fall back to scanning
  `iter_messages()` for `Channel` records using a **separate file handle** — the `NonSeekingReader`
  used for unindexed files is single-use (`_spent` flag), so the scan and main read must not share a handle
- Unknown encodings emit `logger.warning` and are excluded from the supported set
- `from mcap_ros1.decoder import DecoderFactory` and `from mcap_ros2.decoder import DecoderFactory`
  imported lazily — both available via `mcap-ros1-support` and `mcap-ros2-support`

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
- `decode_audio` returns `bytes` (never `None`) — empty bytes are safe for the PCM pipeline;
  `decode_image` and `decode_imu` return `T | None` because `None` is the skip signal downstream
- `build_default_registry() -> SchemaDecoderRegistry` pre-registers all built-in decoders

### Built-in Decode Functions (module-level, pure)

| Function | Schema | Notes |
|---|---|---|
| `_decode_raw_image` | `sensor_msgs/Image` | Handles rgb/bgr encoding, reshapes buffer; mono8 not supported (logs debug) |
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

**File handle and `DecoderNotFoundError` handling in `extract`:**

`extract` owns the file handle and the safe-iteration adapter. `DecoderNotFoundError` is raised
inside the iterator's `__next__()` (inside `decoded_message()` closure in `mcap/reader.py`), so it
cannot be caught inside the `for` loop body. A generator adapter is required:

```python
from mcap.exceptions import DecoderNotFoundError

def _safe_iter(reader, topics):
    """Wrap iter_decoded_messages to skip individual messages that lack a decoder."""
    it = reader.iter_decoded_messages(topics=topics)
    while True:
        try:
            yield next(it)
        except DecoderNotFoundError as e:
            logger.warning("No decoder for message encoding, skipping: {}", e)
        except StopIteration:
            return

def extract(self, mcap_path: str) -> ExtractedData:
    decoder_factories = ProtocolReaderFactory.build_decoder_factories(mcap_path)
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=decoder_factories)
        for schema, channel, message, decoded_message in _safe_iter(reader, topics):
            ...
```

`extract` itself opens the file with `with open(...)` so the handle is always closed. The factory
method `build_decoder_factories` reads the file separately for encoding detection and returns only
the factory list — it does not hold a file handle.

Note: the fourth tuple field is `decoded_message` (the `DecodedMessageTuple` namedtuple field),
not `ros_msg`. Test stubs must yield 4-tuples with `decoded_message` in position 3.

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Unknown encoding detected at summary read | `logger.warning`, excluded from factory set |
| `get_summary()` returns None (unindexed file) | Fall back to `iter_messages()` Channel scan via separate file handle |
| `DecoderNotFoundError` during iteration | `logger.warning` per message via `_safe_iter` adapter, skip and continue |
| Registered schema but malformed message | `logger.debug`, return `None`/`b""`, skip message |
| Unregistered schema on matched topic | Return `None`/`b""`, no log (expected for unknown schemas) |
| File not found / corrupt header | Exception propagates to `runner.analyze_local_file`, logged there |

---

## Testing

### `test_mcap_codecs.py` (new)

- `_decode_compressed_image`: encode a synthetic JPEG, verify BGR shape returned
- `_decode_raw_image`: construct minimal `SimpleNamespace` with height/width/data/encoding
- `SchemaDecoderRegistry`: unknown schema returns `None`; custom decoder registered and called
- `ProtocolReaderFactory._detect_encodings`: mock `make_reader` summary with known encodings;
  also test `None` summary fallback path

### `test_extractor.py` (updated)

- Replace `read_ros2_messages` mock with `ProtocolReaderFactory.from_file` mock
- Mock reader's `iter_decoded_messages` yields `(schema, channel, message, decoded_message)` 4-tuples
  where `decoded_message` is a `SimpleNamespace` with the fields each decode function expects
- Frame sampling, PCM chunking, duration tests: update mock layer only, logic assertions unchanged
- Inject minimal `SchemaDecoderRegistry` with only needed decoders per test

---

## Dependencies

`mcap-ros1-support>=0.4` added to `pyproject.toml` (previously missing; `mcap_ros1` is a separate
package from `mcap-ros2-support`). Both packages are lazily imported inside
`_build_decoder_factories` so an ImportError at that point surfaces a clear message rather than a
top-level import failure.
