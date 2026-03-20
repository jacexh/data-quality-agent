# MCAP Protocol Factory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hardcoded `mcap_ros2` reader in `McapExtractor` with a two-layer factory that auto-detects ROS1/ROS2 encoding and dispatches by schema name, so both `sensor_msgs/Image` and `sensor_msgs/CompressedImage` are handled correctly.

**Architecture:** New `agent/mcap_codecs.py` provides `ProtocolReaderFactory` (builds mcap `DecoderFactory` instances from file metadata) and `SchemaDecoderRegistry` (dispatches raw decoded messages to typed numpy/bytes by schema name). `McapExtractor.extract()` delegates both responsibilities to these classes; private `_decode_*` methods are removed.

**Tech Stack:** `mcap>=1.1`, `mcap-ros1-support>=0.4`, `mcap-ros2-support>=0.5`, `opencv-python-headless>=4.10`, `numpy>=1.26`, `loguru>=0.7`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `agent/mcap_codecs.py` | `ProtocolReaderFactory`, `SchemaDecoderRegistry`, `_safe_iter`, 4 decode fns, `build_default_registry()` |
| Modify | `agent/extractor.py` | Remove private `_decode_*` methods; inject registry; use `ProtocolReaderFactory` + `make_reader` |
| Create | `tests/test_mcap_codecs.py` | Unit tests for registry, decode fns, factory encoding detection |
| Modify | `tests/test_extractor.py` | Replace `read_ros2_messages` mock with `ProtocolReaderFactory.build_decoder_factories` mock; update 4-tuple message shape |

---

## Task 1: SchemaDecoderRegistry + Built-in Decode Functions

**Files:**
- Create: `agent/mcap_codecs.py`
- Create: `tests/test_mcap_codecs.py`

### Step 1.1: Write failing tests for decode functions and registry

- [ ] Create `tests/test_mcap_codecs.py` with the following content:

```python
# tests/test_mcap_codecs.py
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from agent.mcap_codecs import (
    SchemaDecoderRegistry,
    build_default_registry,
    _decode_raw_image,
    _decode_compressed_image,
    _decode_audio,
    _decode_imu,
)


# ── _decode_raw_image ────────────────────────────────────────────────────────

def test_decode_raw_image_bgr8_returns_ndarray():
    msg = SimpleNamespace(height=4, width=4, encoding="bgr8", data=bytes([100] * 48))
    result = _decode_raw_image(msg)
    assert result is not None
    assert result.shape == (4, 4, 3)


def test_decode_raw_image_rgb8_converts_to_bgr():
    """rgb8 images are converted to BGR so downstream code is consistent."""
    msg = SimpleNamespace(
        height=2, width=2, encoding="rgb8",
        data=bytes([255, 0, 0,   255, 0, 0,   255, 0, 0,   255, 0, 0]),  # pure red in RGB
    )
    result = _decode_raw_image(msg)
    assert result is not None
    # After RGB→BGR conversion, pure red pixel becomes (0, 0, 255) in BGR
    assert result[0, 0, 0] == 0    # B
    assert result[0, 0, 2] == 255  # R


def test_decode_raw_image_bad_shape_returns_none():
    """Buffer size mismatch → None, not an exception."""
    msg = SimpleNamespace(height=100, width=100, encoding="bgr8", data=b"\x00")
    assert _decode_raw_image(msg) is None


def test_decode_raw_image_missing_attr_returns_none():
    assert _decode_raw_image(SimpleNamespace()) is None


# ── _decode_compressed_image ─────────────────────────────────────────────────

def test_decode_compressed_image_jpeg_returns_bgr():
    # Encode a small synthetic image to JPEG bytes
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[0, 0] = [200, 100, 50]
    _, buf = cv2.imencode(".jpg", img)
    msg = SimpleNamespace(data=bytes(buf))
    result = _decode_compressed_image(msg)
    assert result is not None
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_decode_compressed_image_bad_bytes_returns_none():
    msg = SimpleNamespace(data=b"not an image")
    assert _decode_compressed_image(msg) is None


def test_decode_compressed_image_missing_attr_returns_none():
    assert _decode_compressed_image(SimpleNamespace()) is None


# ── _decode_audio ────────────────────────────────────────────────────────────

def test_decode_audio_returns_bytes():
    msg = SimpleNamespace(data=b"\x01\x02\x03")
    assert _decode_audio(msg) == b"\x01\x02\x03"


def test_decode_audio_missing_attr_returns_empty_bytes():
    assert _decode_audio(SimpleNamespace()) == b""


# ── _decode_imu ──────────────────────────────────────────────────────────────

def test_decode_imu_returns_6_element_array():
    acc = SimpleNamespace(x=1.0, y=2.0, z=3.0)
    gyro = SimpleNamespace(x=4.0, y=5.0, z=6.0)
    msg = SimpleNamespace(linear_acceleration=acc, angular_velocity=gyro)
    result = _decode_imu(msg)
    assert result is not None
    assert result.shape == (6,)
    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_decode_imu_missing_attr_returns_none():
    assert _decode_imu(SimpleNamespace()) is None


# ── SchemaDecoderRegistry ────────────────────────────────────────────────────

def test_registry_unknown_image_schema_returns_none():
    reg = SchemaDecoderRegistry()
    assert reg.decode_image("unknown/Schema", object()) is None


def test_registry_unknown_audio_schema_returns_empty_bytes():
    reg = SchemaDecoderRegistry()
    assert reg.decode_audio("unknown/Schema", object()) == b""


def test_registry_unknown_imu_schema_returns_none():
    reg = SchemaDecoderRegistry()
    assert reg.decode_imu("unknown/Schema", object()) is None


def test_registry_custom_image_decoder_called():
    reg = SchemaDecoderRegistry()
    sentinel = np.zeros((2, 2, 3), dtype=np.uint8)
    reg.register_image("my_msgs/MyImage", lambda msg: sentinel)
    result = reg.decode_image("my_msgs/MyImage", object())
    assert result is sentinel


def test_registry_custom_audio_decoder_called():
    reg = SchemaDecoderRegistry()
    reg.register_audio("my_msgs/Audio", lambda msg: b"\xff")
    assert reg.decode_audio("my_msgs/Audio", object()) == b"\xff"


# ── build_default_registry ───────────────────────────────────────────────────

def test_build_default_registry_handles_sensor_msgs_image():
    reg = build_default_registry()
    msg = SimpleNamespace(height=2, width=2, encoding="bgr8", data=bytes([0] * 12))
    result = reg.decode_image("sensor_msgs/Image", msg)
    assert result is not None


def test_build_default_registry_handles_compressed_image():
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    reg = build_default_registry()
    msg = SimpleNamespace(data=bytes(buf))
    result = reg.decode_image("sensor_msgs/CompressedImage", msg)
    assert result is not None


def test_build_default_registry_handles_imu():
    reg = build_default_registry()
    acc = SimpleNamespace(x=0.1, y=0.2, z=9.8)
    gyro = SimpleNamespace(x=0.0, y=0.0, z=0.1)
    msg = SimpleNamespace(linear_acceleration=acc, angular_velocity=gyro)
    result = reg.decode_imu("sensor_msgs/Imu", msg)
    assert result is not None
    assert result.shape == (6,)


def test_build_default_registry_handles_audio():
    reg = build_default_registry()
    msg = SimpleNamespace(data=b"\x00\x01")
    assert reg.decode_audio("audio_common_msgs/AudioData", msg) == b"\x00\x01"
```

- [ ] Run tests to verify they fail (module doesn't exist yet):

```bash
uv run pytest tests/test_mcap_codecs.py -v
```

Expected: `ModuleNotFoundError: No module named 'agent.mcap_codecs'`

### Step 1.2: Implement decode functions and SchemaDecoderRegistry

- [ ] Create `agent/mcap_codecs.py` with this content:

```python
# agent/mcap_codecs.py
from __future__ import annotations
from typing import Any, Callable
import numpy as np
from loguru import logger

ImageDecodeFn = Callable[[Any], "np.ndarray | None"]
AudioDecodeFn = Callable[[Any], bytes]
ImuDecodeFn   = Callable[[Any], "np.ndarray | None"]


# ── Built-in decode functions (module-level, pure) ───────────────────────────

def _decode_raw_image(msg: Any) -> np.ndarray | None:
    """Decode sensor_msgs/Image → BGR ndarray. Returns None on failure."""
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
    except (AttributeError, ValueError) as e:
        logger.debug("_decode_raw_image failed: {}", e)
        return None


def _decode_compressed_image(msg: Any) -> np.ndarray | None:
    """Decode sensor_msgs/CompressedImage (JPEG/PNG) → BGR ndarray. Returns None on failure."""
    try:
        import cv2
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if arr is None:
            logger.debug("_decode_compressed_image: cv2.imdecode returned None")
        return arr
    except (AttributeError, ValueError) as e:
        logger.debug("_decode_compressed_image failed: {}", e)
        return None


def _decode_audio(msg: Any) -> bytes:
    """Decode audio_common_msgs/AudioData → raw bytes. Returns b'' on failure."""
    try:
        return bytes(msg.data)
    except AttributeError as e:
        logger.debug("_decode_audio failed: {}", e)
        return b""


def _decode_imu(msg: Any) -> np.ndarray | None:
    """Decode sensor_msgs/Imu → 6-element float64 array [ax,ay,az,gx,gy,gz]. Returns None on failure."""
    try:
        a = msg.linear_acceleration
        g = msg.angular_velocity
        return np.array([a.x, a.y, a.z, g.x, g.y, g.z], dtype=np.float64)
    except AttributeError as e:
        logger.debug("_decode_imu failed: {}", e)
        return None


# ── SchemaDecoderRegistry ────────────────────────────────────────────────────

class SchemaDecoderRegistry:
    """Dispatches decoded messages to typed values by schema name."""

    def __init__(self) -> None:
        self._image: dict[str, ImageDecodeFn] = {}
        self._audio: dict[str, AudioDecodeFn] = {}
        self._imu:   dict[str, ImuDecodeFn]   = {}

    def register_image(self, schema: str, fn: ImageDecodeFn) -> None:
        self._image[schema] = fn

    def register_audio(self, schema: str, fn: AudioDecodeFn) -> None:
        self._audio[schema] = fn

    def register_imu(self, schema: str, fn: ImuDecodeFn) -> None:
        self._imu[schema] = fn

    def decode_image(self, schema: str, msg: Any) -> np.ndarray | None:
        fn = self._image.get(schema)
        return fn(msg) if fn is not None else None

    def decode_audio(self, schema: str, msg: Any) -> bytes:
        fn = self._audio.get(schema)
        return fn(msg) if fn is not None else b""

    def decode_imu(self, schema: str, msg: Any) -> np.ndarray | None:
        fn = self._imu.get(schema)
        return fn(msg) if fn is not None else None


def build_default_registry() -> SchemaDecoderRegistry:
    """Return a registry pre-loaded with all built-in decoders."""
    reg = SchemaDecoderRegistry()
    reg.register_image("sensor_msgs/Image",            _decode_raw_image)
    reg.register_image("sensor_msgs/CompressedImage",  _decode_compressed_image)
    reg.register_audio("audio_common_msgs/AudioData",  _decode_audio)
    reg.register_imu("sensor_msgs/Imu",                _decode_imu)
    return reg
```

- [ ] Run tests to verify they pass:

```bash
uv run pytest tests/test_mcap_codecs.py -v
```

Expected: all tests PASS

- [ ] Commit:

```bash
git add agent/mcap_codecs.py tests/test_mcap_codecs.py
git commit -m "feat(mcap_codecs): add SchemaDecoderRegistry and built-in decode functions"
```

---

## Task 2: ProtocolReaderFactory

**Files:**
- Modify: `agent/mcap_codecs.py`
- Modify: `tests/test_mcap_codecs.py`

### Step 2.1: Write failing tests for ProtocolReaderFactory

- [ ] Append the following to `tests/test_mcap_codecs.py`:

```python
# ── ProtocolReaderFactory ─────────────────────────────────────────────────────

from unittest.mock import patch, MagicMock


def _make_summary(encodings: list[str]):
    """Build a minimal mcap Summary mock with one channel per encoding."""
    channels = {}
    for i, enc in enumerate(encodings):
        ch = MagicMock()
        ch.metadata = {"encoding": enc}
        channels[i] = ch
    summary = MagicMock()
    summary.channels = channels
    return summary


def test_detect_encodings_ros1():
    from agent.mcap_codecs import ProtocolReaderFactory
    summary = _make_summary(["ros1msg"])
    mock_reader = MagicMock()
    mock_reader.get_summary.return_value = summary
    with patch("agent.mcap_codecs.make_reader", return_value=mock_reader):
        encodings = ProtocolReaderFactory._detect_encodings("fake.mcap")
    assert "ros1" in encodings


def test_detect_encodings_ros2():
    from agent.mcap_codecs import ProtocolReaderFactory
    summary = _make_summary(["cdr"])
    mock_reader = MagicMock()
    mock_reader.get_summary.return_value = summary
    with patch("agent.mcap_codecs.make_reader", return_value=mock_reader):
        encodings = ProtocolReaderFactory._detect_encodings("fake.mcap")
    assert "ros2" in encodings


def test_detect_encodings_unknown_emits_warning(caplog):
    import logging
    from agent.mcap_codecs import ProtocolReaderFactory
    summary = _make_summary(["exotic_proto"])
    mock_reader = MagicMock()
    mock_reader.get_summary.return_value = summary
    with patch("agent.mcap_codecs.make_reader", return_value=mock_reader):
        with patch("agent.mcap_codecs.logger") as mock_logger:
            ProtocolReaderFactory._detect_encodings("fake.mcap")
            mock_logger.warning.assert_called_once()


def test_detect_encodings_none_summary_fallback():
    """When get_summary() returns None, fall back to iter_messages() for Channel records."""
    from agent.mcap_codecs import ProtocolReaderFactory
    from mcap.records import Channel

    channel = MagicMock(spec=Channel)
    channel.metadata = {"encoding": "ros1msg"}

    mock_reader = MagicMock()
    mock_reader.get_summary.return_value = None
    mock_reader.iter_messages.return_value = iter([MagicMock(channel=channel)])

    with patch("agent.mcap_codecs.make_reader", return_value=mock_reader):
        encodings = ProtocolReaderFactory._detect_encodings("fake.mcap")

    assert "ros1" in encodings


def test_build_decoder_factories_returns_list():
    """build_decoder_factories returns a non-empty list for known encodings."""
    from agent.mcap_codecs import ProtocolReaderFactory
    with patch.object(ProtocolReaderFactory, "_detect_encodings", return_value={"ros2"}):
        factories = ProtocolReaderFactory.build_decoder_factories("fake.mcap")
    assert isinstance(factories, list)
    assert len(factories) >= 1


def test_build_decoder_factories_empty_for_no_known_encodings():
    from agent.mcap_codecs import ProtocolReaderFactory
    with patch.object(ProtocolReaderFactory, "_detect_encodings", return_value=set()):
        factories = ProtocolReaderFactory.build_decoder_factories("fake.mcap")
    assert factories == []
```

- [ ] Run tests to verify they fail:

```bash
uv run pytest tests/test_mcap_codecs.py -k "ProtocolReader or detect_encodings or build_decoder_factories" -v
```

Expected: `ImportError: cannot import name 'ProtocolReaderFactory'`

### Step 2.2: Implement ProtocolReaderFactory

- [ ] Append the following to `agent/mcap_codecs.py` (after `build_default_registry`):

```python
# ── ProtocolReaderFactory ────────────────────────────────────────────────────

from mcap.reader import make_reader  # noqa: E402 — placed after logger setup


# Encoding strings found in MCAP channel metadata → logical protocol name
_ENCODING_MAP: dict[str, str] = {
    "ros1msg": "ros1",
    "cdr":     "ros2",
}

_SUPPORTED: set[str] = {"ros1", "ros2"}


class ProtocolReaderFactory:
    """Detect MCAP file encoding and build matching mcap DecoderFactory instances."""

    @classmethod
    def build_decoder_factories(cls, path: str) -> list:
        """Read file metadata, return a list of DecoderFactory instances.

        Opens and closes a file handle internally — does not retain one.
        Returns [] if no supported encodings are found.
        """
        encodings = cls._detect_encodings(path)
        return cls._build_factories_for(encodings)

    @classmethod
    def _detect_encodings(cls, path: str) -> set[str]:
        """Return the set of logical protocol names present in the file.

        Reads summary index only (O(1)). Falls back to iter_messages() for
        unindexed files where get_summary() returns None.
        """
        with open(path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()

            if summary is not None:
                raw_encodings = {
                    ch.metadata.get("encoding", "")
                    for ch in summary.channels.values()
                }
            else:
                # Unindexed / streaming-written file — scan Channel records
                # using a separate file handle (NonSeekingReader is single-use)
                raw_encodings = set()
                for item in reader.iter_messages():
                    enc = item.channel.metadata.get("encoding", "")
                    if enc:
                        raw_encodings.add(enc)

        protocols: set[str] = set()
        for enc in raw_encodings:
            protocol = _ENCODING_MAP.get(enc)
            if protocol in _SUPPORTED:
                protocols.add(protocol)
            elif enc:
                logger.warning("Unsupported MCAP encoding {:!r}, skipping", enc)

        return protocols

    @classmethod
    def _build_factories_for(cls, encodings: set[str]) -> list:
        """Lazily import and instantiate DecoderFactory for each known protocol."""
        factories = []
        if "ros1" in encodings:
            from mcap_ros1.decoder import DecoderFactory as Ros1Factory  # type: ignore[import]
            factories.append(Ros1Factory())
        if "ros2" in encodings:
            from mcap_ros2.decoder import DecoderFactory as Ros2Factory
            factories.append(Ros2Factory())
        return factories
```

> **Note:** The `from mcap.reader import make_reader` import needs to move to the top of the file (after the existing imports). Move it there while adding this class.

- [ ] Run tests to verify they pass:

```bash
uv run pytest tests/test_mcap_codecs.py -v
```

Expected: all tests PASS

- [ ] Commit:

```bash
git add agent/mcap_codecs.py tests/test_mcap_codecs.py
git commit -m "feat(mcap_codecs): add ProtocolReaderFactory with ros1/ros2 auto-detection"
```

---

## Task 3: Refactor McapExtractor

**Files:**
- Modify: `agent/extractor.py`
- Modify: `tests/test_extractor.py`

### Step 3.1: Update test_extractor.py to use new mock layer

The existing tests mock `agent.extractor.read_ros2_messages` and produce objects with a `ros_msg` attribute. After refactoring, the extractor will use `make_reader` + `iter_decoded_messages` which yields 4-tuples `(schema, channel, message, decoded_message)`.

- [ ] Replace the content of `tests/test_extractor.py` with:

```python
# tests/test_extractor.py
from types import SimpleNamespace
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from agent.extractor import McapExtractor, chunk_pcm
from agent.mcap_codecs import SchemaDecoderRegistry


# ── chunk_pcm (unchanged logic) ──────────────────────────────────────────────

def test_pcm_frames_are_960_bytes():
    assert 480 * 2 == 960


def test_pcm_chunking():
    raw = b"\x01\x02" * 480 * 3
    frames = chunk_pcm(raw)
    assert len(frames) == 3
    assert all(len(f) == 960 for f in frames)


def test_pcm_chunking_drops_remainder():
    raw = b"\x00" * (960 * 2 + 100)
    frames = chunk_pcm(raw)
    assert len(frames) == 2


def test_audio_bytearray_output_identical_to_concat():
    chunks = [b"\x01\x02" * 100, b"\x03\x04" * 100, b"\x05\x06" * 100]
    old_result = b""
    for chunk in chunks:
        old_result += chunk
    _buf = bytearray()
    for chunk in chunks:
        _buf.extend(chunk)
    assert bytes(_buf) == old_result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _image_registry(frame: np.ndarray) -> SchemaDecoderRegistry:
    """Minimal registry that returns a fixed frame for sensor_msgs/Image."""
    reg = SchemaDecoderRegistry()
    reg.register_image("sensor_msgs/Image", lambda msg: frame)
    return reg


def _make_image_tuple(topic: str, log_time_ns: int, decoded_msg) -> tuple:
    """Build a (schema, channel, message, decoded_message) 4-tuple for image messages."""
    schema = SimpleNamespace(name="sensor_msgs/Image")
    channel = SimpleNamespace(topic=topic)
    message = SimpleNamespace(log_time=log_time_ns)
    return (schema, channel, message, decoded_msg)


def _make_audio_tuple(topic: str, log_time_ns: int, decoded_msg) -> tuple:
    schema = SimpleNamespace(name="audio_common_msgs/AudioData")
    channel = SimpleNamespace(topic=topic)
    message = SimpleNamespace(log_time=log_time_ns)
    return (schema, channel, message, decoded_msg)


def _make_imu_tuple(topic: str, log_time_ns: int, decoded_msg) -> tuple:
    schema = SimpleNamespace(name="sensor_msgs/Imu")
    channel = SimpleNamespace(topic=topic)
    message = SimpleNamespace(log_time=log_time_ns)
    return (schema, channel, message, decoded_msg)


def _patch_extractor(tuples: list[tuple]):
    """Patch ProtocolReaderFactory.build_decoder_factories and make_reader to
    return an iterator over the supplied 4-tuples."""
    mock_reader = MagicMock()
    mock_reader.iter_decoded_messages.return_value = iter(tuples)
    mock_make_reader = MagicMock(return_value=mock_reader)

    return (
        patch("agent.extractor.ProtocolReaderFactory.build_decoder_factories", return_value=[]),
        patch("agent.extractor.make_reader", mock_make_reader),
    )


# ── empty MCAP (no matching topics) ──────────────────────────────────────────

def test_empty_frames_on_missing_camera_topic(tmp_path):
    """An MCAP with no camera topic → frames=[], duration=0."""
    mcap_path = tmp_path / "empty.mcap"
    import mcap.writer as mw
    with open(mcap_path, "wb") as f:
        writer = mw.Writer(f)
        writer.start()
        writer.finish()

    extractor = McapExtractor(camera_topic="/camera/image_raw")
    data = extractor.extract(str(mcap_path))
    assert data["frames"] == []
    assert data["audio_frames"] is None
    assert data["duration_seconds"] == 0.0


# ── frame sampling ────────────────────────────────────────────────────────────

def test_frame_sample_rate_reduces_frame_count():
    """sample_rate=5 on 25 frames → 5 frames."""
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    reg = _image_registry(frame)
    tuples = [
        _make_image_tuple("/camera/image_raw", i * 1_000_000_000, SimpleNamespace())
        for i in range(25)
    ]
    extractor = McapExtractor(frame_sample_rate=5, registry=reg)
    p1, p2 = _patch_extractor(tuples)
    with p1, p2:
        data = extractor.extract("fake.mcap")
    assert len(data["frames"]) == 5


def test_frame_sample_rate_1_returns_all_frames():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    reg = _image_registry(frame)
    tuples = [
        _make_image_tuple("/camera/image_raw", i * 1_000_000_000, SimpleNamespace())
        for i in range(10)
    ]
    extractor = McapExtractor(frame_sample_rate=1, registry=reg)
    p1, p2 = _patch_extractor(tuples)
    with p1, p2:
        data = extractor.extract("fake.mcap")
    assert len(data["frames"]) == 10


def test_frame_sample_rate_larger_than_frame_count_returns_one_frame():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    reg = _image_registry(frame)
    tuples = [
        _make_image_tuple("/camera/image_raw", i * 1_000_000_000, SimpleNamespace())
        for i in range(5)
    ]
    extractor = McapExtractor(frame_sample_rate=100, registry=reg)
    p1, p2 = _patch_extractor(tuples)
    with p1, p2:
        data = extractor.extract("fake.mcap")
    assert len(data["frames"]) >= 1


# ── audio ─────────────────────────────────────────────────────────────────────

def test_audio_not_sampled_by_frame_sample_rate():
    """Audio is never reduced by frame_sample_rate."""
    reg = SchemaDecoderRegistry()
    reg.register_audio("audio_common_msgs/AudioData", lambda msg: bytes([0] * 960))
    tuples = [
        _make_audio_tuple("/audio/data", i * 1_000_000_000, SimpleNamespace())
        for i in range(10)
    ]
    extractor = McapExtractor(frame_sample_rate=100, registry=reg)
    p1, p2 = _patch_extractor(tuples)
    with p1, p2:
        data = extractor.extract("fake.mcap")
    assert data["audio_frames"] is not None
    assert len(data["audio_frames"]) == 10


# ── IMU ───────────────────────────────────────────────────────────────────────

def test_imu_data_accumulated():
    acc = SimpleNamespace(x=1.0, y=2.0, z=9.8)
    gyro = SimpleNamespace(x=0.0, y=0.0, z=0.1)
    imu_msg = SimpleNamespace(linear_acceleration=acc, angular_velocity=gyro)
    from agent.mcap_codecs import build_default_registry
    reg = build_default_registry()
    tuples = [
        _make_imu_tuple("/imu/data", i * 1_000_000_000, imu_msg)
        for i in range(3)
    ]
    extractor = McapExtractor(registry=reg)
    p1, p2 = _patch_extractor(tuples)
    with p1, p2:
        data = extractor.extract("fake.mcap")
    assert "/imu/data" in data["sensor_series"]
    assert data["sensor_series"]["/imu/data"].shape == (3, 6)
```

- [ ] Run tests to verify they fail:

```bash
uv run pytest tests/test_extractor.py -v
```

Expected: failures due to `McapExtractor` not accepting `registry` kwarg and still using old `read_ros2_messages` path.

### Step 3.2: Refactor McapExtractor

- [ ] Replace `agent/extractor.py` with the following:

```python
# agent/extractor.py
from __future__ import annotations
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


_PCM_FRAME_BYTES = 960  # 30ms × 16000Hz × 2 bytes (int16) = 960


def chunk_pcm(raw: bytes) -> list[bytes]:
    """Split raw PCM bytes into 30ms frames (960 bytes each). Drops remainder."""
    return [raw[i:i + _PCM_FRAME_BYTES] for i in range(0, len(raw) - _PCM_FRAME_BYTES + 1, _PCM_FRAME_BYTES)]


def _safe_iter(reader, topics: list[str]):
    """Wrap iter_decoded_messages to skip messages that raise DecoderNotFoundError."""
    it = reader.iter_decoded_messages(topics=topics)
    while True:
        try:
            yield next(it)
        except DecoderNotFoundError as e:
            logger.warning("No decoder for message encoding, skipping: {}", e)
        except StopIteration:
            return


class McapExtractor:
    def __init__(
        self,
        camera_topic: str = "/camera/image_raw",
        audio_topic: str = "/audio/data",
        imu_topic: str = "/imu/data",
        frame_sample_rate: int = 1,
        registry: SchemaDecoderRegistry | None = None,
    ) -> None:
        self._camera_topic = camera_topic
        self._audio_topic = audio_topic
        self._imu_topic = imu_topic
        self._frame_sample_rate = max(1, frame_sample_rate)
        self._registry = registry if registry is not None else build_default_registry()

    def extract(self, mcap_path: str) -> ExtractedData:
        """Parse an MCAP file and return ExtractedData.

        Auto-detects ROS1/ROS2 encoding. Handles both sensor_msgs/Image and
        sensor_msgs/CompressedImage. Audio and IMU extraction unchanged.
        """
        raw_frames: list[np.ndarray] = []
        _audio_buf = bytearray()
        imu_rows: list[np.ndarray] = []
        timestamps: list[float] = []

        topics = [self._camera_topic, self._audio_topic, self._imu_topic]
        decoder_factories = ProtocolReaderFactory.build_decoder_factories(mcap_path)

        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=decoder_factories)
            for schema, channel, message, decoded_message in _safe_iter(reader, topics):
                t = message.log_time / 1e9
                timestamps.append(t)
                topic = channel.topic

                if topic == self._camera_topic:
                    frame = self._registry.decode_image(schema.name, decoded_message)
                    if frame is not None:
                        raw_frames.append(frame)

                elif topic == self._audio_topic:
                    chunk = self._registry.decode_audio(schema.name, decoded_message)
                    if chunk:
                        _audio_buf.extend(chunk)

                elif topic == self._imu_topic:
                    row = self._registry.decode_imu(schema.name, decoded_message)
                    if row is not None:
                        imu_rows.append(row)

        # Apply frame sampling: keep every Nth frame, always keep at least 1 if any exist
        if raw_frames:
            frames = raw_frames[::self._frame_sample_rate]
            if not frames:
                frames = [raw_frames[0]]
        else:
            frames = []

        duration = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0
        raw_audio = bytes(_audio_buf)
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
```

- [ ] Run all tests to verify they pass:

```bash
uv run pytest tests/test_extractor.py tests/test_mcap_codecs.py -v
```

Expected: all tests PASS

- [ ] Run the full test suite to check for regressions:

```bash
uv run pytest -v
```

Expected: all tests PASS (or only pre-existing failures unrelated to this change)

- [ ] Commit:

```bash
git add agent/extractor.py tests/test_extractor.py
git commit -m "feat(extractor): refactor McapExtractor to use ProtocolReaderFactory and SchemaDecoderRegistry"
```

---

## Task 4: Install Dependency and Smoke Test

**Files:** none (runtime verification only)

### Step 4.1: Verify dependency in pyproject.toml and install

- [ ] Check that `mcap-ros1-support>=0.4` is present in `pyproject.toml`:

```bash
grep "mcap-ros1-support" pyproject.toml
```

Expected output: `"mcap-ros1-support>=0.4",`

If the line is missing, add it to the `dependencies` list in `pyproject.toml` alongside `mcap-ros2-support`.

- [ ] Install all dependencies:

```bash
uv pip install -e ".[dev]"
```

Expected: installs `mcap-ros1-support` (and other packages as needed)

### Step 4.2: Verify imports work end-to-end

- [ ] Verify both protocol decoders load correctly:

```bash
uv run python -c "
from mcap_ros1.decoder import DecoderFactory as R1
from mcap_ros2.decoder import DecoderFactory as R2
from agent.mcap_codecs import ProtocolReaderFactory, build_default_registry
print('ros1:', R1())
print('ros2:', R2())
print('registry:', build_default_registry())
print('OK')
"
```

Expected: prints `OK` with no ImportError

- [ ] Final full test run:

```bash
uv run pytest -v
```

Expected: all tests pass

- [ ] Commit if any adjustments were needed:

```bash
git add -p
git commit -m "chore: verify mcap-ros1-support dependency installed"
```
