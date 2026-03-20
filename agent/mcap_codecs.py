# agent/mcap_codecs.py
from __future__ import annotations
from typing import Any, Callable
import cv2
import numpy as np
from loguru import logger
from mcap.reader import make_reader

ImageDecodeFn = Callable[[Any], "np.ndarray | None"]
AudioDecodeFn = Callable[[Any], bytes]
ImuDecodeFn   = Callable[[Any], "np.ndarray | None"]


# ── Built-in decode functions (module-level, pure) ───────────────────────────

def _decode_raw_image(msg: Any) -> np.ndarray | None:
    """Decode sensor_msgs/Image → BGR ndarray. Returns None on failure.

    Supports bgr8 and rgb8 encodings. mono8 and other single-channel encodings
    are not supported and return None.
    """
    try:
        h, w = msg.height, msg.width
        data = bytes(msg.data)
        encoding = getattr(msg, "encoding", "bgr8")
        if "rgb" not in encoding and "bgr" not in encoding:
            logger.debug("_decode_raw_image: unsupported encoding {:!r}, skipping", encoding)
            return None
        channels = 3
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, w, channels)
        if "rgb" in encoding:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    except (AttributeError, ValueError, IndexError, TypeError) as e:
        logger.debug("_decode_raw_image failed: {}", e)
        return None


def _decode_compressed_image(msg: Any) -> np.ndarray | None:
    """Decode sensor_msgs/CompressedImage (JPEG/PNG) → BGR ndarray. Returns None on failure."""
    try:
        buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
        arr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if arr is None:
            logger.debug("_decode_compressed_image: cv2.imdecode returned None")
        return arr
    except (AttributeError, ValueError, IndexError, TypeError) as e:
        logger.debug("_decode_compressed_image failed: {}", e)
        return None


def _decode_audio(msg: Any) -> bytes:
    """Decode audio_common_msgs/AudioData → raw bytes. Returns b'' on failure."""
    try:
        return bytes(msg.data)
    except (AttributeError, TypeError) as e:
        logger.debug("_decode_audio failed: {}", e)
        return b""


def _decode_imu(msg: Any) -> np.ndarray | None:
    """Decode sensor_msgs/Imu → 6-element float64 array [ax,ay,az,gx,gy,gz]. Returns None on failure."""
    try:
        a = msg.linear_acceleration
        g = msg.angular_velocity
        return np.array([a.x, a.y, a.z, g.x, g.y, g.z], dtype=np.float64)
    except (AttributeError, TypeError, ValueError) as e:
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


# ── ProtocolReaderFactory ────────────────────────────────────────────────────

# Encoding strings found in MCAP channel message_encoding field → logical protocol name
# Handles both legacy metadata["encoding"] style ("ros1msg", "cdr") and
# newer message_encoding field style ("ros1", "ros2").
_ENCODING_MAP: dict[str, str] = {
    "ros1msg": "ros1",
    "ros1":    "ros1",
    "cdr":     "ros2",
    "ros2":    "ros2",
}

_SUPPORTED: set[str] = {"ros1", "ros2"}


class ProtocolReaderFactory:
    """Detect MCAP file encoding and build matching mcap DecoderFactory instances."""

    @classmethod
    def build_decoder_factories(cls, path: str) -> list[Any]:
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
                    enc for ch in summary.channels.values()
                    if (enc := (ch.message_encoding or ch.metadata.get("encoding", "")))
                }
            else:
                # Unindexed / streaming-written file — scan Channel records
                raw_encodings = set()
                for item in reader.iter_messages():
                    enc = item.channel.message_encoding or item.channel.metadata.get("encoding", "")
                    if enc:
                        raw_encodings.add(enc)

        # raw_encodings is a plain set of strings — safe to use after file is closed
        protocols: set[str] = set()
        for enc in raw_encodings:
            protocol = _ENCODING_MAP.get(enc)
            if protocol in _SUPPORTED:
                protocols.add(protocol)
            elif enc:
                logger.warning("Unsupported MCAP encoding {:!r}, skipping", enc)

        return protocols

    @classmethod
    def _build_factories_for(cls, encodings: set[str]) -> list[Any]:
        """Lazily import and instantiate DecoderFactory for each known protocol."""
        factories = []
        if "ros1" in encodings:
            from mcap_ros1.decoder import DecoderFactory as Ros1Factory  # type: ignore[import]
            factories.append(Ros1Factory())
        if "ros2" in encodings:
            from mcap_ros2.decoder import DecoderFactory as Ros2Factory
            factories.append(Ros2Factory())
        return factories
