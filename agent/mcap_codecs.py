# agent/mcap_codecs.py
from __future__ import annotations
from typing import Any, Callable
import cv2
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
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr
    except (AttributeError, ValueError) as e:
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
