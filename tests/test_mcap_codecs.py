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


def test_registry_custom_imu_decoder_called():
    reg = SchemaDecoderRegistry()
    sentinel = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    reg.register_imu("my_msgs/MyImu", lambda msg: sentinel)
    result = reg.decode_imu("my_msgs/MyImu", object())
    np.testing.assert_array_equal(result, sentinel)


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
