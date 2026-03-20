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


def _patch_open_and_reader(mock_reader):
    """Patch open() (intercepting only 'fake.mcap') and make_reader."""
    import builtins
    real_open = builtins.open

    def selective_open(path, *args, **kwargs):
        if path == "fake.mcap":
            mock_file = MagicMock()
            mock_file.__enter__ = MagicMock(return_value=mock_file)
            mock_file.__exit__ = MagicMock(return_value=False)
            return mock_file
        return real_open(path, *args, **kwargs)

    return (
        patch("builtins.open", side_effect=selective_open),
        patch("agent.mcap_codecs.make_reader", return_value=mock_reader),
    )


def test_detect_encodings_ros1():
    from agent.mcap_codecs import ProtocolReaderFactory
    summary = _make_summary(["ros1msg"])
    mock_reader = MagicMock()
    mock_reader.get_summary.return_value = summary
    open_patch, reader_patch = _patch_open_and_reader(mock_reader)
    with open_patch, reader_patch:
        encodings = ProtocolReaderFactory._detect_encodings("fake.mcap")
    assert "ros1" in encodings


def test_detect_encodings_ros2():
    from agent.mcap_codecs import ProtocolReaderFactory
    summary = _make_summary(["cdr"])
    mock_reader = MagicMock()
    mock_reader.get_summary.return_value = summary
    open_patch, reader_patch = _patch_open_and_reader(mock_reader)
    with open_patch, reader_patch:
        encodings = ProtocolReaderFactory._detect_encodings("fake.mcap")
    assert "ros2" in encodings


def test_detect_encodings_unknown_emits_warning(caplog):
    import logging
    from agent.mcap_codecs import ProtocolReaderFactory
    summary = _make_summary(["exotic_proto"])
    mock_reader = MagicMock()
    mock_reader.get_summary.return_value = summary
    open_patch, reader_patch = _patch_open_and_reader(mock_reader)
    with open_patch, reader_patch:
        with patch("agent.mcap_codecs.logger") as mock_logger:
            ProtocolReaderFactory._detect_encodings("fake.mcap")
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0]
            assert "exotic_proto" in str(call_args)


def test_detect_encodings_none_summary_fallback():
    """When get_summary() returns None, fall back to iter_messages() for Channel records."""
    from agent.mcap_codecs import ProtocolReaderFactory
    from mcap.records import Channel

    channel = MagicMock(spec=Channel)
    channel.metadata = {"encoding": "ros1msg"}

    mock_reader = MagicMock()
    mock_reader.get_summary.return_value = None
    mock_reader.iter_messages.return_value = iter([MagicMock(channel=channel)])

    open_patch, reader_patch = _patch_open_and_reader(mock_reader)
    with open_patch, reader_patch:
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
