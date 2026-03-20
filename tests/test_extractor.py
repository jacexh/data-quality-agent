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
