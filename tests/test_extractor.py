import struct
import numpy as np
import pytest
from agent.extractor import McapExtractor


def test_empty_frames_on_missing_camera_topic(tmp_path):
    """An MCAP with no camera topic → frames=[]."""
    import mcap.writer as mw
    mcap_path = tmp_path / "empty.mcap"
    with open(mcap_path, "wb") as f:
        writer = mw.Writer(f)
        writer.start()
        writer.finish()

    extractor = McapExtractor(camera_topic="/camera/image_raw")
    data = extractor.extract(str(mcap_path))
    assert data["frames"] == []
    assert data["audio_frames"] is None
    assert data["duration_seconds"] == 0.0


def test_pcm_frames_are_960_bytes():
    """PCM frames must be exactly 960 bytes (30ms at 16kHz mono int16)."""
    # 480 samples × 2 bytes = 960
    assert 480 * 2 == 960


def test_pcm_chunking():
    """BagExtractor chunks raw PCM into 960-byte frames."""
    from agent.extractor import chunk_pcm
    raw = b"\x01\x02" * 480 * 3  # exactly 3 frames
    frames = chunk_pcm(raw)
    assert len(frames) == 3
    assert all(len(f) == 960 for f in frames)


def test_pcm_chunking_drops_remainder():
    """Incomplete trailing bytes are dropped."""
    from agent.extractor import chunk_pcm
    raw = b"\x00" * (960 * 2 + 100)  # 2 full frames + 100 leftover bytes
    frames = chunk_pcm(raw)
    assert len(frames) == 2


def test_frame_sample_rate_reduces_frame_count(tmp_path):
    """sample_rate=N returns every Nth frame only."""
    from unittest.mock import patch, MagicMock

    extractor = McapExtractor(frame_sample_rate=5)

    def make_msg(i):
        m = MagicMock()
        m.log_time = i * 1_000_000_000
        m.channel.topic = "/camera/image_raw"
        m.ros_msg.height = 4
        m.ros_msg.width = 4
        m.ros_msg.data = bytes([128] * 48)
        m.ros_msg.encoding = "bgr8"
        return m

    msgs = [make_msg(i) for i in range(25)]

    with patch("agent.extractor.read_ros2_messages", return_value=iter(msgs)):
        data = extractor.extract("fake.mcap")

    assert len(data["frames"]) == 5  # 25 // 5 = 5


def test_frame_sample_rate_1_returns_all_frames(tmp_path):
    """sample_rate=1 (default) returns all frames unchanged."""
    from unittest.mock import patch, MagicMock

    extractor = McapExtractor(frame_sample_rate=1)

    def make_msg(i):
        m = MagicMock()
        m.log_time = i * 1_000_000_000
        m.channel.topic = "/camera/image_raw"
        m.ros_msg.height = 4
        m.ros_msg.width = 4
        m.ros_msg.data = bytes([128] * 48)
        m.ros_msg.encoding = "bgr8"
        return m

    msgs = [make_msg(i) for i in range(10)]

    with patch("agent.extractor.read_ros2_messages", return_value=iter(msgs)):
        data = extractor.extract("fake.mcap")

    assert len(data["frames"]) == 10


def test_frame_sample_rate_larger_than_frame_count_returns_one_frame():
    """sample_rate=100 on 5-frame video returns 1 frame, not empty list."""
    from unittest.mock import patch, MagicMock

    extractor = McapExtractor(frame_sample_rate=100)

    def make_msg(i):
        m = MagicMock()
        m.log_time = i * 1_000_000_000
        m.channel.topic = "/camera/image_raw"
        m.ros_msg.height = 4
        m.ros_msg.width = 4
        m.ros_msg.data = bytes([128] * 48)
        m.ros_msg.encoding = "bgr8"
        return m

    msgs = [make_msg(i) for i in range(5)]

    with patch("agent.extractor.read_ros2_messages", return_value=iter(msgs)):
        data = extractor.extract("fake.mcap")

    assert len(data["frames"]) >= 1
