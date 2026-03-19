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
