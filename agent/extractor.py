# agent/extractor.py
from __future__ import annotations
from typing import Any, Iterator
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


def _safe_iter(reader: Any, topics: list[str]) -> Iterator[tuple[Any, Any, Any, Any]]:
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
    """Parses MCAP files and extracts frames, audio, and IMU data.

    Supports ROS1 and ROS2 encodings. Auto-detects the camera topic if the
    configured topic is absent in the file.
    """

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

    def _resolve_topics(self, mcap_path: str) -> tuple[str, str, str]:
        """Return (camera_topic, audio_topic, imu_topic), auto-detecting camera topic if needed."""
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            channels = list(summary.channels.values()) if summary else []

        _IMAGE_SCHEMAS = {"sensor_msgs/Image", "sensor_msgs/CompressedImage"}
        _AUDIO_SCHEMAS = {"audio_common_msgs/AudioData"}
        _IMU_SCHEMAS   = {"sensor_msgs/Imu"}

        configured_topics = {self._camera_topic, self._audio_topic, self._imu_topic}
        available_topics  = {ch.topic for ch in channels}

        camera_topic = self._camera_topic
        audio_topic  = self._audio_topic
        imu_topic    = self._imu_topic

        if camera_topic not in available_topics:
            # Fall back to first image topic found in the file
            for ch in channels:
                schema_id = ch.schema_id
                if summary:
                    schema = summary.schemas.get(schema_id)
                    schema_name = schema.name if schema else ""
                    if schema_name in _IMAGE_SCHEMAS:
                        camera_topic = ch.topic
                        logger.info(
                            "camera_topic {!r} not found; using {!r} instead",
                            self._camera_topic, camera_topic,
                        )
                        break

        return camera_topic, audio_topic, imu_topic

    def extract(self, mcap_path: str) -> ExtractedData:
        """Parse an MCAP file and return ExtractedData.

        Auto-detects ROS1/ROS2 encoding. Handles both sensor_msgs/Image and
        sensor_msgs/CompressedImage. Audio and IMU extraction unchanged.
        Falls back to first available image topic if configured topic is absent.
        """
        raw_frames: list[np.ndarray] = []
        _audio_buf = bytearray()
        imu_rows: list[np.ndarray] = []
        timestamps: list[float] = []

        camera_topic, audio_topic, imu_topic = self._resolve_topics(mcap_path)
        topics = [camera_topic, audio_topic, imu_topic]
        decoder_factories = ProtocolReaderFactory.build_decoder_factories(mcap_path)
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=decoder_factories)
            for schema, channel, message, decoded_message in _safe_iter(reader, topics):
                t = message.log_time / 1e9
                timestamps.append(t)
                topic = channel.topic

                if topic == camera_topic:
                    frame = self._registry.decode_image(schema.name, decoded_message)
                    if frame is not None:
                        raw_frames.append(frame)

                elif topic == audio_topic:
                    chunk = self._registry.decode_audio(schema.name, decoded_message)
                    if chunk:
                        _audio_buf.extend(chunk)

                elif topic == imu_topic:
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
