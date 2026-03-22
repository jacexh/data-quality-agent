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


_IMAGE_SCHEMAS = {"sensor_msgs/Image", "sensor_msgs/CompressedImage"}
_AUDIO_SCHEMAS = {"audio_common_msgs/AudioData"}
_IMU_SCHEMAS   = {"sensor_msgs/Imu"}


class McapExtractor:
    """Parses MCAP files and extracts frames, audio, and IMU data for all configured topics.

    camera_topics / audio_topics:
        - Non-empty list: extract only the listed topics present in the file.
        - Empty list: auto-discover all image / audio topics found in the file.
    """

    def __init__(
        self,
        camera_topics: list[str] | None = None,
        audio_topics: list[str] | None = None,
        frame_sample_rate: int = 5,
        min_frames: int = 10,
        max_frames_per_topic: int = 300,
        registry: SchemaDecoderRegistry | None = None,
    ) -> None:
        self._camera_topics: list[str] = camera_topics if camera_topics is not None else []
        self._audio_topics: list[str] = audio_topics if audio_topics is not None else []
        self._frame_sample_rate = max(1, frame_sample_rate)
        self._min_frames = max(0, min_frames)
        self._max_frames_per_topic = max(1, max_frames_per_topic)
        self._registry = registry if registry is not None else build_default_registry()

    def _resolve_topics(self, mcap_path: str) -> tuple[list[str], list[str], str | None]:
        """Return (video_topics, audio_topics, imu_topic) to extract.

        If the configured lists are non-empty, return only those topics that exist
        in the file. If empty, auto-discover all image / audio topics from the file.
        Topics are returned in sorted order for determinism.
        """
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            channels = list(summary.channels.values()) if summary else []
            schemas = summary.schemas if summary else {}

        def schema_name(ch) -> str:
            s = schemas.get(ch.schema_id)
            return s.name if s else ""

        available_image = sorted(ch.topic for ch in channels if schema_name(ch) in _IMAGE_SCHEMAS)
        available_audio = sorted(ch.topic for ch in channels if schema_name(ch) in _AUDIO_SCHEMAS)
        available_imu   = sorted(ch.topic for ch in channels if schema_name(ch) in _IMU_SCHEMAS)

        if self._camera_topics:
            video_topics = [t for t in self._camera_topics if t in set(available_image)]
        else:
            video_topics = available_image

        if self._audio_topics:
            audio_topics = [t for t in self._audio_topics if t in set(available_audio)]
        else:
            audio_topics = available_audio

        imu_topic = available_imu[0] if available_imu else None

        return video_topics, audio_topics, imu_topic

    def extract(self, mcap_path: str) -> ExtractedData:
        """Parse an MCAP file and return ExtractedData with per-topic video/audio dicts."""
        video_topics, audio_topics, imu_topic = self._resolve_topics(mcap_path)
        all_topics = video_topics + audio_topics + ([imu_topic] if imu_topic else [])

        # Stream-sample frames during iteration to avoid accumulating all raw frames
        frame_counters: dict[str, int] = {t: 0 for t in video_topics}
        videos: dict[str, list[np.ndarray]] = {t: [] for t in video_topics}
        full_topics: set[str] = set()
        raw_audio_buf: dict[str, bytearray] = {t: bytearray() for t in audio_topics}
        timestamps: list[float] = []
        imu_rows: list[np.ndarray] = []

        decoder_factories = ProtocolReaderFactory.build_decoder_factories(mcap_path)
        with open(mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=decoder_factories)
            for schema, channel, message, decoded_message in _safe_iter(reader, all_topics):
                t = message.log_time / 1e9
                timestamps.append(t)
                topic = channel.topic

                if topic in videos and topic not in full_topics:
                    frame_counters[topic] += 1
                    if (frame_counters[topic] - 1) % self._frame_sample_rate == 0:
                        frame = self._registry.decode_image(schema.name, decoded_message)
                        if frame is not None:
                            videos[topic].append(frame)
                            if len(videos[topic]) >= self._max_frames_per_topic:
                                full_topics.add(topic)

                elif topic in raw_audio_buf:
                    chunk = self._registry.decode_audio(schema.name, decoded_message)
                    if chunk:
                        raw_audio_buf[topic].extend(chunk)

                elif imu_topic and topic == imu_topic:
                    row = self._registry.decode_imu(schema.name, decoded_message)
                    if row is not None:
                        imu_rows.append(row)

        # Post-loop: min_frames check for videos
        extraction_warnings: dict[str, str] = {}
        for topic, frames in videos.items():
            if 0 < len(frames) < self._min_frames:
                logger.warning(
                    "Topic {} has only {} frames after sampling — treated as empty",
                    topic, len(frames),
                )
                extraction_warnings[topic] = "below_min_frames"
                videos[topic] = []

        # Post-loop: convert audio buffers to PCM frame lists (no sub-sampling)
        audios: dict[str, list[bytes]] = {}
        for topic, buf in raw_audio_buf.items():
            raw = bytes(buf)
            audios[topic] = chunk_pcm(raw) if raw else []

        duration = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0
        sensor_series: dict[str, np.ndarray] = {}
        if imu_rows and imu_topic:
            sensor_series[imu_topic] = np.array(imu_rows, dtype=np.float64)

        return ExtractedData(
            videos=videos,
            audios=audios,
            sensor_series=sensor_series,
            duration_seconds=duration,
            extraction_warnings=extraction_warnings,
        )
