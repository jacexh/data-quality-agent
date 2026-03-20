from __future__ import annotations
import numpy as np
from agent.analyzers.base import ExtractedData


_PCM_FRAME_BYTES = 960  # 30ms × 16000Hz × 2 bytes (int16) = 960

try:
    from mcap_ros2.reader import read_ros2_messages
except ImportError:
    read_ros2_messages = None  # type: ignore[assignment]


def chunk_pcm(raw: bytes) -> list[bytes]:
    """Split raw PCM bytes into 30ms frames (960 bytes each). Drops remainder."""
    return [raw[i:i + _PCM_FRAME_BYTES] for i in range(0, len(raw) - _PCM_FRAME_BYTES + 1, _PCM_FRAME_BYTES)]


class McapExtractor:
    def __init__(
        self,
        camera_topic: str = "/camera/image_raw",
        audio_topic: str = "/audio/data",
        imu_topic: str = "/imu/data",
        frame_sample_rate: int = 1,
    ) -> None:
        self._camera_topic = camera_topic
        self._audio_topic = audio_topic
        self._imu_topic = imu_topic
        self._frame_sample_rate = max(1, frame_sample_rate)

    def extract(self, mcap_path: str) -> ExtractedData:
        """Parse an MCAP file and return ExtractedData.

        Frames are decoded from sensor_msgs/Image messages.
        Audio is decoded from audio_common_msgs/AudioData messages and chunked to 30ms PCM frames.
        IMU is accumulated from sensor_msgs/Imu messages.
        """
        raw_frames: list[np.ndarray] = []
        raw_audio = b""
        imu_rows: list[np.ndarray] = []
        timestamps: list[float] = []

        if read_ros2_messages is None:
            raise RuntimeError("mcap-ros2-support not installed")

        try:
            for msg in read_ros2_messages(mcap_path, topics=[
                self._camera_topic, self._audio_topic, self._imu_topic
            ]):
                t = msg.log_time / 1e9  # nanoseconds → seconds
                timestamps.append(t)
                topic = msg.channel.topic

                if topic == self._camera_topic:
                    frame = self._decode_image(msg.ros_msg)
                    if frame is not None:
                        raw_frames.append(frame)

                elif topic == self._audio_topic:
                    chunk = self._decode_audio(msg.ros_msg)
                    if chunk:
                        raw_audio += chunk

                elif topic == self._imu_topic:
                    row = self._decode_imu(msg.ros_msg)
                    if row is not None:
                        imu_rows.append(row)
        except Exception:
            # Gracefully handle empty or malformed MCAP files
            pass

        # Apply frame sampling: take every Nth frame, always keep at least 1 if any exist
        frames: list[np.ndarray]
        if raw_frames:
            frames = raw_frames[::self._frame_sample_rate]
            if not frames:
                frames = [raw_frames[0]]
        else:
            frames = []

        duration = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0
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

    def _decode_image(self, msg) -> np.ndarray | None:
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
        except Exception:
            return None

    def _decode_audio(self, msg) -> bytes:
        try:
            return bytes(msg.data)
        except Exception:
            return b""

    def _decode_imu(self, msg) -> np.ndarray | None:
        try:
            a = msg.linear_acceleration
            g = msg.angular_velocity
            return np.array([a.x, a.y, a.z, g.x, g.y, g.z], dtype=np.float64)
        except Exception:
            return None
