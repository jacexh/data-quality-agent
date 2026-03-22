from __future__ import annotations
import base64
import json
from typing import Any

import anthropic
import cv2
import numpy as np
from loguru import logger

_SYSTEM_PROMPT = """You are a data quality judge for robot-collected MCAP recordings.

You have access to algorithmic detector results and tools to inspect key frames and IMU data.

Your job:
1. Review any flagged sensitive information (face/voice/gait). Use get_key_frames to verify
   whether detections are genuine (live human) or false positives (poster, screen, background noise).
2. For borderline quality scores (within the review margin of threshold), review key frames and
   IMU context to determine if degradation is expected (e.g. motion blur during fast maneuver).
3. Call submit_verdict with your final decision and a concise narrative (2-4 sentences in Chinese).

Rules:
- If a tool call fails, treat the original detector result as authoritative.
- Do not override clear failures (score < 0.3, unambiguous live face). Only reconsider ambiguous cases.
- You MUST call submit_verdict to complete your review. Do not output free text as your final answer.
"""


def should_invoke_llm(
    detector_results: dict[str, Any],
    clarity_threshold: float,
    continuity_threshold: float,
    margin: float,
) -> bool:
    """Return True if LLM review is warranted.

    Triggers on: sensitive detections (face/voice/gait), cross-modal voice
    ambiguity, or quality scores within ``margin`` of their thresholds.
    """
    face = detector_results.get("face") or {}
    voice = detector_results.get("voice") or {}
    gait = detector_results.get("gait") or {}
    clarity = detector_results.get("clarity") or {}
    continuity = detector_results.get("continuity") or {}

    if face.get("has_face"):
        return True
    if gait.get("has_human_gait"):
        return True
    if voice.get("has_human_voice") and not face.get("has_face") and not gait.get("has_human_gait"):
        return True  # cross-modal ambiguity

    clarity_score = clarity.get("score")
    if clarity_score is not None and abs(clarity_score - clarity_threshold) <= margin:
        return True
    continuity_score = continuity.get("score")
    if continuity_score is not None and abs(continuity_score - continuity_threshold) <= margin:
        return True

    return False


class LLMJudge:
    """Conditional LLM reviewer for borderline or sensitive detector results.

    Wraps a Claude tool-use loop. Only invoked when ``should_invoke_llm``
    returns True; falls back to detector results on any API error.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        clarity_threshold: float,
        continuity_threshold: float,
        margin: float,
        base_url: str = "",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._clarity_threshold = clarity_threshold
        self._continuity_threshold = continuity_threshold
        self._margin = margin
        self._base_url = base_url

    def judge(
        self,
        topic: str,
        detector_results: dict[str, Any],
        frames: list[np.ndarray] | None,
        audio_frames: list[bytes] | None,
        sensor_series: dict[str, np.ndarray],
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Run LLM judgment if warranted.

        Returns (assessment_dict, error_name):
          - (dict, None)  → LLM ran successfully
          - (None, None)  → LLM skipped (not warranted)
          - (None, "llm") → LLM failed; caller uses detector fallback
        """
        if not should_invoke_llm(
            detector_results, self._clarity_threshold, self._continuity_threshold, self._margin
        ):
            return None, None

        try:
            return self._run_agent(detector_results, frames or [], sensor_series), None
        except Exception as exc:
            logger.warning("LLM judge failed for {!r}, falling back: {}", topic, exc)
            return None, "llm"

    def _run_agent(self, detector_results: dict[str, Any], frames, imu) -> dict[str, Any]:
        """Run the Claude tool-use agentic loop; raise on any error."""
        client = anthropic.Anthropic(
            api_key=self._api_key,
            **({"base_url": self._base_url} if self._base_url else {}),
        )

        tools = [
            {
                "name": "get_key_frames",
                "description": "Returns base64-encoded JPEG images for the specified frame indices.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "frame_indices": {"type": "array", "items": {"type": "integer"}}
                    },
                    "required": ["frame_indices"],
                },
            },
            {
                "name": "get_imu_summary",
                "description": "Returns IMU summary stats for a time window (seconds).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "window_start": {"type": "number"},
                        "window_end": {"type": "number"},
                    },
                    "required": ["window_start", "window_end"],
                },
            },
            {
                "name": "submit_verdict",
                "description": "Submit the final quality verdict. Must be called once all review is complete.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "passed": {"type": "boolean"},
                        "overrode_detector": {"type": "boolean"},
                        "override_detail": {"type": ["string", "null"]},
                        "narrative": {"type": "string"},
                        "frames_reviewed": {"type": "array", "items": {"type": "integer"}},
                        "imu_windows_reviewed": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "number"}},
                        },
                    },
                    "required": ["passed", "overrode_detector", "override_detail", "narrative",
                                 "frames_reviewed", "imu_windows_reviewed"],
                },
            },
        ]

        user_message = (
            f"Detector results:\n{json.dumps(detector_results, ensure_ascii=False, indent=2)}\n\n"
            "Please review the flagged detections and/or borderline scores, "
            "then call submit_verdict with your final decision."
        )
        messages = [{"role": "user", "content": user_message}]

        for _ in range(5):  # max tool-use rounds
            response = client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
                timeout=60.0,
            )

            # Process tool calls; return immediately on submit_verdict
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if block.name == "submit_verdict":
                    return dict(block.input)
                result_content = self._dispatch_tool(block.name, block.input, frames, imu)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_content,
                })

            if response.stop_reason != "tool_use":
                # Fallback: LLM stopped without calling submit_verdict — try to parse text as JSON
                text = next(
                    (b.text for b in response.content if hasattr(b, "text")), "{}"
                )
                try:
                    return json.loads(text)
                except json.JSONDecodeError as exc:
                    logger.error("LLM returned invalid JSON: {}…", text[:200])
                    raise RuntimeError("LLM response was not valid JSON") from exc

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        raise RuntimeError("LLM agent exceeded max tool-use rounds")

    def _dispatch_tool(
        self,
        name: str,
        inputs: dict[str, Any],
        frames: list[np.ndarray],
        imu: dict[str, np.ndarray],
    ) -> str:
        if name == "get_key_frames":
            indices = inputs.get("frame_indices", [])
            result = []
            out_of_bounds = []
            for idx in indices:
                if 0 <= idx < len(frames):
                    _, buf = cv2.imencode(".jpg", frames[idx], [cv2.IMWRITE_JPEG_QUALITY, 70])
                    result.append({
                        "frame_index": idx,
                        "image_b64": base64.b64encode(buf.tobytes()).decode(),
                    })
                else:
                    out_of_bounds.append(idx)
            if out_of_bounds:
                result.append({
                    "warning": f"indices {out_of_bounds} out of range (total frames: {len(frames)})"
                })
            return json.dumps(result)

        elif name == "get_imu_summary":
            # IMU rows contain sensor values only; per-row timestamps are not stored,
            # so window_start/window_end cannot be used for precise filtering.
            # Returns overall stats across all available IMU data.
            window_start = inputs.get("window_start")
            window_end = inputs.get("window_end")
            for arr in imu.values():
                if arr.shape[1] >= 3:
                    acc = arr[:, :3]
                    summary: dict[str, Any] = {
                        "mean_acceleration": float(np.linalg.norm(acc, axis=1).mean()),
                        "max_angular_velocity": float(np.abs(arr[:, 3:6]).max()) if arr.shape[1] >= 6 else 0.0,
                        "mean_angular_velocity": float(np.abs(arr[:, 3:6]).mean()) if arr.shape[1] >= 6 else 0.0,
                    }
                    if window_start is not None or window_end is not None:
                        summary["note"] = "per-row timestamps unavailable; stats cover full recording"
                    return json.dumps(summary)
            return json.dumps({
                "mean_acceleration": 0.0,
                "max_angular_velocity": 0.0,
                "mean_angular_velocity": 0.0,
                "note": "no_imu_data",
            })

        return json.dumps({"error": f"unknown tool: {name}"})
