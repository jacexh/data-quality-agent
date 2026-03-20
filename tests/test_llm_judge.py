import base64
import pytest
from unittest.mock import MagicMock, patch
from agent.llm_judge import LLMJudge, should_invoke_llm
from agent.analyzers.base import ExtractedData
import numpy as np


def _make_detector_results(
    clarity_score=0.9, continuity_score=0.9,
    has_face=False, has_voice=False, has_gait=False,
):
    return {
        "clarity": {"score": clarity_score, "method": "laplacian+tenengrad", "detail": {}},
        "continuity": {"score": continuity_score, "method": "optical_flow", "detail": {}},
        "face": {"has_face": has_face, "face_count": 1 if has_face else 0},
        "voice": {"has_human_voice": has_voice},
        "gait": {"has_human_gait": has_gait},
    }


def test_should_skip_when_all_clear():
    results = _make_detector_results()
    assert should_invoke_llm(results, clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1) is False


def test_should_invoke_when_face_detected():
    results = _make_detector_results(has_face=True)
    assert should_invoke_llm(results, clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1) is True


def test_should_invoke_when_score_borderline():
    results = _make_detector_results(clarity_score=0.62)  # within 0.1 of 0.6 threshold
    assert should_invoke_llm(results, clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1) is True


def test_should_invoke_on_cross_modal_ambiguity():
    """Voice detected but no face and no gait → ambiguous."""
    results = _make_detector_results(has_voice=True)
    assert should_invoke_llm(results, clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1) is True


def test_llm_failure_falls_back_to_detector_verdict(sharp_data):
    """If Anthropic API raises, LLMJudge returns None assessment and 'llm' error."""
    judge = LLMJudge(api_key="fake", model="claude-sonnet-4-6", clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1)
    results = _make_detector_results(has_face=True)

    with patch("agent.llm_judge.anthropic.Anthropic") as mock_client_cls:
        mock_client_cls.return_value.messages.create.side_effect = RuntimeError("API down")
        assessment, error = judge.judge(results, sharp_data)

    assert assessment is None
    assert error == "llm"


def test_llm_skipped_returns_none_assessment_and_no_error(sharp_data):
    judge = LLMJudge(api_key="fake", model="claude-sonnet-4-6", clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1)
    results = _make_detector_results()  # all clear → should skip
    assessment, error = judge.judge(results, sharp_data)
    assert assessment is None
    assert error is None


# ── should_invoke_llm: additional equivalence / boundary cases ─────────────

def test_should_invoke_when_gait_detected():
    results = _make_detector_results(has_gait=True)
    assert should_invoke_llm(results, clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1) is True


def test_should_invoke_when_continuity_borderline():
    """Continuity score within margin of threshold triggers LLM review."""
    results = _make_detector_results(continuity_score=0.65)  # abs(0.65 - 0.6) = 0.05 ≤ 0.1
    assert should_invoke_llm(results, clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1) is True


def test_should_not_invoke_when_score_clearly_below_threshold():
    """Score well below threshold is NOT borderline — LLM adds no value, detectors already fail it."""
    results = _make_detector_results(clarity_score=0.3)  # abs(0.3 - 0.6) = 0.3 > 0.1
    assert should_invoke_llm(results, clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1) is False


def test_score_at_exact_threshold_triggers_llm():
    """Score exactly equal to threshold: diff = 0.0 ≤ margin → borderline, invoke LLM."""
    results = _make_detector_results(clarity_score=0.6)  # abs(0.6 - 0.6) = 0.0 ≤ 0.1
    assert should_invoke_llm(results, clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1) is True


# ── LLM agent error paths ──────────────────────────────────────────────────

def test_llm_max_rounds_exceeded_returns_llm_error(sharp_data):
    """Agent that only returns tool_use forever hits 5-round limit → (None, 'llm')."""
    judge = LLMJudge(api_key="fake", model="claude-sonnet-4-6", clarity_threshold=0.6, continuity_threshold=0.6, margin=0.1)
    results = _make_detector_results(has_face=True)

    mock_response = MagicMock()
    mock_response.stop_reason = "tool_use"
    mock_response.content = []  # no tool_use blocks → tool_results=[], loop never exits

    with patch("agent.llm_judge.anthropic.Anthropic") as mock_client_cls:
        mock_client_cls.return_value.messages.create.return_value = mock_response
        assessment, error = judge.judge(results, sharp_data)

    assert assessment is None
    assert error == "llm"
