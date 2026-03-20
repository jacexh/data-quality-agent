# tests/test_cli.py
import json
import pytest
from unittest.mock import patch


PASSING_REPORT = {
    "report_id": "abc",
    "source_file": "/tmp/test.mcap",
    "minio_bucket": "",
    "analyzed_at": "2026-01-01T00:00:00Z",
    "duration_seconds": 5.0,
    "scores": None,
    "sensitive_info": None,
    "llm_assessment": None,
    "llm_skipped_reason": "all_detectors_clear_no_borderline_scores",
    "analyzer_errors": [],
    "passed": True,
    "failure_reasons": [],
}

FAILING_REPORT = {**PASSING_REPORT, "passed": False, "failure_reasons": ["clarity"]}


def _run_cli(argv, mock_report=None):
    """Run cli.main() with given argv, return (stdout_lines, exit_code)."""
    import sys
    from io import StringIO

    if mock_report is None:
        mock_report = PASSING_REPORT

    captured = StringIO()
    exit_code = 0

    with patch("sys.argv", ["agent-cli"] + argv), \
         patch("sys.stdout", captured), \
         patch("agent.runner.analyze_local_file", return_value=mock_report):
        try:
            from agent import cli
            import importlib
            importlib.reload(cli)  # ensure fresh import each time
            cli.main()
        except SystemExit as e:
            exit_code = e.code

    return captured.getvalue(), exit_code


def test_analyze_passing_file_exits_0(tmp_path):
    mcap = tmp_path / "recording.mcap"
    mcap.write_bytes(b"fake")
    out, code = _run_cli(["analyze", str(mcap)], mock_report=PASSING_REPORT)
    assert code == 0
    report = json.loads(out)
    assert report["passed"] is True


def test_analyze_failing_file_exits_1(tmp_path):
    mcap = tmp_path / "recording.mcap"
    mcap.write_bytes(b"fake")
    out, code = _run_cli(["analyze", str(mcap)], mock_report=FAILING_REPORT)
    assert code == 1
    report = json.loads(out)
    assert report["passed"] is False


def test_output_is_valid_json(tmp_path):
    mcap = tmp_path / "recording.mcap"
    mcap.write_bytes(b"fake")
    out, _ = _run_cli(["analyze", str(mcap)])
    parsed = json.loads(out)
    assert "passed" in parsed
    assert "source_file" in parsed


def test_file_not_found_exits_2(tmp_path):
    _, code = _run_cli(["analyze", str(tmp_path / "missing.mcap")])
    assert code == 2


def test_non_mcap_extension_exits_2(tmp_path):
    f = tmp_path / "recording.bag"
    f.write_bytes(b"fake")
    _, code = _run_cli(["analyze", str(f)])
    assert code == 2


def test_no_args_exits_2():
    _, code = _run_cli([])
    assert code == 2


def test_missing_subcommand_exits_2():
    _, code = _run_cli(["analyze"])
    assert code == 2
