import pytest
from agent.pipeline import AnalysisPipeline
from agent.analyzers.base import ExtractedData


class _OkAnalyzer:
    def name(self) -> str:
        return "ok"

    def analyze(self, data):
        return {"value": 42}


class _BrokenAnalyzer:
    def name(self) -> str:
        return "broken"

    def analyze(self, data):
        raise RuntimeError("oops")


def test_all_analyzers_run(sharp_data):
    pipeline = AnalysisPipeline(analyzers=[_OkAnalyzer()])
    results, errors = pipeline.run(sharp_data)
    assert results["ok"] == {"value": 42}
    assert errors == []


def test_broken_analyzer_does_not_abort_others(sharp_data):
    pipeline = AnalysisPipeline(analyzers=[_OkAnalyzer(), _BrokenAnalyzer()])
    results, errors = pipeline.run(sharp_data)
    assert results["ok"] == {"value": 42}
    assert results["broken"] is None
    assert "broken" in errors


def test_all_broken_returns_all_none(sharp_data):
    pipeline = AnalysisPipeline(analyzers=[_BrokenAnalyzer()])
    results, errors = pipeline.run(sharp_data)
    assert results["broken"] is None
    assert errors == ["broken"]
