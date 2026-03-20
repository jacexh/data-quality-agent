from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from agent.analyzers.base import Analyzer, ExtractedData


class AnalysisPipeline:
    """Runs all analyzers concurrently via a thread pool."""

    def __init__(self, analyzers: list[Analyzer], max_workers: int = 5) -> None:
        self._analyzers = analyzers
        self._max_workers = max_workers

    def run(self, data: ExtractedData) -> tuple[dict[str, Any], list[str]]:
        """Run all analyzers concurrently.

        Returns:
            results: dict mapping analyzer.name() → result dict (or None on error)
            errors: list of analyzer names that raised
        """
        results: dict[str, Any] = {}
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(a.analyze, data): a for a in self._analyzers}
            for future, analyzer in futures.items():
                name = analyzer.name()
                try:
                    results[name] = future.result()
                except Exception:
                    results[name] = None
                    errors.append(name)

        return results, errors
