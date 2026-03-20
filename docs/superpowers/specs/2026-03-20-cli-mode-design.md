# CLI Mode Design

**Date:** 2026-03-20
**Status:** Approved
**Scope:** Add `agent-cli analyze <file.mcap>` command for local file analysis

---

## Problem

The current system only supports triggering analysis via MinIO webhook → FastAPI server. There is no way to run quality analysis on a local MCAP file without spinning up the full server stack. This makes debugging and ad-hoc inspection difficult.

---

## Goals

- Analyze a local `.mcap` file directly from the command line
- Output JSON report to stdout (same schema as existing loguru report)
- Exit code `0` = passed, `1` = failed/error
- No new dependencies (stdlib `sys.argv` only)
- Install as `agent-cli` console script via `pyproject.toml`

---

## Non-Goals

- Human-readable / formatted output
- MinIO/S3 download support
- Multiple subcommands or a full CLI framework
- Interactive mode

---

## Architecture

### Approach: Minimal Invasion (Method A)

Extract the core analysis logic from `main.py` into a shared function, then have `agent/cli.py` call it directly.

```
agent/
├── main.py        # FastAPI server — extracts _analyze_local() as shared helper
├── cli.py         # NEW: CLI entry point
└── ...

pyproject.toml     # Adds [project.scripts] entry
```

### Shared Function Extraction

`main.py` currently contains `_analyze_and_log(source_file, bucket, local_path)` which is tightly coupled to the server context (logs via loguru, uses module-level singletons). We extract the pure analysis logic into a standalone function:

```python
# agent/main.py (or a shared module)
def analyze_local_file(local_path: str, source_file: str = "", bucket: str = "") -> dict:
    """Run the full pipeline on a local MCAP file. Returns the report dict."""
    data = _extractor.extract(local_path)
    detector_results, detector_errors = _pipeline.run(data)
    llm_assessment, llm_error = _judge.judge(detector_results, data)
    return _builder.build(
        source_file=source_file or local_path,
        bucket=bucket,
        detector_results=detector_results,
        detector_errors=detector_errors,
        llm_assessment=llm_assessment,
        llm_error=llm_error,
        duration_seconds=data["duration_seconds"],
    )
```

`_analyze_and_log` becomes a thin wrapper that calls this and logs. No behavior change for the server path.

### CLI Entry Point (`agent/cli.py`)

```python
def main():
    # 1. Parse argv: agent-cli analyze <path>
    # 2. Validate file exists and ends with .mcap
    # 3. Initialize pipeline singletons (same as main.py)
    # 4. Call analyze_local_file(path)
    # 5. print(json.dumps(report, ensure_ascii=False, indent=2))
    # 6. sys.exit(0 if report["passed"] else 1)
```

### `pyproject.toml` Change

```toml
[project.scripts]
agent-cli = "agent.cli:main"
```

---

## Data Flow

```
agent-cli analyze /data/file.mcap
        │
        ▼
agent/cli.py::main()
        │  validates path
        ▼
analyze_local_file(path)   ← shared with main.py server path
        │
        ├── McapExtractor.extract()
        ├── AnalysisPipeline.run()  (ThreadPoolExecutor, 5 analyzers)
        └── LLMJudge.judge()  (optional, same thresholds as server)
        │
        ▼
report dict  →  json.dumps()  →  stdout
                                  exit 0/1
```

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| File not found | Print error message to stderr, exit 2 |
| File not `.mcap` | Print error message to stderr, exit 2 |
| MCAP extraction failure | Report with `detector_errors: ["mcap_extraction"]`, exit 1 |
| Analyzer failure | Report with failing analyzer in `analyzer_errors`, pipeline continues |
| LLM failure | Graceful degradation (existing behavior), report still emitted |
| Missing `ANTHROPIC_API_KEY` | LLM judge skipped (existing behavior) |

---

## Usage

```bash
# Install
uv pip install -e ".[dev]"

# Run
agent-cli analyze /path/to/recording.mcap

# Output (stdout)
{
  "passed": true,
  "source_file": "/path/to/recording.mcap",
  ...
}

# Exit codes
# 0 = quality check passed
# 1 = quality check failed or analysis error
# 2 = invalid arguments / file not found
```

---

## Testing

- Add `tests/test_cli.py` covering:
  - Happy path: valid `.mcap` → JSON output, exit 0/1
  - File not found → exit 2
  - Non-`.mcap` extension → exit 2
  - Uses existing mock patterns from `tests/test_main.py`
- No new test infrastructure required

---

## Files Changed

| File | Change |
|------|--------|
| `agent/main.py` | Extract `analyze_local_file()` helper; `_analyze_and_log` becomes thin wrapper |
| `agent/cli.py` | New file: CLI entry point |
| `pyproject.toml` | Add `[project.scripts]` entry |
| `tests/test_cli.py` | New file: CLI tests |
