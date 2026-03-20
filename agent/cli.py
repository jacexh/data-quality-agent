# agent/cli.py
from __future__ import annotations
import json
import os
import sys


def main() -> None:
    args = sys.argv[1:]

    # Validate: must be "analyze <path>"
    if len(args) < 2 or args[0] != "analyze":
        print(
            "Usage: agent-cli analyze <path/to/file.mcap>",
            file=sys.stderr,
        )
        sys.exit(2)

    path = args[1]

    if not os.path.exists(path):
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(2)

    if not path.endswith(".mcap"):
        print(f"Error: file must have .mcap extension: {path}", file=sys.stderr)
        sys.exit(2)

    from agent.runner import analyze_local_file

    report = analyze_local_file(path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report["passed"] else 1)
