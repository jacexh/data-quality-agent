## Project Overview

**Data Quality Agent** — 机器人传感器数据（MCAP 文件）自动质量评估系统。
MinIO 存储桶触发 webhook → 下载 MCAP → 并发算法检测 → LLM 判裁（可选）→ JSON 质量报告。

## Commands

```bash
# 安装依赖
uv pip install -e ".[dev]"

# 运行测试
uv run pytest                          # 全部
uv run pytest -xvs tests/test_main.py  # 单文件，详细输出
uv run pytest tests/analyzers/         # 仅 analyzer 测试

# 本地运行（需要 .env 中配置 ANTHROPIC_API_KEY）
uv run uvicorn agent.main:app --reload  # port 8000

# Docker（完整栈：MinIO + agent）
docker-compose up --build
docker-compose down
```

## Architecture

```
agent/
├── main.py          # FastAPI: POST /notify (webhook), GET /health
├── config.py        # Pydantic Settings，从 .env 读取
├── extractor.py     # McapExtractor: .mcap → ExtractedData (帧/音频/传感器)
├── pipeline.py      # AnalysisPipeline: ThreadPoolExecutor 并发运行 5 个检测器
├── llm_judge.py     # LLMJudge: Claude tool_use 循环，最多 5 轮
├── report.py        # ReportBuilder: 合并结果 → JSON 报告
└── analyzers/
    ├── base.py      # Analyzer Protocol + TypedDicts（ExtractedData, Results）
    ├── clarity.py   # Laplacian + Tenengrad 清晰度
    ├── continuity.py# Farneback 光流连续性
    ├── face.py      # YuNet ONNX 人脸检测
    ├── voice.py     # WebRTC VAD 语音检测
    └── gait.py      # HOG 行人检测
```

**数据流：** MinIO webhook → extractor → pipeline → llm_judge → report → loguru JSON 输出

## Environment Setup

复制 `.env.example` 为 `.env`，必填项：
- `ANTHROPIC_API_KEY` — Claude API 密钥（无默认值）
- 其余字段均有合理默认值（MinIO: `minio:9000`，bucket: `robot-uploads`）

## Key Patterns & Gotchas

- **Analyzer Protocol（非继承）**: 所有检测器实现 `Analyzer` Protocol（TypedDict），不用 OOP 继承
- **LLM 仅按需调用**: 有敏感检测（face/voice/gait）或分数在阈值 ±0.1 内才调用 LLM
- **静默失败防护**: 任何检测器返回 `null` 均判定为失败，绝不静默放行
- **错误隔离**: 一个检测器失败不中止其他检测器（`analyzer_errors` 字段记录）
- **LLM 降级**: Claude API 异常时自动回退到检测器结果，不影响报告输出
- **音频预分帧**: 音频已在 extractor 中切成 30ms PCM 帧（960 bytes @ 16kHz mono int16）再传给检测器
- **HOG 最小尺寸**: 帧小于 128×64 像素时 GaitDetector 跳过处理
- **模型文件**: `models/yunet.onnx`（~233 KB）已内置，无需运行时下载

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One tack per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimat Impact**: Changes should only touch what's necessary. Avoid introducing bugs.