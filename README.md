# Data Quality Agent

机器人传感器数据（MCAP 文件）自动质量评估系统。

MinIO 存储桶上传触发 webhook → 下载 MCAP → 并发算法检测 → LLM 裁决（按需）→ JSON 质量报告。

## 功能概览

- **图像清晰度检测**：Laplacian + Tenengrad 方差，量化视频帧的焦距质量
- **运动连续性检测**：Farneback 光流，检测帧间抖动或跳帧
- **人脸检测**：YuNet ONNX 模型，识别画面中是否存在人脸（隐私合规）
- **人声检测**：WebRTC VAD，检测录音中是否含有人类语音
- **步态检测**：HOG 描述符，检测画面中是否存在人形步态
- **LLM 裁决**：分数接近阈值或敏感信息被标记时，调用 Claude 进行二次审核，支持关键帧图像查阅和 IMU 数据分析

## 系统架构

```
agent/
├── main.py          # FastAPI：POST /notify（webhook）, GET /health
├── config.py        # Pydantic Settings，从 .env 读取
├── extractor.py     # McapExtractor：.mcap → ExtractedData（帧/音频/传感器）
├── pipeline.py      # AnalysisPipeline：ThreadPoolExecutor 并发运行 5 个检测器
├── llm_judge.py     # LLMJudge：Claude tool_use 循环，最多 5 轮
├── report.py        # ReportBuilder：合并结果 → JSON 报告
└── analyzers/
    ├── base.py      # Analyzer Protocol + TypedDicts（ExtractedData, Results）
    ├── clarity.py   # 图像清晰度
    ├── continuity.py# 运动连续性
    ├── face.py      # 人脸检测
    ├── voice.py     # 人声检测
    └── gait.py      # 步态检测
```

**数据流：**

```
MinIO S3 事件
    └─→ POST /notify
            └─→ McapExtractor（提取帧/音频/IMU）
                    └─→ AnalysisPipeline（5 个检测器并发）
                            └─→ LLMJudge（按需调用 Claude）
                                    └─→ ReportBuilder → loguru JSON 输出
```

## 快速开始

### 前置条件

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) 包管理器
- Docker & Docker Compose（完整栈运行）

### 本地开发

```bash
# 安装依赖
uv pip install -e ".[dev]"

# 配置环境变量
cp .env.example .env
# 编辑 .env，填写 ANTHROPIC_API_KEY

# 启动服务（port 8000）
uv run uvicorn agent.main:app --reload
```

### Docker 完整栈（推荐）

```bash
# 启动 MinIO + Agent
docker-compose up --build

# 停止
docker-compose down
```

启动后：
- Agent API：`http://localhost:8000`
- MinIO 控制台：`http://localhost:9001`（账号 `minioadmin` / `minioadmin`）
- MinIO S3 API：`http://localhost:9000`

上传 `.mcap` 文件到 `robot-uploads` 存储桶即可自动触发质量评估。

## 环境变量

复制 `.env.example` 为 `.env` 并按需修改：

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `ANTHROPIC_API_KEY` | _(必填)_ | Claude API 密钥 |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO 服务地址 |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO 访问密钥 |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO 秘密密钥 |
| `MINIO_BUCKET` | `robot-uploads` | 监听的存储桶名称 |
| `MINIO_USE_SSL` | `false` | 是否启用 HTTPS |
| `CLARITY_THRESHOLD` | `0.6` | 清晰度合格分数线（0-1） |
| `CONTINUITY_THRESHOLD` | `0.6` | 连续性合格分数线（0-1） |
| `LLM_REVIEW_MARGIN` | `0.1` | 触发 LLM 二次审核的阈值容差 |
| `MINIMUM_DURATION_SECONDS` | `1.0` | 录制最短时长要求（秒） |
| `WEBHOOK_AUTH_TOKEN` | _(可选)_ | webhook Bearer Token 鉴权 |
| `LLM_MODEL` | `claude-sonnet-4-6` | 使用的 Claude 模型 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## API

### `GET /health`

健康检查。

```json
{"status": "ok"}
```

### `POST /notify`

接收 MinIO S3 事件通知。非 `.mcap` 文件将被忽略，分析在后台异步执行。

**请求体（MinIO 标准格式）：**

```json
{
  "Records": [{
    "s3": {
      "bucket": {"name": "robot-uploads"},
      "object": {"key": "session_001.mcap"}
    }
  }]
}
```

**响应：**

```json
{"status": "accepted"}
```

## 质量报告

分析完成后，报告以结构化 JSON 输出到日志（通过 loguru）：

```json
{
  "report_id": "uuid",
  "source_file": "robot-uploads/session_001.mcap",
  "minio_bucket": "robot-uploads",
  "analyzed_at": "2026-03-20T10:00:00Z",
  "duration_seconds": 30.5,
  "scores": {
    "clarity": {"score": 0.82, "method": "laplacian+tenengrad"},
    "continuity": {"score": 0.75, "method": "farneback"}
  },
  "sensitive_info": {
    "has_face": false,
    "face_count": 0,
    "has_human_voice": false,
    "has_human_gait": false
  },
  "llm_assessment": null,
  "llm_skipped_reason": "all_detectors_clear_no_borderline_scores",
  "analyzer_errors": [],
  "passed": true,
  "failure_reasons": []
}
```

`passed: false` 时，`failure_reasons` 列举具体原因，如 `clarity`、`has_face`、`duration_too_short` 等。

## LLM 裁决逻辑

以下任一条件满足时自动调用 LLM：

- 检测到人脸（`has_face: true`）
- 检测到人类步态（`has_human_gait: true`）
- 检测到人声（跨模态验证）
- 清晰度或连续性分数在阈值 ±0.1（`LLM_REVIEW_MARGIN`）范围内

LLM 可调用两个工具辅助判断：
- `get_key_frames`：获取指定帧的 JPEG 图像（Base64 编码）
- `get_imu_summary`：获取指定时间窗口的 IMU 数据统计

Claude API 异常时自动降级，仅使用检测器结果作为最终判定，不影响报告输出。

## 测试

```bash
# 运行全部测试
uv run pytest

# 仅运行检测器测试
uv run pytest tests/analyzers/

# 详细输出
uv run pytest -xvs tests/test_main.py
```

## 模型文件

`models/yunet.onnx`（~233 KB）已内置于仓库，无需运行时下载。
