# Data Quality Agent — TB/PB 规模性能优化设计

**Date:** 2026-03-20
**Status:** Approved
**Scope:** 在保守技术栈（FastAPI + MinIO + Docker Compose）约束下，解决大文件 + 高并发场景下的性能缺陷

---

## 背景与问题

现有设计为单机单文件处理模型，在 TB/PB 规模（大文件 + 高并发）下存在以下致命缺陷：

| 缺陷 | 现象 | 影响 |
|------|------|------|
| 全量帧加载内存 | `frames: list[np.ndarray]` 堆积所有帧 | 10min 1080p 录制 ≈ 108 GB RAM → OOM |
| 音频 O(n²) 拼接 | `raw_audio += chunk` | 大文件时内存分配指数增长 |
| BackgroundTasks 无界堆积 | 无背压机制 | 高并发 webhook → 无限任务堆积 → OOM |
| S3 连接每次重建 | `boto3.client()` per request | 高并发下连接数爆炸 |
| 无去重 | 同一 key 多次 webhook | 重复处理浪费资源 |

---

## 约束条件

- **技术栈**：FastAPI + MinIO + Docker Compose，不引入 Kafka/Redis 等新组件
- **规模**：单个 MCAP 文件可达几十 GB + 每天大量文件高并发上传
- **延迟**：近实时，文件上传后几分钟内出报告
- **持久化**：loguru JSON 日志输出（不变）
- **部署**：允许多实例水平扩展

---

## 优化方案

### 核心原则

改动集中在 3 个文件（`config.py`、`extractor.py`、`main.py`），`analyzers/` 及 `pipeline.py` 完全不动。

### 1. 帧采样（extractor.py）

**问题根源**：`McapExtractor.extract()` 将所有帧装入 `frames: list[np.ndarray]`，内存为 O(文件大小)。

**解决方案**：在提取时采样，`ExtractedData.frames` 只包含采样后的帧，Analyzer 接口不变。

#### 采样策略

| Analyzer | 策略 | 配置项 | 原理 |
|----------|------|--------|------|
| `clarity` | 每 N 帧取 1 帧（均匀间隔） | `FRAME_SAMPLE_RATE=5` | Laplacian/Tenengrad 逐帧独立，均匀采样反映整体分布 |
| `continuity` | 每 N 帧取 1 帧（保留相邻关系） | `FRAME_SAMPLE_RATE=5` | 相邻采样帧做光流，仍可检测大幅跳变 |
| `face` | 均匀抽取 K 帧 | `MAX_DETECTION_FRAMES=60` | 人脸出现一帧即触发，无需全量扫描 |
| `gait` | 均匀抽取 K 帧（同 face） | `MAX_DETECTION_FRAMES=60` | HOG 同理 |
| `voice` | 不采样 | — | 音频帧已是 30ms/960B 切片，1 小时总量仅 ~216 MB |

采样在 `McapExtractor` 内完成，新增 `frame_sample_rate: int` 和 `max_detection_frames: int` 参数，对下游完全透明。

#### 音频拼接优化

```python
# 现有（O(n²)）
raw_audio += chunk

# 优化后（O(n)）
_buf = bytearray()
_buf.extend(chunk)
raw_audio = bytes(_buf)  # 最终一次转换
```

#### 内存效果（10min 1080p 录制，18,000 帧）

| 配置 | 帧数 | 峰值内存 |
|------|------|---------|
| 优化前 | 18,000 | ~108 GB |
| sample_rate=5 | 3,600 | ~21.6 GB |
| sample_rate=30（推荐） | 600 | ~3.6 GB |

### 2. asyncio.Queue 有界队列 + Worker（main.py）

**问题根源**：`BackgroundTasks.add_task()` 无界堆积，无背压。

**解决方案**：模块级有界 `asyncio.Queue`，固定数量 worker task 消费队列，队列满时返回 429。

```python
_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=settings.max_queue_size)

@app.post("/notify")
async def notify(...):
    ...
    try:
        _queue.put_nowait((bucket, key))
    except asyncio.QueueFull:
        return JSONResponse({"status": "queue_full"}, status_code=429)
    return JSONResponse({"status": "accepted"})

async def _worker():
    while True:
        bucket, key = await _queue.get()
        try:
            await asyncio.to_thread(_analyze_and_log, source_file, bucket, local_path)
        except Exception:
            ...
        finally:
            _processing.discard(key)
            _queue.task_done()
```

#### FastAPI lifespan 管理 Worker 生命周期

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks = [asyncio.create_task(_worker()) for _ in range(settings.worker_count)]
    yield
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

app = FastAPI(lifespan=lifespan)
```

Worker 生命周期与应用绑定，优雅关闭时等待正在处理的任务完成。

### 3. 处理中去重（main.py）

```python
_processing: set[str] = set()

@app.post("/notify")
async def notify(...):
    if key in _processing:
        return JSONResponse({"status": "duplicate"})
    _processing.add(key)
    _queue.put_nowait((bucket, key))
    ...

async def _worker():
    ...
    finally:
        _processing.discard(key)  # 保证清理
```

单实例内有效。多实例各自独立去重（可接受：MinIO webhook 通常只发送一次）。

### 4. S3 客户端单例（main.py）

```python
# 模块顶层，应用启动时创建一次
_s3_client = boto3.client(
    "s3",
    endpoint_url=...,
    aws_access_key_id=settings.minio_access_key,
    aws_secret_access_key=settings.minio_secret_key,
    config=Config(
        signature_version="s3v4",
        max_pool_connections=settings.s3_max_pool_connections,
    ),
)
```

### 5. 水平扩展（nginx + docker-compose）

新增 nginx 服务作负载均衡，`proxy_next_upstream http_429` 使某实例队列满时自动重试其他实例。

```yaml
# docker-compose.yml 新增
nginx:
  image: nginx:alpine
  ports: ["8000:80"]
  volumes: ["./nginx.conf:/etc/nginx/nginx.conf:ro"]
  depends_on: [agent]

# agent 服务去掉端口暴露
agent:
  expose: ["8000"]  # 仅内网可达
```

```nginx
# nginx.conf
upstream agents {
    server agent:8000;  # Docker DNS 自动解析多实例
}
server {
    listen 80;
    location / {
        proxy_pass http://agents;
        proxy_next_upstream error timeout http_429;
    }
}
```

扩容命令：`docker compose up --scale agent=4`

---

## 数据流

```
MinIO
  │  (bucket notification: s3:ObjectCreated:*, .mcap only)
  ▼
Nginx（轮询 + 429 自动重试）
  ▼
Webhook Server (FastAPI, POST /notify)
  │  去重检查（_processing set）
  │  asyncio.Queue.put_nowait() → QueueFull → 429
  ▼
asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
  ▼  (WORKER_COUNT 个 worker task 并发消费)
S3 单例 client → download_file（连接池复用）
  ▼
McapExtractor（采样模式）
  ├─ frames: 每 FRAME_SAMPLE_RATE 帧取 1，最多 MAX_DETECTION_FRAMES 帧
  ├─ audio_frames: bytearray 拼接，全量 PCM 帧
  └─ sensor_series: 不变
  ▼
AnalysisPipeline（ThreadPoolExecutor，不变）
  ▼
LLMJudge（不变）
  ▼
ReportBuilder → loguru JSON 输出（不变）
```

---

## 新增配置项（config.py）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_QUEUE_SIZE` | `100` | 队列满时返回 429 |
| `WORKER_COUNT` | `4` | 并发 worker task 数量（建议 = CPU 核数） |
| `FRAME_SAMPLE_RATE` | `5` | clarity/continuity：每 N 帧取 1 帧 |
| `MAX_DETECTION_FRAMES` | `60` | face/gait：最多检测帧数 |
| `S3_MAX_POOL_CONNECTIONS` | `20` | boto3 连接池大小 |

---

## 错误处理

| 场景 | 行为 | HTTP 响应 |
|------|------|-----------|
| 队列已满 | 拒绝入队，记录 warning | `429 {"status":"queue_full"}` |
| 重复 key | 跳过，记录 info | `200 {"status":"duplicate"}` |
| worker 内部异常 | finally 清理 _processing，worker 继续运行 | 无（异步 worker） |
| S3 下载失败 | 现有行为（error report） | 无 |
| MCAP 解析失败 | 现有行为（error report） | 无 |
| 单 Analyzer 异常 | 现有行为（null + analyzer_errors） | 无 |
| LLM API 失败 | 现有行为（降级到 detector 结果） | 无 |

---

## 测试策略

### 新增测试（test_main.py）

| 用例 | 验证点 |
|------|--------|
| 队列满时返回 429 | `QueueFull` → `status_code=429` |
| 重复 key 返回 duplicate | `_processing` set 命中 → `status:"duplicate"` |
| worker 异常后 _processing 清理 | finally 块执行，set 不泄漏 |
| lifespan 启动/关闭 worker | worker task 创建与 cancel |

### 新增测试（test_extractor.py）

| 用例 | 验证点 |
|------|--------|
| `sample_rate=1` | 返回全帧 |
| `sample_rate=5` | 返回 ⌊N/5⌋ 帧 |
| `max_detection_frames=10` | 帧数不超 10 |
| 大文件场景 | 帧数上限生效 |

### 不变

- `analyzers/test_*.py` — 全部保留，接口不变
- `test_pipeline.py`、`test_llm_judge.py`、`test_report.py` — 全部保留

---

## 变更文件清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `agent/config.py` | 修改 | 新增 5 个配置项 |
| `agent/extractor.py` | 修改 | 帧采样 + bytearray 音频拼接 |
| `agent/main.py` | 修改 | asyncio.Queue + lifespan worker + S3 单例 + 去重 |
| `nginx.conf` | 新增 | 轮询负载均衡 + 429 重试 |
| `docker-compose.yml` | 修改 | 新增 nginx 服务，agent 去掉端口暴露，新增环境变量 |
| `agent/pipeline.py` | **不变** | — |
| `agent/analyzers/*` | **不变** | — |
| `agent/llm_judge.py` | **不变** | — |
| `agent/report.py` | **不变** | — |
