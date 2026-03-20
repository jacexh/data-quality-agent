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
| 音频 O(n²) 拼接 | `raw_audio += chunk` 字节拼接 | 大文件时内存分配指数增长 |
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

改动集中在 3 个文件（`config.py`、`extractor.py`、`main.py`）。`base.py`、`analyzers/`、`pipeline.py` 完全不动。

---

### 1. 帧采样（extractor.py）

**问题根源**：`McapExtractor.extract()` 将所有帧装入 `frames: list[np.ndarray]`，内存为 O(文件大小)。

**解决方案**：统一采样，`ExtractedData.frames` 只包含采样后的帧，`base.py` 不变。

#### 统一采样策略

所有视觉 analyzer（clarity、continuity、face、gait）共用同一份采样帧：

```python
sampled_frames = raw_frames[::frame_sample_rate]
```

`ExtractedData.frames` = `sampled_frames`，类型 `list[np.ndarray]` 不变，`base.py` 不改动。

**voice 不采样**：音频帧已是 30ms/960B 切片，1 小时录制总量仅 ~216 MB。

**采样率选择**：`FRAME_SAMPLE_RATE` 越大内存越小，但 clarity/continuity 分辨率降低、face/gait 检出率略降。对于人脸/步态检测，`sample_rate=30`（每秒 1 帧）在 10 分钟录制中产生 600 帧，足以检测人员是否出现。

#### 连续性采样的语义变化

`ContinuityAnalyzer` 使用 Farneback 光流计算相邻帧运动量。采样后，相邻帧间距从 1 帧变为 `FRAME_SAMPLE_RATE` 帧，光流幅值随间距扩大而增大。

- **检测语义**：从"逐帧微抖动"变为"宏观跳变"检测，对录制中断/画面跳帧仍有效。
- **阈值影响**：`CONTINUITY_THRESHOLD=0.6` 可能需要向上调整（避免假通过）。
- **重标定方法**：对已知"连续"和"跳帧"样本各测 20 个，取分数交叉点作为新阈值。
- **重标定为已知后续动作**，不阻塞本次实现。

#### 音频拼接优化

```python
# 现有（O(n²)）— 每次 += 产生新 bytes 对象
raw_audio: bytes = b""
raw_audio += chunk

# 优化后（O(n)）— 原地扩展，最终一次转换
_buf = bytearray()
_buf.extend(chunk)
raw_audio: bytes = bytes(_buf)  # bytes(bytearray) 等价于原来的 raw_audio，与 chunk_pcm() 入参兼容
```

音频不经采样，`chunk_pcm(raw_audio)` 行为与现有一致，byte 输出逐字节等价。

#### 内存效果（10min 1080p，18,000 帧，每帧 ≈ 6 MB）

> 每帧大小：1920 × 1080 × 3 bytes ≈ **6 MB**

| 配置 | 采样帧数 | 峰值帧内存（总计） |
|------|---------|-----------------|
| 优化前 | 18,000 | **108 GB** |
| sample_rate=5 | 3,600 | **21.6 GB** |
| sample_rate=30（推荐起点） | 600 | **3.6 GB** |

---

### 2. asyncio.Queue 有界队列 + Worker（main.py）

**问题根源**：`BackgroundTasks.add_task()` 无界堆积，无背压。

**解决方案**：**完全移除 `BackgroundTasks`**，替换为模块级有界 `asyncio.Queue` + 固定数量 worker coroutine。`notify` 函数签名同时移除 `background_tasks: BackgroundTasks` 参数。

```python
# 替换前（移除）
async def notify(request: Request, background_tasks: BackgroundTasks):
    ...
    background_tasks.add_task(_process_mcap, bucket=bucket, key=key)

# 替换后
async def notify(request: Request):
    ...
    try:
        _queue.put_nowait((bucket, key))
    except asyncio.QueueFull:
        logger.warning(f"Queue full, rejecting {key}")
        return JSONResponse({"status": "queue_full"}, status_code=429)
    return JSONResponse({"status": "accepted"})
```

#### Worker coroutine

每个 worker 在 lifespan 启动时接收独属的 `s3_client` 实例（见 S3 客户端节），持续消费队列：

```python
async def _worker(s3_client) -> None:
    while True:
        bucket, key = await _queue.get()
        # try/finally 包裹每项处理，保证 _processing 清理
        try:
            await asyncio.to_thread(_process_and_log, s3_client, bucket, key)
        except Exception as exc:
            logger.error(f"Worker failed for {key}: {exc}")
            # 失败项目被丢弃（见已知限制）
        finally:
            # finally 在 per-item try/finally 内，覆盖正常完成、异常、任务取消三种情况
            _processing.discard(key)
            _queue.task_done()
```

**`_processing` 清理保证**：`_processing.discard(key)` 在 per-item 的 `finally` 块内，不在 `while True` 的外层。`asyncio.CancelledError`（lifespan 关闭时触发）在 `await asyncio.to_thread(...)` 处抛出，同样被 `finally` 捕获。`_queue.get()` 阻塞期间的取消不涉及任何 key，无清理需求。

#### FastAPI lifespan 管理 Worker 生命周期

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 每个 worker 独占一个 S3 client 实例，via 参数传入，无共享
    clients = [_make_s3_client() for _ in range(settings.worker_count)]
    tasks = [
        asyncio.create_task(_worker(client))
        for client in clients
    ]
    yield
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

app = FastAPI(lifespan=lifespan)
```

#### S3 客户端：每 Worker 独立实例

```python
def _make_s3_client():
    """每个 worker 调用一次，返回独占 client 实例。"""
    return boto3.client(
        "s3",
        endpoint_url=f"http{'s' if settings.minio_use_ssl else ''}://{settings.minio_endpoint}",
        aws_access_key_id=settings.minio_access_key,
        aws_secret_access_key=settings.minio_secret_key,
        config=Config(signature_version="s3v4", max_pool_connections=4),
    )
```

每个 worker coroutine 通过 `asyncio.to_thread(_process_and_log, s3_client, ...)` 调用其独占 client，在单一线程中串行执行，完全规避 boto3 并发线程安全问题。总 S3 client 数 = `WORKER_COUNT`。

#### 线程数分析

| 阶段 | 线程来源 | 峰值数量（WORKER_COUNT=4） |
|------|---------|--------------------------|
| S3 下载 | `asyncio.to_thread` 线程池 | ≤ 4 |
| CV 分析 | `pipeline.py` ThreadPoolExecutor(max_workers=5) | ≤ 20 |
| 重叠期峰值 | 两者叠加 | ≤ 24 |

**建议**：`WORKER_COUNT = max(1, CPU核数 // 2)`，使峰值线程数约为 CPU 核数 × 3。

#### 已知限制：失败项目被丢弃

Worker 处理异常时当前项目被丢弃，无死信队列（保守技术栈约束）。通过 error 日志可观测，需人工重触发或定期扫描 bucket 对账补偿。

---

### 3. 处理中去重（main.py）

```python
_processing: set[str] = set()

async def notify(request: Request):
    ...
    # check（in 判断） → add → put_nowait 均为同步操作，无 await
    # asyncio 协作式调度保证单进程内原子性
    if key in _processing:
        logger.info(f"Duplicate webhook for {key}, skipping")
        return JSONResponse({"status": "duplicate"})
    _processing.add(key)
    _queue.put_nowait((bucket, key))
    return JSONResponse({"status": "accepted"})
```

**并发安全**：asyncio 协作式调度，coroutine 仅在 `await` 处让出。`if key in _processing` → `_processing.add(key)` → `_queue.put_nowait()` 全程无 `await`，单进程事件循环内原子。

**多实例限制**：各实例独立 set，跨实例不去重。MinIO 对同一 key 仅发一次 webhook，重复率极低，可接受。

---

### 4. 水平扩展（nginx + docker-compose）

```yaml
# docker-compose.yml 新增服务
nginx:
  image: nginx:alpine
  ports: ["8000:80"]
  volumes: ["./nginx.conf:/etc/nginx/nginx.conf:ro"]
  depends_on: [agent]

# agent 修改：去掉外部端口暴露
agent:
  expose: ["8000"]
  environment:
    FRAME_SAMPLE_RATE: "30"
    MAX_QUEUE_SIZE: "100"
    WORKER_COUNT: "4"
```

```nginx
upstream agents {
    server agent:8000;  # Docker DNS 自动解析多实例 IP
}
server {
    listen 80;
    location / {
        proxy_pass http://agents;
        proxy_next_upstream error timeout http_429;
        # 某实例返回 429（队列满）时，nginx 自动重试其他实例
        # 所有实例均 429 时，nginx 向 MinIO 返回 502
        # MinIO webhook 默认在 5xx 时重试，502 触发重发，不丢失文件
    }
}
```

扩容命令：`docker compose up --scale agent=4`

---

## 数据流

```
MinIO
  │  (s3:ObjectCreated:*, .mcap only)
  ▼
Nginx（轮询 + http_429 重试；全满时 502，MinIO 5xx 重试兜底）
  ▼
POST /notify（FastAPI，无 BackgroundTasks）
  │  check-then-add _processing（无 await，单进程原子）
  │  asyncio.Queue.put_nowait() → QueueFull → 429
  ▼
asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
  ▼  (WORKER_COUNT 个 worker coroutine，各持独占 S3 client)
asyncio.to_thread(_process_and_log, s3_client, ...)
  │  s3_client.download_file(bucket, key, local_path)
  ▼
McapExtractor（统一采样：raw_frames[::FRAME_SAMPLE_RATE]）
  ├─ frames: 采样后，类型 list[np.ndarray] 不变
  ├─ audio_frames: bytearray 拼接 → bytes()，全量不采样
  └─ sensor_series: 不变
  ▼
AnalysisPipeline（ThreadPoolExecutor(max_workers=5)，不变）
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
| `WORKER_COUNT` | `4` | 建议 = `max(1, CPU核数 // 2)` |
| `FRAME_SAMPLE_RATE` | `30` | 所有视觉 analyzer：每 N 帧取 1 帧（调大降内存，调小提精度） |

---

## 错误处理

| 场景 | 行为 | HTTP 响应 / 说明 |
|------|------|-----------------|
| 队列已满 | 拒绝入队，warning 日志 | `429 {"status":"queue_full"}` |
| 重复 key（单实例） | 跳过，info 日志 | `200 {"status":"duplicate"}` |
| 所有实例队列均满 | nginx 返回 502，MinIO 5xx 重试兜底 | 不丢失文件 |
| worker 意外 crash | per-item finally 清理 _processing，项目丢弃，error 日志 | 需人工重触发 |
| S3 下载失败 | 现有行为（error report） | 不变 |
| MCAP 解析失败 | 现有行为（error report） | 不变 |
| 单 Analyzer 异常 | 现有行为（null + analyzer_errors） | 不变 |
| LLM API 失败 | 现有行为（降级到 detector 结果） | 不变 |

---

## 测试策略

### 新增测试（test_main.py）

| 用例 | 验证点 |
|------|--------|
| 队列满时返回 429 | mock `_queue.put_nowait` 抛 `QueueFull` → `status_code=429` |
| 重复 key 返回 duplicate | 预填 `_processing` → `status:"duplicate"`，set 大小不变 |
| worker 异常后 _processing 清理 | `asyncio.to_thread` mock 抛异常 → finally 执行 → key 不在 `_processing` |
| lifespan 启动正确 worker 数量 | lifespan 后 task 数 == `settings.worker_count` |
| `notify` 不含 `BackgroundTasks` 参数 | 函数签名检查或调用时不传该参数 |

### 新增测试（test_extractor.py）

| 用例 | 验证点 |
|------|--------|
| `sample_rate=1` | `len(frames)` == 原始帧数 |
| `sample_rate=5`，原始 25 帧 | `len(frames)` == 5 |
| `sample_rate=30`，原始帧 < 30 | `len(frames)` == 1（至少 1 帧，不崩溃） |
| voice 不采样 | `len(audio_frames)` == 全量 PCM 帧数，与 `sample_rate` 无关 |
| bytearray 等价性 | 优化后 `raw_audio` bytes 内容与 `+=` 拼接结果逐字节一致 |

### 不变

- `analyzers/test_*.py` — 接口不变，全部保留
- `test_pipeline.py`、`test_llm_judge.py`、`test_report.py` — 全部保留

---

## 已知限制与后续动作

| 限制 | 说明 | 缓解 / 后续动作 |
|------|------|----------------|
| CONTINUITY_THRESHOLD 需重标定 | 帧采样后光流幅值分布变化，默认 0.6 可能过宽松 | 用真实样本测定新阈值后更新配置默认值 |
| worker crash 丢失项目 | 无死信队列（保守技术栈约束） | error 日志可观测；定期扫描 bucket 对账补偿 |
| 去重仅单实例有效 | 多实例无共享状态 | MinIO 单次发送，重复率极低，可接受 |

---

## 变更文件清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `agent/config.py` | 修改 | 新增 3 个配置项（`MAX_QUEUE_SIZE`、`WORKER_COUNT`、`FRAME_SAMPLE_RATE`） |
| `agent/extractor.py` | 修改 | 统一帧采样（`raw_frames[::FRAME_SAMPLE_RATE]`）+ bytearray 音频拼接 |
| `agent/main.py` | 修改 | 移除 `BackgroundTasks`；asyncio.Queue + lifespan worker + 每 worker 独立 S3 client + 去重 |
| `nginx.conf` | 新增 | 轮询负载均衡 + 429 重试 |
| `docker-compose.yml` | 修改 | 新增 nginx 服务，agent 去掉端口暴露，新增环境变量 |
| `agent/base.py` | **不变** | — |
| `agent/pipeline.py` | **不变** | — |
| `agent/analyzers/*` | **不变** | — |
| `agent/llm_judge.py` | **不变** | — |
| `agent/report.py` | **不变** | — |
