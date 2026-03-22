# Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 CPU 占用过高和程序无响应，解决帧内存爆炸、巨型 IPC 序列化、HOG 参数过激和全分辨率运算四类根本问题。

**Architecture:** 四个独立改动，互不依赖可并行验证：(1) extractor 流式采样替代全量积累；(2) runner 用 ThreadPoolExecutor 替换 ProcessPoolExecutor，消除 pickle IPC；(3) visual worker 增加帧降分辨率步骤；(4) GaitDetector scale 参数调整。

**Tech Stack:** Python 3.12, OpenCV, NumPy, concurrent.futures, pytest

---

## 文件变更地图

| 文件 | 操作 | 变更内容 |
|------|------|---------|
| `agent/extractor.py` | Modify | 用流式计数器采样替代后处理 linspace |
| `agent/runner.py` | Modify | `ProcessPoolExecutor` → `ThreadPoolExecutor` |
| `agent/pipeline.py` | Modify | `_run_visual_worker` 增加 resize 步骤 |
| `agent/config.py` | Modify | 新增 `max_analysis_dim` 配置项 |
| `agent/analyzers/gait.py` | Modify | `scale=1.05` → `1.15`，`winStride=(8,8)` → `(16,16)` |
| `tests/test_extractor.py` | Modify | 更新 `test_frame_cap_applied_with_linspace`（旧逻辑已移除） |
| `tests/analyzers/test_gait.py` | Verify | 确认现有测试仍通过（参数改动不影响正确性） |

---

## Task 1: 流式帧采样（消除内存爆炸）

**Files:**
- Modify: `agent/extractor.py:100-168`
- Modify: `tests/test_extractor.py:327-345`

**背景：** 当前代码先把所有帧 decode 进 `raw_videos`，loop 结束后才做 `[::sample_rate]` + linspace。4 路 1080p 10 分钟录像 = 72,000 帧全在内存，远超实际需要。

**目标行为：** 在消息循环内用帧计数器决定是否 decode，超过 `max_frames_per_topic` 立即停止 decode（但继续迭代其他 topic 的消息）。

- [ ] **Step 1: 确认现有采样测试全部通过**

```bash
uv run pytest tests/test_extractor.py -v
```
期望：全部 PASS（建立基线）

- [ ] **Step 2: 修改 extractor.py — 用流式计数器替换全量积累**

将 `extract()` 方法的帧积累逻辑从：
```python
# 当前：先全量积累
raw_videos: dict[str, list[np.ndarray]] = {t: [] for t in video_topics}
# ...循环内无条件 append...
# 循环后再采样
frames = frames[::self._frame_sample_rate]
if len(frames) > self._max_frames_per_topic:
    indices = np.linspace(...)
    frames = [frames[i] for i in indices]
```

改为：
```python
# 新：流式计数器，循环内按需 decode
frame_counters: dict[str, int] = {t: 0 for t in video_topics}
videos: dict[str, list[np.ndarray]] = {t: [] for t in video_topics}

# 循环内（替换原有 raw_videos[topic].append(frame) 逻辑）:
if topic in videos:
    frame_counters[topic] += 1
    if (frame_counters[topic] % self._frame_sample_rate == 1
            and len(videos[topic]) < self._max_frames_per_topic):
        frame = self._registry.decode_image(schema.name, decoded_message)
        if frame is not None:
            videos[topic].append(frame)
```

注意：`% sample_rate == 1`（1-indexed）确保 rate > 总帧数时仍返回第 1 帧，与原行为 `[::rate]` 一致。

循环后删除旧的 post-loop 采样块（`frames[::self._frame_sample_rate]` 和 linspace 部分），直接使用 `videos` 字典。min_frames 检查保留：

```python
# Post-loop: 仅保留 min_frames 检查（linspace 块整体删除）
extraction_warnings: dict[str, str] = {}
for topic, frames in videos.items():
    if 0 < len(frames) < self._min_frames:
        logger.warning(...)
        extraction_warnings[topic] = "below_min_frames"
        videos[topic] = []
```

- [ ] **Step 3: 运行采样测试**

```bash
uv run pytest tests/test_extractor.py -v
```

期望：
- `test_frame_sample_rate_reduces_frame_count` PASS（25帧 rate=5 → 5帧）
- `test_frame_sample_rate_1_returns_all_frames` PASS（10帧 → 10帧）
- `test_frame_sample_rate_larger_than_frame_count_returns_one_frame` PASS（5帧 rate=100 → 1帧）
- `test_frame_cap_applied_with_linspace` — 此测试直接测旧 linspace 逻辑，已与 extractor 解耦，**跳过不影响**

- [ ] **Step 4: 更新 `test_frame_cap_applied_with_linspace` — 改测流式上限行为**

```python
def test_max_frames_per_topic_caps_output():
    """流式采样：max_frames_per_topic 作为硬性上限，不超过该数量。"""
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    reg = _image_registry(frame)
    # 生成 20 帧，rate=1，上限=5
    tuples = [
        _make_image_tuple("/cam", i * 1_000_000_000, SimpleNamespace())
        for i in range(20)
    ]
    extractor = McapExtractor(
        camera_topics=["/cam"],
        audio_topics=[],
        frame_sample_rate=1,
        min_frames=1,
        max_frames_per_topic=5,
        registry=reg,
    )
    p1, p2, p3 = _patch_extractor(tuples, image_topics=["/cam"], audio_topics=[])
    with p1, p2, p3:
        data = extractor.extract("fake.mcap")
    assert len(data["videos"]["/cam"]) == 5
```

- [ ] **Step 5: 运行全部 extractor 测试**

```bash
uv run pytest tests/test_extractor.py -v
```
期望：全部 PASS

- [ ] **Step 6: Commit**

```bash
git add agent/extractor.py tests/test_extractor.py
git commit -m "perf(extractor): stream-sample frames during iteration to prevent memory explosion"
```

---

## Task 2: ProcessPoolExecutor → ThreadPoolExecutor（消除 IPC 序列化）

**Files:**
- Modify: `agent/runner.py:183`（一行改动）

**背景：** ProcessPoolExecutor 要将每个 topic 的 frames（numpy array list）pickle → OS 管道 → 子进程 unpickle，300 帧 × 1080p ≈ 1.8 GB/topic 序列化负载。OpenCV 的重型运算（Farneback、HOG、YuNet、Laplacian、FFT）均主动释放 GIL，线程池可实现真正 CPU 并发。

- [ ] **Step 1: 确认现有 runner/pipeline 测试通过**

```bash
uv run pytest tests/test_pipeline.py -v
```
期望：全部 PASS

- [ ] **Step 2: 修改 runner.py — 替换 executor 类型**

```python
# agent/runner.py
# 改前：
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# ...
with ProcessPoolExecutor(max_workers=settings.max_concurrent_topics) as executor:

# 改后（只改这一处）：
from concurrent.futures import ThreadPoolExecutor
# ...
with ThreadPoolExecutor(max_workers=settings.max_concurrent_topics) as executor:
```

同时删除 `ProcessPoolExecutor` 的 import（如果不再使用）。`_run_visual_worker` 和 `_run_audio_worker` 函数签名无需改变，仍在 pipeline.py 中保留（线程中运行同样有效）。

- [ ] **Step 3: 运行全部测试**

```bash
uv run pytest -x -v
```
期望：全部 PASS

- [ ] **Step 4: Commit**

```bash
git add agent/runner.py
git commit -m "perf(runner): replace ProcessPoolExecutor with ThreadPoolExecutor to eliminate numpy IPC serialization"
```

---

## Task 3: 分析前帧降分辨率（减少 80-90% 计算量）

**Files:**
- Modify: `agent/config.py`（新增 `max_analysis_dim`）
- Modify: `agent/pipeline.py:_run_visual_worker`（resize 注入）

**背景：** Farneback 光流、FFT、HOG、YuNet 均在原始分辨率运算。降到 640px 短边后，像素数减少约 1/9，计算量对应降低 80-90%，对清晰度/连续性判断结论无实质影响。

注意：resize 仅作用于分析时的副本，`camera_intermediates["frames"]` 保持原始分辨率以供 LLM judge 使用。

- [ ] **Step 1: 在 config.py 新增配置项**

```python
# agent/config.py — 在 max_frames_per_topic 字段后添加
max_analysis_dim: int = Field(default=640, gt=0)
```

- [ ] **Step 2: 在 pipeline.py 的 `_run_visual_worker` 添加 resize 步骤**

```python
# agent/pipeline.py — _run_visual_worker 函数内，在 pipeline.run_visual(frames) 调用前插入

def _resize_frames(frames: list[np.ndarray], max_dim: int) -> list[np.ndarray]:
    """Resize frames so that max(h, w) <= max_dim. Returns new list; does not mutate input."""
    if max_dim <= 0:
        return frames
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        if max(h, w) <= max_dim:
            resized.append(f)
        else:
            scale = max_dim / max(h, w)
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            resized.append(cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA))
    return resized


def _run_visual_worker(
    topic: str,
    frames: list[np.ndarray],
    model_path: str,
    max_analysis_dim: int = 640,   # 新增参数
) -> tuple[str, dict[str, Any], list[str]]:
    import cv2
    from agent.analyzers.clarity import ClarityAnalyzer
    from agent.analyzers.continuity import ContinuityAnalyzer
    from agent.analyzers.face import FaceDetector
    from agent.analyzers.gait import GaitDetector

    analysis_frames = _resize_frames(frames, max_analysis_dim)  # 新增

    pipeline = AnalysisPipeline(
        visual_analyzers=[
            ClarityAnalyzer(),
            ContinuityAnalyzer(),
            FaceDetector(model_path=model_path),
            GaitDetector(),
        ],
        audio_analyzers=[],
    )
    results, errors = pipeline.run_visual(analysis_frames)   # 用 analysis_frames
    return topic, results, errors
```

- [ ] **Step 3: 在 runner.py 传入 max_analysis_dim 参数**

```python
# agent/runner.py — executor.submit 调用处
cam_futures = {
    executor.submit(_run_visual_worker, topic, frames, _model_path,
                    settings.max_analysis_dim): topic   # 新增参数
    for topic, frames in data["videos"].items()
    if frames and topic not in camera_intermediates
}
```

- [ ] **Step 4: 运行全部测试**

```bash
uv run pytest -x -v
```
期望：全部 PASS

- [ ] **Step 5: Commit**

```bash
git add agent/config.py agent/pipeline.py agent/runner.py
git commit -m "perf(pipeline): resize frames to max_analysis_dim before CV analysis to reduce compute 80-90%"
```

---

## Task 4: 修正 GaitDetector HOG 参数（降低金字塔层数 65%）

**Files:**
- Modify: `agent/analyzers/gait.py:29-31`

**背景：** `scale=1.05` 在 1080p 下产生约 44 层金字塔，`winStride=(8,8)` 步长极小，单帧 HOG 扫描时间极长。改为 `scale=1.15, winStride=(16,16)` 后层数降至约 15 层，检测精度对本业务场景（有无行人，非精确定位）影响可忽略。

- [ ] **Step 1: 确认现有 gait 测试通过**

```bash
uv run pytest tests/analyzers/test_gait.py -v
```
期望：全部 PASS

- [ ] **Step 2: 修改 gait.py 参数**

```python
# agent/analyzers/gait.py:29-31
# 改前：
rects, weights = self._hog.detectMultiScale(
    frame, winStride=(8, 8), padding=(4, 4), scale=1.05
)

# 改后：
rects, weights = self._hog.detectMultiScale(
    frame, winStride=(16, 16), padding=(8, 8), scale=1.15
)
```

- [ ] **Step 3: 运行 gait + 全部 analyzer 测试**

```bash
uv run pytest tests/analyzers/ -v
```
期望：全部 PASS（scale 改动不影响有无行人的正确性判断）

- [ ] **Step 4: Commit**

```bash
git add agent/analyzers/gait.py
git commit -m "perf(gait): tune HOG scale 1.05->1.15 and winStride (8,8)->(16,16) to reduce pyramid layers ~65%"
```

---

## Task 5: 全量回归验证

- [ ] **Step 1: 运行完整测试套件**

```bash
uv run pytest -v --tb=short 2>&1 | tail -30
```
期望：全部 PASS，无回归

- [ ] **Step 2: 检查 config 环境变量文档**

确认 `.env.example` 中有 `MAX_ANALYSIS_DIM` 说明（可选，有默认值 640）：

```bash
grep -n "MAX_ANALYSIS_DIM\|max_analysis_dim" .env.example || echo "需要添加"
```

若缺失，在 `.env.example` 末尾追加：
```
# MAX_ANALYSIS_DIM=640    # 分析用帧的最大边长（像素），降低 CV 计算量
```

- [ ] **Step 3: 最终 commit**

```bash
git add .env.example
git commit -m "docs: add MAX_ANALYSIS_DIM to env example"
```

---

## 预期收益

| 改动 | 问题 | 预期改善 |
|------|------|---------|
| Task 1 流式采样 | 内存爆炸 → OOM → 系统 swap | 内存从全量帧降到采样帧，4路1080p从~64GB→~200MB |
| Task 2 ThreadPool | pickle 序列化阻塞主进程 | 消除 1.8GB/topic 的序列化延迟，响应性立即改善 |
| Task 3 帧降分辨率 | 全分辨率 FFT/光流/HOG | 每个分析器计算量降低约 85% |
| Task 4 HOG 参数 | 44 层金字塔 per 帧 | 金字塔层数从 44 降到 15，单帧 HOG 时间降 65% |
