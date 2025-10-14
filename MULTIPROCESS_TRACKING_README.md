# 多进程批量视频跟踪使用说明

## 功能介绍

`batch_track_videos_multiprocess.py` 是一个**多进程**批量视频跟踪脚本，可以：
- 🚀 **并行处理多个视频**（每个进程占用一个GPU）
- 🎯 **自动分配GPU资源**（支持多GPU负载均衡）
- 💾 从txt文件读取多个视频路径
- 📊 将跟踪结果保存到JSONL文件
- 🎬 可选：保存带标注的跟踪视频

## 性能对比

| 方式 | GPU使用 | 处理速度 | 适用场景 |
|------|---------|----------|----------|
| 单进程版本 | 1个GPU | 基准速度 | 视频数量少，或只有1个GPU |
| **多进程版本** | **多个GPU** | **N倍速度** | **视频数量多，有多个GPU** |

### 示例：处理10个视频
- **单进程（1个GPU）**: ~10分钟
- **多进程（4个GPU）**: ~2.5分钟 ⚡

## 使用方法

### 1. 准备视频列表文件

创建一个txt文件（如 `video_list.txt`），每行一个视频路径：

```
assets/car.mp4
assets/01_dog.mp4
assets/02_cups.mp4
assets/03_blocks.mp4
```

### 2. 快速开始

#### 使用脚本（推荐）

```bash
chmod +x run_batch_tracking_multiprocess.sh
./run_batch_tracking_multiprocess.sh
```

#### 手动运行

##### (1) 自动使用所有可用GPU

```bash
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl
```

##### (2) 指定使用特定GPU

```bash
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl \
    --gpu_ids 0 1 2  # 使用GPU 0, 1, 2
```

##### (3) 指定工作进程数

```bash
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl \
    --gpu_ids 0 1 \
    --num_workers 4  # 4个进程，GPU会循环使用
```

##### (4) 完整参数示例

```bash
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl \
    --output_video_dir tracked_videos \
    --sam_type sam2.1_hiera_large \
    --model_path models/sam2/checkpoints/sam2.1_hiera_large.pt \
    --detection_frequency 1 \
    --fps 5 \
    --gpu_ids 0 1 2 3 \
    --num_workers 4
```

### 3. 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--video_list` | ✓ | - | 包含视频路径的txt文件 |
| `--text_prompt` | ✓ | - | 目标检测的文本提示 |
| `--output_jsonl` | ✓ | - | 输出JSONL文件路径 |
| `--output_video_dir` | ✗ | None | 保存跟踪视频的目录 |
| `--sam_type` | ✗ | sam2.1_hiera_large | SAM模型类型 |
| `--model_path` | ✗ | models/sam2/checkpoints/sam2.1_hiera_large.pt | SAM模型权重路径 |
| `--detection_frequency` | ✗ | 1 | 检测频率 |
| `--max_frames` | ✗ | 60 | 内存中最大保留帧数 |
| `--fps` | ✗ | None | 处理帧率 |
| **`--gpu_ids`** | ✗ | None | **指定GPU ID列表（如：0 1 2）** |
| **`--num_workers`** | ✗ | None | **并行进程数（默认=GPU数量）** |

### 4. GPU分配策略

脚本会自动将视频**轮流分配**到不同GPU：

```
视频1 -> GPU 0
视频2 -> GPU 1
视频3 -> GPU 2
视频4 -> GPU 0  # 循环使用
视频5 -> GPU 1
...
```

#### 示例：4个GPU，10个视频

```bash
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl \
    --gpu_ids 0 1 2 3
```

分配情况：
- GPU 0: 视频1, 5, 9
- GPU 1: 视频2, 6, 10
- GPU 2: 视频3, 7
- GPU 3: 视频4, 8

## 输出格式

输出格式与单进程版本完全相同（参见 `BATCH_TRACKING_README.md`）。

### JSONL格式

```json
{
  "video_path": "assets/car.mp4",
  "text_prompt": "car",
  "video_info": {...},
  "frames": [...]
}
```

## 使用示例

### 示例1: 使用所有GPU处理视频

```bash
# 自动检测并使用所有可用GPU
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "person" \
    --output_jsonl person_results.jsonl
```

### 示例2: 使用指定GPU

```bash
# 只使用GPU 0和GPU 1
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl car_results.jsonl \
    --gpu_ids 0 1
```

### 示例3: 更多进程数（GPU复用）

```bash
# 2个GPU，但启动4个进程
# GPU 0运行2个进程，GPU 1运行2个进程
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "dog" \
    --output_jsonl dog_results.jsonl \
    --gpu_ids 0 1 \
    --num_workers 4
```

### 示例4: 保存跟踪视频

```bash
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl \
    --output_video_dir tracked_videos \
    --gpu_ids 0 1 2 3
```

## 性能优化建议

### 1. GPU配置
- **充分利用GPU**: 如果有4个GPU，使用 `--gpu_ids 0 1 2 3`
- **避免过载**: 不要让 `num_workers` 远大于GPU数量
- **显存管理**: 如果GPU显存不足，减少并发数或使用小模型

### 2. 处理速度优化
```bash
# 快速处理（低质量）
--fps 5 --detection_frequency 3 --sam_type sam2.1_hiera_tiny

# 平衡处理
--fps 10 --detection_frequency 1 --sam_type sam2.1_hiera_large

# 高质量处理
--fps 30 --detection_frequency 1 --sam_type sam2.1_hiera_large
```

### 3. 内存优化
- 降低 `--max_frames` 参数
- 不保存视频（省略 `--output_video_dir`）
- 使用较低的 `--fps`

## 监控和调试

### 查看GPU使用情况

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或使用
gpustat -i 1
```

### 日志输出

脚本会输出每个GPU的处理进度：

```
[GPU 0] Processing: assets/car.mp4
[GPU 1] Processing: assets/01_dog.mp4
[GPU 0] ✓ Completed: assets/car.mp4
[GPU 0]   - 150 frames in 5.23s (28.68 fps)
[GPU 0]   - 2 objects tracked
```

## 故障排查

### 问题1: CUDA out of memory

**解决方案**:
```bash
# 减少并发进程数
--num_workers 2

# 使用小模型
--sam_type sam2.1_hiera_tiny

# 降低帧率
--fps 5
```

### 问题2: 进程卡死

**原因**: 可能是GPU资源竞争
**解决方案**:
```bash
# 确保每个GPU只运行一个进程
--num_workers 4 --gpu_ids 0 1 2 3  # 4个GPU，4个进程
```

### 问题3: 某些视频处理失败

**说明**: 脚本会跳过失败的视频，继续处理其他视频
**检查**: 查看日志中的错误信息

## 与单进程版本对比

### 单进程版本 (`batch_track_videos.py`)
- ✅ 模型加载一次
- ✅ 内存占用小
- ❌ 只用1个GPU
- ❌ 速度慢（顺序处理）

### 多进程版本 (`batch_track_videos_multiprocess.py`)
- ✅ **多GPU并行**
- ✅ **速度快（N倍提升）**
- ✅ 自动负载均衡
- ⚠️ 每个进程加载一次模型（显存占用大）

### 选择建议

| 场景 | 推荐版本 |
|------|---------|
| 只有1个GPU | 单进程版本 |
| 有多个GPU | **多进程版本** |
| 视频数量少（<5） | 单进程版本 |
| 视频数量多（>10） | **多进程版本** |
| GPU显存紧张 | 单进程版本 |
| 需要快速处理 | **多进程版本** |

## 进阶用法

### 1. 按GPU能力分配任务

如果GPU性能不同，可以手动调整：

```bash
# GPU 0是RTX 4090，GPU 1是RTX 3080
# 让GPU 0处理更多任务
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl \
    --gpu_ids 0 0 1  # GPU 0分配2份，GPU 1分配1份
```

### 2. 分批处理大量视频

```bash
# 将大文件拆分
split -l 100 large_video_list.txt batch_

# 分别处理
for batch in batch_*; do
    python scripts/batch_track_videos_multiprocess.py \
        --video_list $batch \
        --text_prompt "car" \
        --output_jsonl results_${batch}.jsonl
done

# 合并结果
cat results_batch_*.jsonl > final_results.jsonl
```

### 3. 与任务调度系统集成

```bash
# SLURM示例
#!/bin/bash
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16

python scripts/batch_track_videos_multiprocess.py \
    --video_list $VIDEO_LIST \
    --text_prompt "$TEXT_PROMPT" \
    --output_jsonl $OUTPUT \
    --gpu_ids 0 1 2 3
```

## 技术细节

### 多进程架构

```
主进程
 ├── 读取视频列表
 ├── 分配GPU任务
 └── ProcessPoolExecutor
      ├── 进程1 (GPU 0) -> 加载模型 -> 处理视频1
      ├── 进程2 (GPU 1) -> 加载模型 -> 处理视频2
      ├── 进程3 (GPU 2) -> 加载模型 -> 处理视频3
      └── 进程4 (GPU 3) -> 加载模型 -> 处理视频4
```

### 关键特性

1. **进程隔离**: 每个进程独立加载模型，避免GPU冲突
2. **自动清理**: 处理完成后自动释放GPU内存
3. **异常处理**: 单个视频失败不影响其他视频
4. **结果收集**: 使用 `as_completed` 实时收集结果

## 常见问题

**Q: 多进程版本会加载多次模型吗？**
A: 是的，每个进程加载一次。但由于并行处理，总体速度更快。

**Q: 可以在单GPU上用多进程吗？**
A: 可以，但不推荐。会导致GPU资源竞争，效果可能更差。

**Q: 如何选择最佳进程数？**
A: 通常设置为GPU数量。如果GPU显存充足，可以设为GPU数量的2倍。

**Q: 输出顺序会乱吗？**
A: 输出顺序可能与输入不同，但每个视频的结果都是完整的。

## 参考资料

- [单进程版本文档](BATCH_TRACKING_README.md)
- [Python ProcessPoolExecutor文档](https://docs.python.org/3/library/concurrent.futures.html)
- [PyTorch多进程最佳实践](https://pytorch.org/docs/stable/notes/multiprocessing.html)
