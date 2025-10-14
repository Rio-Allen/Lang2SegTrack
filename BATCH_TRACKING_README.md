# 批量视频跟踪脚本使用说明

## 功能介绍

### `batch_track_videos.py` - 单进程版本
- 从txt文件读取多个视频路径
- **只加载一次模型**（节省时间）
- 对每个视频进行目标跟踪
- 将跟踪结果保存到JSONL文件
- 可选：保存带标注的跟踪视频

### `batch_track_videos_multiprocess.py` - 多进程版本 ⭐
- **多进程并行处理**，每个进程独占一个GPU
- 自动检测可用GPU并分配任务
- **断点重续**功能，跳过已处理的视频
- **按对象组织**输出格式，更便于轨迹分析
- 其他功能同单进程版本

## 使用方法

### 1. 准备视频列表文件

创建一个txt文件（如 `video_list.txt`），每行一个视频路径：

```
assets/car.mp4
assets/01_dog.mp4
assets/02_cups.mp4
```

支持相对路径或绝对路径。

### 2. 运行脚本

#### 单进程版本（基础用法）

```bash
python scripts/batch_track_videos.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl
```

#### 多进程版本（推荐，自动使用所有GPU）⭐

```bash
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl \
    --output_video_dir tracked_videos
```

#### 多进程版本（指定GPU）

```bash
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl \
    --output_video_dir tracked_videos \
    --gpu_ids 0 1 2 \
    --num_workers 3 \
    --fps 5
```

#### 断点重续

```bash
# 首次运行
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl

# 中断后重新运行，自动跳过已完成的视频
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl

# 禁用断点重续，重新处理所有视频
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl \
    --no-resume
```

### 3. 参数说明

#### 单进程版本参数

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--video_list` | ✓ | - | 包含视频路径的txt文件 |
| `--text_prompt` | ✓ | - | 目标检测的文本提示（如"car", "person"） |
| `--output_jsonl` | ✓ | - | 输出JSONL文件路径 |
| `--output_video_dir` | ✗ | None | 保存跟踪视频的目录（不指定则不保存视频） |
| `--sam_type` | ✗ | sam2.1_hiera_large | SAM模型类型 |
| `--model_path` | ✗ | models/sam2/checkpoints/sam2.1_hiera_large.pt | SAM模型权重路径 |
| `--device` | ✗ | cuda:0 | GPU设备 |
| `--detection_frequency` | ✗ | 1 | 检测频率（每N帧检测一次） |
| `--max_frames` | ✗ | 60 | 内存中最大保留帧数 |
| `--fps` | ✗ | None | 处理帧率（None=使用原始视频帧率） |

#### 多进程版本参数

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--video_list` | ✓ | - | 包含视频路径的txt文件 |
| `--text_prompt` | ✓ | - | 目标检测的文本提示（如"car", "person"） |
| `--output_jsonl` | ✓ | - | 输出JSONL文件路径 |
| `--output_video_dir` | ✗ | None | 保存跟踪视频的目录（不指定则不保存视频） |
| `--sam_type` | ✗ | sam2.1_hiera_large | SAM模型类型 |
| `--model_path` | ✗ | models/sam2/checkpoints/sam2.1_hiera_large.pt | SAM模型权重路径 |
| `--detection_frequency` | ✗ | 1 | 检测频率（每N帧检测一次） |
| `--max_frames` | ✗ | 60 | 内存中最大保留帧数 |
| `--fps` | ✗ | None | 处理帧率（None=使用原始视频帧率） |
| `--num_workers` | ✗ | None | 并行进程数（None=使用GPU数量） |
| `--gpu_ids` | ✗ | None | 使用的GPU ID列表（如：--gpu_ids 0 1 2） |
| `--resume` | ✗ | True | 启用断点重续（默认开启） |
| `--no-resume` | ✗ | - | 禁用断点重续，重新处理所有视频 |

## 输出格式

### JSONL格式说明（多进程版本 - 按对象组织）⭐

每行是一个JSON对象，包含一个视频的完整跟踪结果。**数据按对象组织**，每个对象包含其完整的轨迹：

```json
{
  "video_path": "assets/car.mp4",
  "text_prompt": "car",
  "video_info": {
    "width": 1920,
    "height": 1080,
    "original_fps": 30.0,
    "process_fps": 5.0,
    "total_frames": 300
  },
  "objects": [
    {
      "obj_id": 0,
      "label": "car",
      "score": 0.95,
      "frames": [
        {
          "frame_idx": 0,
          "bbox": [100, 200, 150, 80]  // [x, y, width, height]
        },
        {
          "frame_idx": 1,
          "bbox": [105, 202, 150, 80]
        },
        // ... 更多帧
      ]
    },
    {
      "obj_id": 1,
      "label": "car",
      "score": 0.92,
      "frames": [
        {
          "frame_idx": 5,
          "bbox": [500, 300, 140, 75]
        },
        // ... 更多帧
      ]
    }
    // ... 更多对象
  ]
}
```

### 字段说明

- **video_path**: 视频文件路径
- **text_prompt**: 使用的文本提示
- **video_info**: 视频元信息
  - `width/height`: 视频分辨率
  - `original_fps`: 原始视频帧率
  - `process_fps`: 实际处理帧率
  - `total_frames`: 总帧数
- **objects**: 跟踪到的所有对象列表（按对象组织）
  - `obj_id`: 对象ID（唯一标识）
  - `label`: 对象类别标签
  - `score`: 检测置信度
  - `frames`: 该对象在所有帧中的轨迹
    - `frame_idx`: 帧索引
    - `bbox`: 边界框 [x, y, width, height]

### JSONL格式说明（单进程版本 - 按帧组织）

单进程版本按帧组织数据：

```json
{
  "video_path": "assets/car.mp4",
  "text_prompt": "car",
  "video_info": {
    "width": 1920,
    "height": 1080,
    "original_fps": 30.0,
    "process_fps": 5.0,
    "total_frames": 300
  },
  "frames": [
    {
      "frame_idx": 0,
      "objects": [
        {
          "obj_id": 0,
          "bbox": [100, 200, 150, 80],
          "label": "car",
          "score": 0.95
        }
      ]
    }
    // ... 更多帧
  ]
}
```

## 示例

### 示例1: 多进程跟踪（推荐）⭐

```bash
# 创建视频列表
cat > car_videos.txt << EOF
assets/car.mp4
/path/to/traffic_video1.mp4
/path/to/traffic_video2.mp4
EOF

# 使用所有可用GPU并行处理
python scripts/batch_track_videos_multiprocess.py \
    --video_list car_videos.txt \
    --text_prompt "car" \
    --output_jsonl car_tracking_results.jsonl
```

### 示例2: 指定GPU多进程处理

```bash
# 使用GPU 0, 1, 2进行并行处理
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "person" \
    --output_jsonl person_results.jsonl \
    --output_video_dir person_tracked_videos \
    --gpu_ids 0 1 2 \
    --num_workers 3 \
    --fps 10
```

### 示例3: 断点重续

```bash
# 首次运行（假设处理到一半时中断）
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl

# 重新运行，自动跳过已完成的视频，继续处理剩余视频
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl
```

### 示例4: 单进程跟踪（兼容旧版）

```bash
python scripts/batch_track_videos.py \
    --video_list video_list.txt \
    --text_prompt "dog" \
    --output_jsonl dog_results.jsonl \
    --detection_frequency 5 \
    --fps 5
```

## 读取JSONL结果

### Python示例（多进程版本 - 按对象组织）⭐

```python
import json

# 读取所有结果
results = []
with open('results.jsonl', 'r') as f:
    for line in f:
        result = json.loads(line)
        results.append(result)

# 处理第一个视频的结果
video_result = results[0]
print(f"Video: {video_result['video_path']}")
print(f"Total objects: {len(video_result['objects'])}")

# 遍历每个对象
for obj in video_result['objects']:
    obj_id = obj['obj_id']
    label = obj['label']
    score = obj['score']
    num_frames = len(obj['frames'])
    
    print(f"\nObject {obj_id}: {label} (score: {score:.2f})")
    print(f"  Tracked in {num_frames} frames")
    
    # 遍历该对象的所有帧
    for frame_data in obj['frames'][:5]:  # 只显示前5帧
        frame_idx = frame_data['frame_idx']
        bbox = frame_data['bbox']
        print(f"  Frame {frame_idx}: bbox={bbox}")
```

### Python示例（单进程版本 - 按帧组织）

```python
import json

# 读取所有结果
results = []
with open('results.jsonl', 'r') as f:
    for line in f:
        result = json.loads(line)
        results.append(result)

# 处理第一个视频的结果
video_result = results[0]
print(f"Video: {video_result['video_path']}")
print(f"Total frames: {len(video_result['frames'])}")

# 遍历每一帧
for frame in video_result['frames']:
    frame_idx = frame['frame_idx']
    objects = frame['objects']
    print(f"Frame {frame_idx}: {len(objects)} objects")
    
    for obj in objects:
        print(f"  Object {obj['obj_id']}: {obj['label']} @ {obj['bbox']}")
```

### 统计分析示例（多进程版本）

```python
import json
from collections import defaultdict

# 统计每个视频中的目标数量和轨迹长度
with open('results.jsonl', 'r') as f:
    for line in f:
        result = json.loads(line)
        video_path = result['video_path']
        objects = result['objects']
        
        print(f"\n{video_path}:")
        print(f"  Total objects tracked: {len(objects)}")
        
        # 统计每个对象的轨迹长度
        for obj in objects:
            obj_id = obj['obj_id']
            label = obj['label']
            trajectory_length = len(obj['frames'])
            
            # 计算平均位置
            avg_x = sum(f['bbox'][0] for f in obj['frames']) / trajectory_length
            avg_y = sum(f['bbox'][1] for f in obj['frames']) / trajectory_length
            
            print(f"  Object {obj_id} ({label}): {trajectory_length} frames, avg_pos=({avg_x:.1f}, {avg_y:.1f})")
```

### 轨迹可视化示例

```python
import json
import matplotlib.pyplot as plt

# 读取结果
with open('results.jsonl', 'r') as f:
    result = json.loads(f.readline())

# 绘制轨迹
plt.figure(figsize=(12, 8))
for obj in result['objects']:
    # 提取轨迹中心点
    trajectory = [(f['bbox'][0] + f['bbox'][2]/2, f['bbox'][1] + f['bbox'][3]/2) 
                  for f in obj['frames']]
    
    x_coords = [p[0] for p in trajectory]
    y_coords = [p[1] for p in trajectory]
    
    plt.plot(x_coords, y_coords, marker='o', label=f"Object {obj['obj_id']}")

plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title(f"Object Trajectories: {result['video_path']}")
plt.legend()
plt.gca().invert_yaxis()  # 反转Y轴（图像坐标系）
plt.grid(True)
plt.show()
```

## 性能优化建议

### 多进程版本优化
1. **使用多GPU**: 自动检测并使用所有可用GPU，或使用 `--gpu_ids 0 1 2` 指定
2. **断点重续**: 默认启用，中断后重新运行会自动跳过已完成的视频
3. **降低FPS**: 使用 `--fps 5` 减少处理帧数
4. **降低检测频率**: 使用 `--detection_frequency 3` 每3帧检测一次
5. **不保存视频**: 省略 `--output_video_dir` 只保存跟踪数据

### 单进程版本优化
1. **降低FPS**: 使用 `--fps 5` 减少处理帧数
2. **降低检测频率**: 使用 `--detection_frequency 3` 每3帧检测一次
3. **使用小模型**: 使用 `--sam_type sam2.1_hiera_tiny` 加快速度
4. **不保存视频**: 省略 `--output_video_dir` 只保存跟踪数据

## 注意事项

### 多进程版本
1. **GPU分配**: 每个进程独占一个GPU，确保有足够的GPU显存（建议至少8GB/GPU）
2. **断点重续**: 默认启用，通过检查JSONL文件中的video_path跳过已处理视频
3. **并行处理**: 视频会自动分配到不同GPU进行并行处理
4. **内存管理**: 每个进程独立管理内存，自动释放旧帧
5. **错误处理**: 如果某个视频处理失败，不影响其他视频的处理
6. **实时保存**: 每个视频处理完成后立即追加写入JSONL文件

### 单进程版本
1. **内存管理**: 脚本会自动释放旧帧，防止内存溢出
2. **模型加载**: 模型只在开始时加载一次，大幅节省时间
3. **错误处理**: 如果某个视频处理失败，会跳过并继续处理下一个
4. **实时保存**: 每个视频处理完成后立即写入JSONL文件
5. **GPU显存**: 确保GPU显存足够，建议至少8GB

## 故障排查

### 视频无法打开
- 检查视频路径是否正确
- 确认视频格式是否支持（推荐mp4）

### GPU内存不足
- 降低 `--max_frames` 参数
- 使用更小的模型（如 sam2.1_hiera_small）
- 降低处理FPS
- 多进程版本：减少 `--num_workers` 数量

### 检测结果为空
- 调整 `--text_prompt` 文本提示
- 降低置信度阈值（需修改代码中的 `score_threshold`）

### 断点重续不工作
- 检查JSONL文件是否存在且格式正确
- 确认视频路径在video_list.txt中与JSONL中完全一致
- 如需重新处理所有视频，使用 `--no-resume` 参数

### 多进程版本运行缓慢
- 检查GPU是否被充分利用（使用 `nvidia-smi` 监控）
- 确保 `--num_workers` 不超过GPU数量
- 考虑减少检测频率或降低FPS

## 最佳实践

### 推荐工作流程

1. **首次运行**: 使用多进程版本并启用断点重续
```bash
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl \
    --fps 5
```

2. **监控进度**: 使用 `nvidia-smi` 监控GPU使用情况
```bash
watch -n 1 nvidia-smi
```

3. **查看已完成视频数**:
```bash
wc -l results.jsonl
```

4. **中断后重续**: 直接重新运行相同命令即可
```bash
python scripts/batch_track_videos_multiprocess.py \
    --video_list video_list.txt \
    --text_prompt "car" \
    --output_jsonl results.jsonl \
    --fps 5
```

5. **分析结果**: 使用Python脚本分析JSONL文件

### 数据格式选择

- **按对象组织**（多进程版本）：适合轨迹分析、行为识别、运动分析
- **按帧组织**（单进程版本）：适合逐帧可视化、视频编辑、帧级标注

## 版本对比

| 特性 | 单进程版本 | 多进程版本 |
|------|-----------|-----------|
| 处理速度 | 慢 | 快（多GPU并行） |
| GPU利用 | 单GPU | 多GPU |
| 断点重续 | ✗ | ✓ |
| 数据格式 | 按帧组织 | 按对象组织 |
| 适用场景 | 小规模处理 | 大规模批量处理 |
| 推荐度 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
