#!/bin/bash

# 多进程批量视频跟踪脚本
# 使用多个GPU并行处理视频
# 特性：
#   - 多GPU并行处理
#   - 断点重续（默认启用）
#   - 按对象组织输出格式

echo "=========================================="
echo "多进程批量视频跟踪"
echo "=========================================="

# 创建输出目录
mkdir -p output
mkdir -p output/tracked_videos

# 配置参数
VIDEO_LIST="videos.txt"
TEXT_PROMPT="person"
OUTPUT_JSONL="output/tracking.jsonl"
OUTPUT_VIDEO_DIR="output/tracked_videos"

# GPU配置
# 方式1: 自动使用所有可用GPU
GPU_IDS=""

# 方式2: 手动指定GPU (例如使用GPU 0和1)
# GPU_IDS="--gpu_ids 0 1"

# 工作进程数 (None表示使用GPU数量)
NUM_WORKERS=""
# NUM_WORKERS="--num_workers 4"

# 断点重续设置
# 方式1: 启用断点重续（默认）- 跳过已处理的视频
RESUME=""

# 方式2: 禁用断点重续 - 重新处理所有视频
# RESUME="--no-resume"

echo ""
echo "配置:"
echo "  视频列表: $VIDEO_LIST"
echo "  文本提示: $TEXT_PROMPT"
echo "  输出JSONL: $OUTPUT_JSONL"
echo "  输出视频目录: $OUTPUT_VIDEO_DIR"
echo "  GPU设置: ${GPU_IDS:-所有可用GPU}"
echo "  工作进程数: ${NUM_WORKERS:-与GPU数量相同}"
echo "  断点重续: ${RESUME:-启用（默认）}"
echo ""

# 运行多进程批量跟踪
python scripts/batch_track_videos_multiprocess.py \
    --video_list "$VIDEO_LIST" \
    --text_prompt "$TEXT_PROMPT" \
    --output_jsonl "$OUTPUT_JSONL" \
    --output_video_dir "$OUTPUT_VIDEO_DIR" \
    --sam_type sam2.1_hiera_large \
    --model_path models/sam2/checkpoints/sam2.1_hiera_large.pt \
    --detection_frequency 1 \
    --fps 5 \
    $GPU_IDS \
    $NUM_WORKERS \
    $RESUME

echo ""
echo "=========================================="
echo "处理完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  JSONL文件: $OUTPUT_JSONL"
echo "  跟踪视频: $OUTPUT_VIDEO_DIR/"
echo ""
echo "统计信息:"
echo "  已处理视频数: $(wc -l < $OUTPUT_JSONL 2>/dev/null || echo 0)"
echo ""
echo "查看JSONL内容:"
echo "  cat $OUTPUT_JSONL | python -m json.tool"
echo ""
echo "提示:"
echo "  - 数据格式: 按对象组织（每个对象包含完整轨迹）"
echo "  - 断点重续: 中断后重新运行此脚本会自动跳过已完成的视频"
echo "  - 重新处理: 取消注释 'RESUME=\"--no-resume\"' 来重新处理所有视频"
echo ""
