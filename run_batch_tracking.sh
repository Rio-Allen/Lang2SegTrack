#!/bin/bash

# 批量视频跟踪测试脚本
# 使用示例视频进行快速测试

echo "=========================================="
echo "批量视频跟踪测试"
echo "=========================================="

# 创建输出目录
mkdir -p batch_output
mkdir -p batch_output/tracked_videos

# 使用示例视频列表
VIDEO_LIST="video_list_example.txt"
TEXT_PROMPT="car"
OUTPUT_JSONL="batch_output/tracking_results.jsonl"
OUTPUT_VIDEO_DIR="batch_output/tracked_videos"

echo ""
echo "配置:"
echo "  视频列表: $VIDEO_LIST"
echo "  文本提示: $TEXT_PROMPT"
echo "  输出JSONL: $OUTPUT_JSONL"
echo "  输出视频目录: $OUTPUT_VIDEO_DIR"
echo ""

# 运行批量跟踪
python scripts/batch_track_videos.py \
    --video_list "$VIDEO_LIST" \
    --text_prompt "$TEXT_PROMPT" \
    --output_jsonl "$OUTPUT_JSONL" \
    --output_video_dir "$OUTPUT_VIDEO_DIR" \
    --sam_type sam2.1_hiera_large \
    --model_path models/sam2/checkpoints/sam2.1_hiera_large.pt \
    --device cuda:0 \
    --detection_frequency 1 \
    --fps 5

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "查看结果:"
echo "  JSONL文件: $OUTPUT_JSONL"
echo "  跟踪视频: $OUTPUT_VIDEO_DIR/"
echo ""
echo "查看JSONL内容:"
echo "  cat $OUTPUT_JSONL | python -m json.tool"
echo ""
