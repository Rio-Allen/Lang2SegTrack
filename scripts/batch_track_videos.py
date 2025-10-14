import os
import json
import argparse
import time
from pathlib import Path
import torch
import gc
import cv2
import numpy as np
from PIL import Image
import imageio

from models.gdino.models.gdino import GDINO
from models.sam2.sam import SAM
from utils.color import COLOR
from utils.utils import batch_box_iou, filter_mask_outliers


class BatchVideoTracker:
    """批量视频跟踪处理器 - 只加载一次模型，处理多个视频"""
    
    def __init__(self, sam_type="sam2.1_hiera_large", 
                 model_path="models/sam2/checkpoints/sam2.1_hiera_large.pt",
                 device="cuda:0",
                 detection_frequency=1,
                 max_frames=60,
                 fps=None):
        """
        初始化批量跟踪器
        
        参数:
            sam_type: SAM模型类型
            model_path: SAM模型权重路径
            device: GPU设备
            detection_frequency: 检测频率（每N帧检测一次）
            max_frames: 最大保留帧数（防止内存溢出）
            fps: 处理视频的帧率（None表示使用原始视频帧率）
        """
        self.device = device
        self.detection_frequency = detection_frequency
        self.max_frames = max_frames
        self.fps = fps
        
        # 跟踪参数
        self.iou_threshold = 0.3
        self.box_threshold = 0.5
        self.text_threshold = 0.8
        self.score_threshold = 0.3
        
        print("=" * 60)
        print("Initializing models (this will be done only once)...")
        print("=" * 60)
        
        # 初始化SAM模型（只加载一次）
        print(f"Loading SAM model: {sam_type}")
        self.sam = SAM()
        self.sam.build_model(sam_type, model_path, predictor_type="video", 
                            device=device, use_txt_prompt=True)
        print("✓ SAM model loaded successfully")
        
        # 初始化GroundingDINO模型（只加载一次）
        print("Loading GroundingDINO model...")
        self.gdino = GDINO()
        self.gdino.build_model(device=device)
        print("✓ GroundingDINO model loaded successfully")
        
        print("=" * 60)
        print("Models initialization complete!")
        print("=" * 60)
    
    def detect_objects(self, frame, text_prompt):
        """使用GroundingDINO检测目标"""
        detection = self.gdino.predict(
            [Image.fromarray(frame)],
            [text_prompt],
            self.box_threshold, 
            self.text_threshold
        )[0]
        
        scores = detection['scores'].cpu().numpy()
        labels = detection['labels']
        boxes = detection['boxes'].cpu().numpy().astype(np.int32)
        
        # 过滤低置信度检测
        filter_mask = scores > self.score_threshold
        valid_boxes = boxes[filter_mask]
        valid_labels = labels[filter_mask]
        valid_scores = scores[filter_mask]
        
        return valid_boxes, valid_labels, valid_scores
    
    def track_single_video(self, video_path, text_prompt, output_video_path=None):
        """
        跟踪单个视频
        
        返回:
            dict: 包含跟踪结果的字典
        """
        print(f"\nProcessing video: {video_path}")
        print(f"Text prompt: {text_prompt}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return None
        
        # 获取视频信息
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算处理帧率
        if self.fps is None:
            process_fps = original_fps
            frame_interval = 1
        else:
            process_fps = self.fps
            frame_interval = max(1, round(original_fps / self.fps))
        
        print(f"Video info: {width}x{height}, {original_fps:.2f} fps, {total_frames} frames")
        print(f"Processing: {process_fps:.2f} fps (interval: {frame_interval})")
        
        # 读取第一帧
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Cannot read first frame")
            cap.release()
            return None
        
        # 初始化视频写入器（如果需要保存视频）
        writer = None
        if output_video_path:
            writer = imageio.get_writer(output_video_path, fps=process_fps)
        
        # 初始化跟踪结果
        results = {
            'video_path': video_path,
            'text_prompt': text_prompt,
            'video_info': {
                'width': width,
                'height': height,
                'original_fps': original_fps,
                'process_fps': process_fps,
                'total_frames': total_frames
            },
            'frames': []
        }
        
        # 跟踪状态
        prompts = {'prompts': [], 'labels': [], 'scores': []}
        existing_obj_outputs = []
        last_text_prompt = None
        
        predictor = self.sam.video_predictor
        
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            # 初始化跟踪状态
            state = predictor.init_state_from_numpy_frames(
                [first_frame], 
                offload_state_to_cpu=False, 
                offload_video_to_cpu=False
            )
            
            frame_count = 0
            video_frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                video_frame_count += 1
                
                # 帧率控制：跳帧处理
                if frame_interval > 1 and video_frame_count % frame_interval != 0:
                    continue
                
                frame_count += 1
                
                # 检测新对象
                should_detect = (
                    (state['num_frames'] - 1) % self.detection_frequency == 0 or 
                    last_text_prompt is None or
                    last_text_prompt != text_prompt
                )
                
                if should_detect:
                    valid_boxes, valid_labels, valid_scores = self.detect_objects(frame, text_prompt)
                    
                    # 文本提示改变时，直接添加所有检测结果
                    if last_text_prompt != text_prompt:
                        if len(valid_boxes) > 0:
                            prompts['prompts'].extend(valid_boxes.tolist())
                            prompts['labels'].extend(valid_labels)
                            prompts['scores'].extend(valid_scores.tolist())
                            add_new = True
                    
                    # 已有跟踪对象时，只添加新对象
                    elif existing_obj_outputs and len(valid_boxes) > 0:
                        iou_matrix = batch_box_iou(valid_boxes, np.array(existing_obj_outputs))
                        is_new = np.max(iou_matrix, axis=1) < self.iou_threshold
                        
                        new_boxes = valid_boxes[is_new]
                        new_labels = valid_labels[is_new]
                        new_scores = valid_scores[is_new]
                        
                        if len(new_boxes) > 0:
                            prompts['prompts'].extend(new_boxes.tolist())
                            prompts['labels'].extend(new_labels)
                            prompts['scores'].extend(new_scores.tolist())
                            add_new = True
                    
                    # 首次检测
                    elif len(valid_boxes) > 0 and not existing_obj_outputs:
                        prompts['prompts'].extend(valid_boxes.tolist())
                        prompts['labels'].extend(valid_labels)
                        prompts['scores'].extend(valid_scores.tolist())
                        add_new = True
                    else:
                        add_new = False
                    
                    last_text_prompt = text_prompt
                else:
                    add_new = False
                
                # 添加新对象到跟踪状态
                if add_new:
                    predictor.reset_state(state)
                    frame_idx = state["num_frames"] - 1
                    
                    for id, bbox in enumerate(prompts['prompts']):
                        predictor.add_new_points_or_box(state, box=bbox, 
                                                       frame_idx=frame_idx, obj_id=id)
                
                # 添加帧到推理状态
                predictor.append_frame_to_inference_state(state, frame)
                
                # 执行跟踪
                frame_result = {
                    'frame_idx': frame_count - 1,
                    'objects': []
                }
                
                if (any(len(state["point_inputs_per_obj"][i]) > 0 for i in range(len(state["point_inputs_per_obj"]))) or
                    any(len(state["mask_inputs_per_obj"][i]) > 0 for i in range(len(state["mask_inputs_per_obj"])))):
                    
                    for frame_idx, obj_ids, masks in predictor.propagate_in_frame(state, state["num_frames"] - 1):
                        existing_obj_outputs = []
                        
                        for obj_id, mask in zip(obj_ids, masks):
                            mask = mask[0].cpu().numpy() > 0.0
                            mask = filter_mask_outliers(mask)
                            nonzero = np.argwhere(mask)
                            
                            if nonzero.size == 0:
                                bbox = [0, 0, 0, 0]
                            else:
                                y_min, x_min = nonzero.min(axis=0)
                                y_max, x_max = nonzero.max(axis=0)
                                bbox = [int(x_min), int(y_min), 
                                       int(x_max - x_min), int(y_max - y_min)]
                            
                            existing_obj_outputs.append([bbox[0], bbox[1], 
                                                        bbox[0] + bbox[2], bbox[1] + bbox[3]])
                            
                            # 记录对象信息
                            frame_result['objects'].append({
                                'obj_id': int(obj_id),
                                'bbox': bbox,  # [x, y, w, h]
                                'label': prompts['labels'][obj_id] if obj_id < len(prompts['labels']) else None,
                                'score': float(prompts['scores'][obj_id]) if obj_id < len(prompts['scores']) else None
                            })
                            
                            # 可视化（如果需要保存视频）
                            if writer:
                                mask_img = np.zeros((height, width, 3), dtype=np.uint8)
                                mask_img[mask] = COLOR[obj_id % len(COLOR)]
                                frame = cv2.addWeighted(frame, 1, mask_img, 0.6, 0)
                                x, y, w, h = bbox
                                cv2.rectangle(frame, (x, y), (x + w, y + h), 
                                            COLOR[obj_id % len(COLOR)], 2)
                        
                        prompts['prompts'] = existing_obj_outputs.copy()
                
                results['frames'].append(frame_result)
                
                # 保存帧到视频
                if writer:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(rgb)
                
                # 内存管理
                if state["num_frames"] % self.max_frames == 0:
                    if len(state["output_dict"]["non_cond_frame_outputs"]) != 0:
                        predictor.append_frame_as_cond_frame(state, state["num_frames"] - 2)
                    predictor.release_old_frames(state)
                
                # 进度显示
                if frame_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"  Processed {frame_count} frames ({fps:.2f} fps)", end='\r')
        
        # 清理资源
        cap.release()
        if writer:
            writer.close()
        
        # 清理状态
        del predictor, state
        gc.collect()
        torch.cuda.empty_cache()
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed: {frame_count} frames in {elapsed:.2f}s ({frame_count/elapsed:.2f} fps)")
        print(f"  Total objects tracked: {len(set(obj['obj_id'] for f in results['frames'] for obj in f['objects']))}")
        
        return results
    
    def process_video_list(self, video_list_path, text_prompt, output_jsonl_path, 
                          output_video_dir=None):
        """
        批量处理视频列表
        
        参数:
            video_list_path: 包含视频路径的txt文件
            text_prompt: 文本提示（用于所有视频）
            output_jsonl_path: 输出jsonl文件路径
            output_video_dir: 输出视频目录（可选）
        """
        # 读取视频列表
        with open(video_list_path, 'r') as f:
            video_paths = [line.strip() for line in f if line.strip()]
        
        print(f"\nFound {len(video_paths)} videos to process")
        
        # 创建输出目录
        if output_video_dir:
            os.makedirs(output_video_dir, exist_ok=True)
        
        # 打开jsonl文件
        with open(output_jsonl_path, 'w') as jsonl_file:
            for idx, video_path in enumerate(video_paths, 1):
                print(f"\n{'='*60}")
                print(f"Processing video {idx}/{len(video_paths)}")
                print(f"{'='*60}")
                
                # 检查视频文件是否存在
                if not os.path.exists(video_path):
                    print(f"Warning: Video not found: {video_path}")
                    continue
                
                # 生成输出视频路径
                output_video_path = None
                if output_video_dir:
                    video_name = Path(video_path).stem
                    output_video_path = os.path.join(output_video_dir, 
                                                     f"{video_name}_tracked.mp4")
                
                # 处理视频
                result = self.track_single_video(video_path, text_prompt, output_video_path)
                
                if result:
                    # 写入jsonl
                    jsonl_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                    jsonl_file.flush()  # 立即写入磁盘
                    
                    if output_video_path:
                        print(f"✓ Saved tracked video to: {output_video_path}")
        
        print(f"\n{'='*60}")
        print("All videos processed!")
        print(f"Results saved to: {output_jsonl_path}")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Batch video tracking')
    parser.add_argument('--video_list', type=str, required=True,
                       help='Path to txt file containing video paths (one per line)')
    parser.add_argument('--text_prompt', type=str, required=True,
                       help='Text prompt for object detection')
    parser.add_argument('--output_jsonl', type=str, required=True,
                       help='Output JSONL file path')
    parser.add_argument('--output_video_dir', type=str, default=None,
                       help='Directory to save tracked videos (optional)')
    parser.add_argument('--sam_type', type=str, default='sam2.1_hiera_large',
                       help='SAM model type')
    parser.add_argument('--model_path', type=str, 
                       default='models/sam2/checkpoints/sam2.1_hiera_large.pt',
                       help='SAM model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run on')
    parser.add_argument('--detection_frequency', type=int, default=1,
                       help='Detection frequency (detect every N frames)')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Max frames to keep in memory')
    parser.add_argument('--fps', type=float, default=None,
                       help='Processing FPS (None = use original video fps)')
    
    args = parser.parse_args()
    
    # 创建批量跟踪器
    tracker = BatchVideoTracker(
        sam_type=args.sam_type,
        model_path=args.model_path,
        device=args.device,
        detection_frequency=args.detection_frequency,
        max_frames=args.max_frames,
        fps=args.fps
    )
    
    # 处理视频列表
    tracker.process_video_list(
        video_list_path=args.video_list,
        text_prompt=args.text_prompt,
        output_jsonl_path=args.output_jsonl,
        output_video_dir=args.output_video_dir
    )


if __name__ == "__main__":
    main()
