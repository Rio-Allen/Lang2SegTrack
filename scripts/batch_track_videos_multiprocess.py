import os
import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
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


class VideoTracker:
    """单视频跟踪处理器 - 每个进程独立使用一个GPU"""
    
    def __init__(self, sam_type="sam2.1_hiera_large", 
                 model_path="models/sam2/checkpoints/sam2.1_hiera_large.pt",
                 device="cuda:0",
                 detection_frequency=1,
                 max_frames=60,
                 fps=None):
        """
        初始化视频跟踪器
        
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
        
        # 初始化SAM模型
        self.sam = SAM()
        self.sam.build_model(sam_type, model_path, predictor_type="video", 
                            device=device, use_txt_prompt=True)
        
        # 初始化GroundingDINO模型
        self.gdino = GDINO()
        self.gdino.build_model(device=device)
    
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
    
    def track_video(self, video_path, text_prompt, output_video_path=None):
        """
        跟踪单个视频
        
        返回:
            dict: 包含跟踪结果的字典
        """
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
        
        # 初始化跟踪结果 - 按对象组织
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
            'objects': {}  # 改为字典，key为object_id
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
                            
                            # 按对象组织数据
                            obj_id_str = str(obj_id)
                            if obj_id_str not in results['objects']:
                                results['objects'][obj_id_str] = {
                                    'obj_id': int(obj_id),
                                    'label': prompts['labels'][obj_id] if obj_id < len(prompts['labels']) else None,
                                    'score': float(prompts['scores'][obj_id]) if obj_id < len(prompts['scores']) else None,
                                    'frames': []
                                }
                            
                            # 添加帧数据到对象轨迹
                            results['objects'][obj_id_str]['frames'].append({
                                'frame_idx': frame_count - 1,
                                'bbox': bbox  # [x, y, w, h]
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
                
                # 保存帧到视频
                if writer:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(rgb)
                
                # 内存管理
                if state["num_frames"] % self.max_frames == 0:
                    if len(state["output_dict"]["non_cond_frame_outputs"]) != 0:
                        predictor.append_frame_as_cond_frame(state, state["num_frames"] - 2)
                    predictor.release_old_frames(state)
        
        # 清理资源
        cap.release()
        if writer:
            writer.close()
        
        # 清理状态
        del predictor, state
        gc.collect()
        torch.cuda.empty_cache()
        
        elapsed = time.time() - start_time
        num_objects = len(results['objects'])
        
        # 将objects字典转换为列表（更符合JSON习惯）
        results['objects'] = list(results['objects'].values())
        
        return results, frame_count, elapsed, num_objects


def process_single_video(args_tuple):
    """
    处理单个视频的工作函数（用于多进程）
    
    参数:
        args_tuple: (video_path, text_prompt, output_video_path, config)
    
    返回:
        tuple: (result, video_path, success)
    """
    video_path, text_prompt, output_video_path, config = args_tuple
    
    # 获取当前进程的GPU设备
    gpu_id = config['gpu_id']
    device = f"cuda:{gpu_id}"
    
    print(f"[GPU {gpu_id}] Processing: {video_path}")
    
    try:
        # 创建跟踪器（每个进程独立创建，使用指定的GPU）
        tracker = VideoTracker(
            sam_type=config['sam_type'],
            model_path=config['model_path'],
            device=device,
            detection_frequency=config['detection_frequency'],
            max_frames=config['max_frames'],
            fps=config['fps']
        )
        
        # 执行跟踪
        result, frame_count, elapsed, num_objects = tracker.track_video(
            video_path, text_prompt, output_video_path
        )
        
        if result:
            print(f"[GPU {gpu_id}] ✓ Completed: {video_path}")
            print(f"[GPU {gpu_id}]   - {frame_count} frames in {elapsed:.2f}s ({frame_count/elapsed:.2f} fps)")
            print(f"[GPU {gpu_id}]   - {num_objects} objects tracked")
            return (result, video_path, True)
        else:
            print(f"[GPU {gpu_id}] ✗ Failed: {video_path}")
            return (None, video_path, False)
    
    except Exception as e:
        print(f"[GPU {gpu_id}] ✗ Error processing {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return (None, video_path, False)
    
    finally:
        # 清理GPU内存
        gc.collect()
        torch.cuda.empty_cache()


def get_available_gpus():
    """获取可用的GPU列表"""
    if not torch.cuda.is_available():
        return []
    
    num_gpus = torch.cuda.device_count()
    return list(range(num_gpus))


def distribute_videos_to_gpus(video_paths, gpu_ids):
    """
    将视频分配到不同的GPU
    
    返回:
        list: [(video_path, gpu_id), ...]
    """
    if not gpu_ids:
        raise ValueError("No GPUs available")
    
    video_gpu_pairs = []
    for idx, video_path in enumerate(video_paths):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        video_gpu_pairs.append((video_path, gpu_id))
    
    return video_gpu_pairs


def load_completed_videos(output_jsonl_path):
    """
    从JSONL文件中加载已完成处理的视频路径
    
    参数:
        output_jsonl_path: JSONL文件路径
    
    返回:
        set: 已完成处理的视频路径集合
    """
    completed_videos = set()
    
    if not os.path.exists(output_jsonl_path):
        return completed_videos
    
    try:
        with open(output_jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        result = json.loads(line)
                        if 'video_path' in result:
                            completed_videos.add(result['video_path'])
                    except json.JSONDecodeError as e:
                        print(f"Warning: Failed to parse line in {output_jsonl_path}: {e}")
                        continue
        
        if completed_videos:
            print(f"Found {len(completed_videos)} already processed videos in {output_jsonl_path}")
    
    except Exception as e:
        print(f"Warning: Error reading {output_jsonl_path}: {e}")
    
    return completed_videos


def process_video_list_multiprocess(video_list_path, text_prompt, output_jsonl_path,
                                    output_video_dir=None, sam_type="sam2.1_hiera_large",
                                    model_path="models/sam2/checkpoints/sam2.1_hiera_large.pt",
                                    detection_frequency=1, max_frames=60, fps=None,
                                    num_workers=None, gpu_ids=None, resume=True):
    """
    使用多进程批量处理视频列表
    
    参数:
        video_list_path: 包含视频路径的txt文件
        text_prompt: 文本提示（用于所有视频）
        output_jsonl_path: 输出jsonl文件路径
        output_video_dir: 输出视频目录（可选）
        sam_type: SAM模型类型
        model_path: SAM模型权重路径
        detection_frequency: 检测频率
        max_frames: 最大保留帧数
        fps: 处理帧率
        num_workers: 并行进程数（None表示使用GPU数量）
        gpu_ids: 使用的GPU ID列表（None表示使用所有可用GPU）
        resume: 是否启用断点重续（默认True）
    """
    # 读取视频列表
    with open(video_list_path, 'r') as f:
        video_paths = [line.strip() for line in f if line.strip()]
    
    # 检查视频文件
    valid_video_paths = []
    for video_path in video_paths:
        if os.path.exists(video_path):
            valid_video_paths.append(video_path)
        else:
            print(f"Warning: Video not found: {video_path}")
    
    video_paths = valid_video_paths
    
    # 断点重续：加载已完成的视频
    completed_videos = set()
    if resume:
        completed_videos = load_completed_videos(output_jsonl_path)
        
        # 过滤掉已完成的视频
        original_count = len(video_paths)
        video_paths = [vp for vp in video_paths if vp not in completed_videos]
        
        if original_count > len(video_paths):
            print(f"Resume mode: Skipping {original_count - len(video_paths)} already processed videos")
            print(f"Remaining videos to process: {len(video_paths)}")
    
    video_paths = valid_video_paths
    
    if not video_paths:
        if resume and completed_videos:
            print("All videos have been processed already!")
        else:
            print("No valid videos to process")
        return
    
    # 获取可用GPU
    if gpu_ids is None:
        gpu_ids = get_available_gpus()
    
    if not gpu_ids:
        raise ValueError("No GPUs available. This script requires CUDA-enabled GPUs.")
    
    # 设置工作进程数
    if num_workers is None:
        num_workers = len(gpu_ids)
    
    print(f"\n{'='*60}")
    print(f"Multi-Process Video Tracking")
    print(f"{'='*60}")
    print(f"Total videos in list: {len(video_paths) + len(completed_videos)}")
    if resume and completed_videos:
        print(f"Already completed: {len(completed_videos)}")
        print(f"Remaining to process: {len(video_paths)}")
    else:
        print(f"Videos to process: {len(video_paths)}")
    print(f"Available GPUs: {gpu_ids}")
    print(f"Number of workers: {num_workers}")
    print(f"Text prompt: {text_prompt}")
    print(f"Resume mode: {'Enabled' if resume else 'Disabled'}")
    print(f"{'='*60}\n")
    
    # 创建输出目录
    if output_video_dir:
        os.makedirs(output_video_dir, exist_ok=True)
    
    # 分配视频到GPU
    video_gpu_pairs = distribute_videos_to_gpus(video_paths, gpu_ids)
    
    # 准备任务参数
    tasks = []
    for video_path, gpu_id in video_gpu_pairs:
        # 生成输出视频路径
        output_video_path = None
        if output_video_dir:
            video_name = Path(video_path).stem
            output_video_path = os.path.join(output_video_dir, 
                                             f"{video_name}_tracked.mp4")
        
        # 配置参数
        config = {
            'sam_type': sam_type,
            'model_path': model_path,
            'detection_frequency': detection_frequency,
            'max_frames': max_frames,
            'fps': fps,
            'gpu_id': gpu_id
        }
        
        tasks.append((video_path, text_prompt, output_video_path, config))
    
    # 使用进程池执行任务
    start_time = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = {executor.submit(process_single_video, task): task for task in tasks}
        
        # 收集结果
        completed = 0
        for future in as_completed(futures):
            result, video_path, success = future.result()
            completed += 1
            
            if success and result:
                results.append(result)
                print(f"\nProgress: {completed}/{len(video_paths)} videos completed")
            else:
                print(f"\nProgress: {completed}/{len(video_paths)} videos completed (last one failed)")
    
    # 保存结果到JSONL（追加模式）
    with open(output_jsonl_path, 'a') as jsonl_file:
        for result in results:
            jsonl_file.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    elapsed = time.time() - start_time
    total_videos = len(video_paths) + len(completed_videos)
    
    print(f"\n{'='*60}")
    print(f"All videos processed!")
    print(f"{'='*60}")
    print(f"Total time: {elapsed:.2f}s")
    if len(video_paths) > 0:
        print(f"Average time per video: {elapsed/len(video_paths):.2f}s")
    print(f"Successfully processed this run: {len(results)}/{len(video_paths)}")
    if resume and completed_videos:
        print(f"Total completed (including previous runs): {len(results) + len(completed_videos)}/{total_videos}")
    print(f"Results saved to: {output_jsonl_path}")
    if output_video_dir:
        print(f"Videos saved to: {output_video_dir}/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Batch video tracking with multi-process')
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
    parser.add_argument('--detection_frequency', type=int, default=1,
                       help='Detection frequency (detect every N frames)')
    parser.add_argument('--max_frames', type=int, default=60,
                       help='Max frames to keep in memory')
    parser.add_argument('--fps', type=float, default=None,
                       help='Processing FPS (None = use original video fps)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of parallel workers (None = use number of GPUs)')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None,
                       help='GPU IDs to use (e.g., --gpu_ids 0 1 2). None = use all GPUs')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='Enable resume mode (skip already processed videos, default: True)')
    parser.add_argument('--no-resume', dest='resume', action='store_false',
                       help='Disable resume mode (reprocess all videos)')
    
    args = parser.parse_args()
    
    # 处理视频列表
    process_video_list_multiprocess(
        video_list_path=args.video_list,
        text_prompt=args.text_prompt,
        output_jsonl_path=args.output_jsonl,
        output_video_dir=args.output_video_dir,
        sam_type=args.sam_type,
        model_path=args.model_path,
        detection_frequency=args.detection_frequency,
        max_frames=args.max_frames,
        fps=args.fps,
        num_workers=args.num_workers,
        gpu_ids=args.gpu_ids,
        resume=args.resume
    )


if __name__ == "__main__":
    main()
