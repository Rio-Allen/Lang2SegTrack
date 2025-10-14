import base64
import os
import shutil
import threading
import queue
import time
from io import BytesIO

# Set OpenCV to headless mode before importing cv2
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import cv2
import torch
import gc
import numpy as np
import imageio
from PIL import Image

from models.gdino.models.gdino import GDINO
from models.sam2.sam import SAM
from utils.color import COLOR
import pyrealsense2 as rs
from utils.utils import batch_box_iou, filter_mask_outliers

class Lang2SegTrack:
    def __init__(self, sam_type:str="sam2.1_hiera_tiny", model_path:str="models/sam2/checkpoints/sam2.1_hiera_large.pt",
                 video_path:str="", output_path:str="", use_txt_prompt:bool=False, max_frames:int=60,
                 first_prompts: list | None = None, save_video=True, device="cuda:0", mode="realtime", 
                 headless=False, detection_frequency:int=1, fps:int|None=None):
        """
        初始化 Lang2SegTrack 跟踪器（高频检测版本，无后向跟踪）
        
        参数:
            sam_type: SAM模型类型
            model_path: SAM模型权重路径
            video_path: 视频文件路径（video模式必需）
            output_path: 输出视频保存路径
            use_txt_prompt: 是否启用文本提示
            max_frames: 最大保留帧数（防止内存溢出）
            first_prompts: 初始提示（bbox/point/mask）
            save_video: 是否保存输出视频
            device: GPU设备
            mode: 运行模式 "video" 或 "realtime"
            headless: 是否无头模式（无GUI显示）
            detection_frequency: 检测频率（每N帧检测一次，默认1表示每帧检测）
            fps: 处理视频时的帧率以及保存输出结果的帧率（None表示使用原始视频帧率，不进行跳帧）
        """
        self.sam_type = sam_type
        self.model_path = model_path
        self.video_path = video_path
        self.output_path = output_path
        self.max_frames = max_frames
        self.first_prompts = first_prompts
        self.save_video = save_video
        self.device = device
        self.mode = mode
        self.headless = headless
        self.detection_frequency = detection_frequency  # 高频检测，默认每帧检测
        self.fps = fps  # 帧率（None表示使用原始视频帧率）
        
        if self.mode == 'img' and not use_txt_prompt:
            raise ValueError("In 'img' mode, use_txt_prompt must be True")

        # 初始化SAM模型
        self.sam = SAM()
        self.sam.build_model(self.sam_type, self.model_path, predictor_type=mode, device=device, use_txt_prompt=use_txt_prompt)
        
        # 初始化GroundingDINO模型（如果使用文本提示）
        if use_txt_prompt:
            self.gdino = GDINO()
            self.gdino_16 = False
            if not self.gdino_16:
                print("Building GroundingDINO model...")
                self.gdino.build_model(device=device)
        else:
            self.gdino = None

        # 数据管理（移除后向跟踪相关的数据结构）
        self.existing_obj_outputs = []  # 当前存在的目标输出
        self.current_text_prompt = None
        self.last_text_prompt = None
        
        # 初始化提示
        if self.first_prompts is not None:
            self.prompts = {
                'prompts': self.first_prompts, 
                'labels': [None] * len(self.first_prompts), 
                'scores': [None] * len(self.first_prompts)
            }
            self.add_new = True
        else:
            self.prompts = {'prompts': [], 'labels': [], 'scores': []}
        
        # 参数配置
        self.iou_threshold = 0.3  # IoU阈值，用于判断是否为新目标
        self.box_threshold = 0.5   # 检测框置信度阈值
        self.text_threshold = 0.8 # 文本匹配置信度阈值
        self.score_threshold = 0.3 # 最终过滤阈值

        # 交互相关
        self.input_queue = queue.Queue()
        self.drawing = False
        self.add_new = False
        self.ix, self.iy = -1, -1
        self.frame_display = None
        self.height, self.width = None, None
        self.prev_time = 0
        
        # 统计信息
        self.frame_count = 0
        self.total_detections = 0

    def input_thread(self):
        """后台线程，监听用户文本输入"""
        while True:
            user_input = input()
            self.input_queue.put(user_input)

    def draw_bbox(self, event, x, y, flags, param):
        """鼠标回调函数，用于手动绘制bbox或点击点"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                # Ctrl+左键：添加点提示
                self.prompts['prompts'].append((x, y))
                self.prompts['labels'].append(None)
                self.prompts['scores'].append(None)
                self.add_new = True
                cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
            else:
                # 普通左键：开始绘制矩形
                self.drawing = True
                self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            # 鼠标移动：实时显示矩形
            img = param.copy()
            cv2.rectangle(img, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Video Tracking", img)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            # 松开左键：完成矩形绘制
            if abs(x - self.ix) > 2 and abs(y - self.iy) > 2:
                bbox = [self.ix, self.iy, x, y]
                self.prompts['prompts'].append(bbox)
                self.prompts['labels'].append(None)
                self.prompts['scores'].append(None)
                self.add_new = True
                cv2.rectangle(param, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
            self.drawing = False

    def add_to_state(self, predictor, state, prompts):
        """将提示添加到SAM跟踪状态"""
        frame_idx = state["num_frames"] - 1
        for id, item in enumerate(prompts['prompts']):
            if len(item) == 4:
                # Bounding box提示
                x1, y1, x2, y2 = item
                cv2.rectangle(self.frame_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                predictor.add_new_points_or_box(state, box=item, frame_idx=frame_idx, obj_id=id)
            elif len(item) == 2:
                # Point提示
                x, y = item
                cv2.circle(self.frame_display, (x, y), 5, (0, 255, 0), -1)
                pt = torch.tensor([[x, y]], dtype=torch.float32)
                lbl = torch.tensor([1], dtype=torch.int32)
                predictor.add_new_points_or_box(state, points=pt, labels=lbl, frame_idx=frame_idx, obj_id=id)
            else:
                # Mask提示
                predictor.add_new_mask(state, mask=item, frame_idx=frame_idx, obj_id=id)

    def track_and_visualize(self, predictor, state, frame, writer):
        """执行跟踪并可视化结果"""
        if (any(len(state["point_inputs_per_obj"][i]) > 0 for i in range(len(state["point_inputs_per_obj"]))) or
            any(len(state["mask_inputs_per_obj"][i]) > 0 for i in range(len(state["mask_inputs_per_obj"])))):
            
            for frame_idx, obj_ids, masks in predictor.propagate_in_frame(state, state["num_frames"] - 1):
                self.existing_obj_outputs = []
                
                for obj_id, mask in zip(obj_ids, masks):
                    mask = mask[0].cpu().numpy() > 0.0
                    mask = filter_mask_outliers(mask)
                    nonzero = np.argwhere(mask)
                    
                    if nonzero.size == 0:
                        bbox = [0, 0, 0, 0]
                    else:
                        y_min, x_min = nonzero.min(axis=0)
                        y_max, x_max = nonzero.max(axis=0)
                        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                    
                    self.draw_mask_and_bbox(frame, mask, bbox, obj_id)
                    self.existing_obj_outputs.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                
                # 更新prompts为当前跟踪的目标
                self.prompts['prompts'] = self.existing_obj_outputs.copy()

        # 显示FPS
        frame_dis = self.show_fps(frame)
        if not self.headless:
            cv2.imshow("Video Tracking", frame_dis)

        # 保存到视频
        if writer:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(rgb)

    def draw_mask_and_bbox(self, frame, mask, bbox, obj_id):
        """在帧上绘制掩码和边界框"""
        mask_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        mask_img[mask] = COLOR[obj_id % len(COLOR)]
        frame[:] = cv2.addWeighted(frame, 1, mask_img, 0.6, 0)
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR[obj_id % len(COLOR)], 2)
        # 可选：显示对象ID
        # cv2.putText(frame, f"obj_{obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR[obj_id % len(COLOR)], 2)

    def show_fps(self, frame):
        """显示FPS和统计信息"""
        frame = frame.copy()
        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = curr_time
        
        # 显示FPS
        fps_str = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # 显示检测频率
        freq_str = f"Det Freq: 1/{self.detection_frequency}"
        cv2.putText(frame, freq_str, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        
        # 显示当前对象数
        obj_count_str = f"Objects: {len(self.existing_obj_outputs)}"
        cv2.putText(frame, obj_count_str, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)
        
        return frame

    def detect_new_objects(self, frame):
        """检测新目标（高频版本）"""
        detection = self.gdino.predict(
            [Image.fromarray(frame)],
            [self.current_text_prompt],
            self.box_threshold, 
            self.text_threshold
        )[0]
        
        scores = detection['scores'].cpu().numpy()
        labels = detection['labels']
        boxes = detection['boxes'].cpu().numpy().tolist()
        
        # 转换为numpy数组
        boxes_np = np.array(boxes, dtype=np.int32)
        labels_np = np.array(labels)
        scores_np = np.array(scores)
        
        # 过滤低置信度检测
        filter_mask = scores_np > self.score_threshold
        valid_boxes = boxes_np[filter_mask]
        valid_labels = labels_np[filter_mask]
        valid_scores = scores_np[filter_mask]
        
        return valid_boxes, valid_labels, valid_scores

    def track(self):
        """主跟踪循环"""
        predictor = self.sam.video_predictor

        # 初始化视频源
        if self.mode == "realtime":
            print("Start with realtime mode.")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipeline.start(config)
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            get_frame = lambda: np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
        elif self.mode == "video":
            print("Start with video mode.")
            cap = cv2.VideoCapture(self.video_path)
            # 获取原始视频帧率
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            if original_fps == 0:
                raise
            
            # 如果未指定fps，使用原始视频帧率
            if self.fps is None:
                self.fps = original_fps
                frame_interval = 1  # 不跳帧
                print(f"Using original video FPS: {original_fps}")
            else:
                # 计算帧间隔（每隔多少帧取一帧）
                frame_interval = max(1, round(original_fps / self.fps))
                print(f"Original video FPS: {original_fps}, Target FPS: {self.fps}")
                print(f"Frame interval: processing 1 out of every {frame_interval} frames")
            
            ret, color_image = cap.read()
            get_frame = lambda: cap.read()
        else:
            raise ValueError("The mode is not supported in this method.")

        self.height, self.width = color_image.shape[:2]

        # 初始化视频写入器
        if self.save_video:
            writer = imageio.get_writer(self.output_path, fps=self.fps)
        else:
            writer = None

        # 创建显示窗口
        if not self.headless:
            cv2.namedWindow("Video Tracking")

        # 启动输入线程
        threading.Thread(target=self.input_thread, daemon=True).start()

        print(f"Starting tracking with detection frequency: 1/{self.detection_frequency} frames")
        print(f"IOU threshold: {self.iou_threshold}")
        print(f"Box threshold: {self.box_threshold}, Text threshold: {self.text_threshold}")

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            state = predictor.init_state_from_numpy_frames(
                [color_image], 
                offload_state_to_cpu=False, 
                offload_video_to_cpu=False
            )
            
            # 视频模式下的帧计数器（用于fps控制）
            video_frame_count = 0
            
            while True:
                # 获取下一帧
                if self.mode == "realtime":
                    frame = get_frame()
                else:
                    ret, frame = get_frame()
                    if not ret:
                        break
                    
                    # 视频模式下的fps控制：跳帧处理（仅当frame_interval>1时）
                    video_frame_count += 1
                    if frame_interval > 1 and video_frame_count % frame_interval != 0:
                        continue  # 跳过不需要处理的帧
                
                self.frame_count += 1
                self.frame_display = frame.copy()
                
                # 设置鼠标回调
                if not self.headless:
                    cv2.setMouseCallback("Video Tracking", self.draw_bbox, param=self.frame_display)

                # 检查用户输入
                if not self.input_queue.empty():
                    self.current_text_prompt = self.input_queue.get()
                    print(f"Updated text prompt: {self.current_text_prompt}")

                # 高频检测逻辑
                if self.current_text_prompt is not None:
                    # 检查是否需要执行检测
                    should_detect = (
                        (state['num_frames'] - 1) % self.detection_frequency == 0 or 
                        self.last_text_prompt is None or
                        self.last_text_prompt != self.current_text_prompt
                    )
                    
                    if should_detect:
                        valid_boxes, valid_labels, valid_scores = self.detect_new_objects(frame)
                        self.total_detections += len(valid_boxes)
                        
                        # 文本提示改变时，直接添加所有检测结果
                        if self.last_text_prompt != self.current_text_prompt:
                            if len(valid_boxes) > 0:
                                self.prompts['prompts'].extend(valid_boxes)
                                self.prompts['labels'].extend(valid_labels)
                                self.prompts['scores'].extend(valid_scores)
                                self.add_new = True
                                print(f"New prompt detected, added {len(valid_boxes)} objects")
                        
                        # 已有跟踪对象时，只添加新对象
                        elif self.existing_obj_outputs and len(valid_boxes) > 0:
                            iou_matrix = batch_box_iou(valid_boxes, np.array(self.existing_obj_outputs))
                            is_new = np.max(iou_matrix, axis=1) < self.iou_threshold
                            
                            new_boxes = valid_boxes[is_new]
                            new_labels = valid_labels[is_new]
                            new_scores = valid_scores[is_new]
                            
                            if len(new_boxes) > 0:
                                self.prompts['prompts'].extend(new_boxes)
                                self.prompts['labels'].extend(new_labels)
                                self.prompts['scores'].extend(new_scores)
                                self.add_new = True
                                print(f"Frame {self.frame_count}: Detected {len(new_boxes)} new objects")
                        
                        # 首次检测
                        elif len(valid_boxes) > 0 and not self.existing_obj_outputs:
                            self.prompts['prompts'].extend(valid_boxes)
                            self.prompts['labels'].extend(valid_labels)
                            self.prompts['scores'].extend(valid_scores)
                            self.add_new = True
                            print(f"Frame {self.frame_count}: Initial detection, found {len(valid_boxes)} objects")
                    
                    self.last_text_prompt = self.current_text_prompt

                # 添加新对象到跟踪状态
                if self.add_new:
                    existing_obj_ids = set(state["obj_ids"])
                    predictor.reset_state(state)
                    self.add_to_state(predictor, state, self.prompts)
                    current_obj_ids = set(state["obj_ids"])
                    newly_added_ids = current_obj_ids - existing_obj_ids
                    
                    if newly_added_ids:
                        print(f"Added {len(newly_added_ids)} new objects to tracking state")
                    
                    self.add_new = False

                # 添加帧到推理状态
                predictor.append_frame_to_inference_state(state, frame)
                
                # 跟踪并可视化
                self.track_and_visualize(predictor, state, frame, writer)

                # 内存管理：定期释放旧帧
                if state["num_frames"] % self.max_frames == 0:
                    if len(state["output_dict"]["non_cond_frame_outputs"]) != 0:
                        predictor.append_frame_as_cond_frame(state, state["num_frames"] - 2)
                    predictor.release_old_frames(state)
                    print(f"Released old frames at frame {state['num_frames']}")

                # 检查退出
                if not self.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("User requested exit")
                        break

        # 清理资源
        if self.mode == "realtime":
            pipeline.stop()
        else:
            cap.release()
        
        if writer:
            writer.close()
        
        if not self.headless:
            cv2.destroyAllWindows()
        
        # 打印统计信息
        print("\n=== Tracking Summary ===")
        print(f"Total frames processed: {self.frame_count}")
        if self.mode == "video":
            print(f"Total video frames read: {video_frame_count}")
            if frame_interval > 1:
                print(f"Original FPS: {original_fps:.2f}, Target FPS: {self.fps:.2f}, Frame interval: {frame_interval}")
            else:
                print(f"Video FPS: {self.fps:.2f} (using original, no frame skipping)")
        print(f"Total detections: {self.total_detections}")
        print(f"Detection frequency: 1/{self.detection_frequency}")
        print(f"Final tracked objects: {len(self.existing_obj_outputs)}")
        
        # 清理GPU内存
        del predictor, state
        gc.collect()
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # 示例配置：高频检测版本
    tracker = Lang2SegTrack(
        sam_type="sam2.1_hiera_large",
        model_path="models/sam2/checkpoints/sam2.1_hiera_large.pt",
        video_path="assets/car.mp4",
        output_path="high_frequency_tracked_video.mp4",
        mode="video",
        save_video=True,
        use_txt_prompt=True,
        headless=True,
        detection_frequency=1,  # 每帧检测（可调整为2、3等以降低计算量）
        max_frames=60,
        fps=5  # 可选：指定输出视频帧率，不指定则使用原始视频帧率
    )
    
    # 设置初始文本提示
    tracker.current_text_prompt = 'car'
    
    # 开始跟踪
    tracker.track()
