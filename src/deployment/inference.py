# File: src/deployment/inference.py (Đã sửa lỗi UnpicklingError)

import torch
import numpy as np
import cv2
import os
from typing import List, Dict, Tuple, Any

# Cần import YOLO từ ultralytics
from ultralytics import YOLO 
# Dù model_architectures có hàm tạo, ta sẽ tải trực tiếp ở đây
# from src.models.model_architectures import create_yolo_model 

# Các lớp đã định nghĩa: 0: intact, 1: cracked, 2: broken
CLASS_NAMES = ['egg', 'damaged egg']
DEFAULT_WEIGHTS_PATH = 'trained_models/best_egg_detector.pt'


# --- HÀM NÀY ĐÃ ĐƯỢC SỬA LỖI TẢI YOLOv8 ---
def load_model(weights_path: str = DEFAULT_WEIGHTS_PATH, 
               num_classes: int = 3, 
               device: str = 'cpu') -> YOLO: # Trả về đối tượng YOLO
    """
    Tải mô hình YOLOv8 đã được huấn luyện bằng phương thức chính xác của Ultralytics.
    """
    try:
        if not os.path.exists(weights_path):
             print(f"LỖI: Không tìm thấy file trọng số tại {weights_path}. Chạy chế độ giả lập.")
             # Tải mô hình cơ sở (nếu không tìm thấy trọng số)
             model = YOLO('yolov8n.pt') 
             return model
             
        # Tải mô hình YOLOv8 trực tiếp từ file .pt
        model = YOLO(weights_path)
        
        # Đặt thiết bị và chế độ đánh giá
        model.to(device)
        model.eval() 
        print(f"Mô hình YOLOv8 đã tải thành công từ: {weights_path}")
        return model
        
    except Exception as e:
        print(f"LỖI TẢI MÔ HÌNH: {e}")
        # Trả về mô hình cơ sở nếu gặp lỗi nghiêm trọng
        return YOLO('yolov8n.pt') 


def preprocess_image(image_data: np.ndarray, image_size: int = 416) -> torch.Tensor:
    """
    Tiền xử lý ảnh đầu vào (Không cần thiết cho YOLO.predict, nhưng giữ lại cho sự nhất quán)
    """
    # ... (Giữ nguyên logic tiền xử lý nếu cần)
    image = cv2.resize(image_data, (image_size, image_size))
    image_tensor = image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).unsqueeze(0) 
    return image_tensor


# --- HÀM NÀY ĐÃ ĐƯỢC SỬA LỖI ĐỂ SỬ DỤNG PHƯƠNG THỨC .predict() CỦA ULTRALYTICS ---
def predict_image(model: YOLO, image_data: np.ndarray, 
                  confidence_threshold: float = 0.5) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """
    Chạy suy luận trên ảnh đầu vào bằng phương thức .predict() của YOLOv8.
    """
    H, W, _ = image_data.shape
    
    # Chạy suy luận YOLOv8
    # Source: https://docs.ultralytics.com/modes/predict/
    results = model.predict(
        source=image_data, 
        conf=confidence_threshold, 
        imgsz=416, 
        verbose=False,
        save=False,
        device=model.device # Sử dụng thiết bị đã tải
    )
    
    # Lấy ảnh đã vẽ và kết quả từ đối tượng Results
    if not results:
        return [], image_data.copy()

    result = results[0] # Lấy kết quả đầu tiên (và duy nhất)
    
    # Lấy ảnh đã vẽ Bounding Box (chuyển từ BGR sang RGB nếu cv2/ultralytics làm mờ ảnh)
    drawn_image = result.plot() 
    if drawn_image.shape[-1] == 3 and drawn_image.dtype != np.uint8:
        drawn_image = (drawn_image * 255).astype(np.uint8)
        
    drawn_image = cv2.cvtColor(drawn_image, cv2.COLOR_BGR2RGB) # YOLO plot dùng BGR
    
    final_detections = []
    
    # Xử lý kết quả đầu ra
    if result.boxes:
        boxes = result.boxes.xyxy.cpu().numpy() # xmin, ymin, xmax, ymax
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for box, conf, cls_id in zip(boxes, confidences, classes):
            final_detections.append({
                'class': CLASS_NAMES[int(cls_id)],
                'confidence': float(conf),
                'box': (int(box[0]), int(box[1]), int(box[2]), int(box[3])) # Tọa độ pixel
            })

    return final_detections, drawn_image