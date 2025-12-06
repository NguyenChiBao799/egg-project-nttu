# File: src/models/evaluate.py

import torch
from ultralytics import YOLO
from .model_architectures import create_yolo_model, NUM_CLASSES
from .train import DATA_YAML_PATH

def evaluate_model(model, data_loader=None, device='cuda', save_dir='trained_models'):
    """
    Đánh giá mô hình YOLOv8 trên tập Validation hoặc Test bằng phương thức .val().
    
    Args:
        model: Đối tượng ultralytics.YOLO đã được huấn luyện.
        data_loader: Không cần thiết vì YOLOv8 sử dụng file YAML.
        device: Thiết bị chạy (cuda/cpu).
        
    Returns:
        dict: Các chỉ số hiệu suất mAP.
    """
    
    print("\n[EVALUATE] Bắt đầu đánh giá mô hình trên tập Validation...")
    
    # Thực hiện đánh giá
    metrics = model.val(
        data=DATA_YAML_PATH, 
        imgsz=416, 
        device=device,
        split='val'
    )
    
    # Lấy các chỉ số quan trọng
    results_metrics = {
        'mAP@0.5': metrics.box.map50,
        'mAP@0.5:0.95': metrics.box.map,
        'Precision': metrics.box.precision,
        'Recall': metrics.box.recall,
        'Total_Instances': metrics.box.n_cls
    }
    
    return results_metrics