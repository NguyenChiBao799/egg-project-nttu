# File: src/models/model_architectures.py

import os
import torch
# SỬ DỤNG THƯ VIỆN ULTRALYTICS CHO MÔ HÌNH YOLOv8
from ultralytics import YOLO

# ===============================================================
# 🧩 CẤU HÌNH MÔ HÌNH YOLOv8m DETECTION CHO TRỨNG NỨT / KHÔNG NỨT
# ===============================================================

# ✅ Chỉ còn 2 lớp (not_damaged, damaged)
NUM_CLASSES = 2
CLASS_NAMES = ['not_damaged', 'damaged']


def create_yolo_model(num_classes: int = NUM_CLASSES, weights_path: str = None, model_type: str = 'yolov8m'):
    """
    Khởi tạo mô hình Object Detection YOLOv8 (Medium version) cho bài toán phát hiện trứng nứt.
    
    Args:
        num_classes (int): Số lượng lớp cần phát hiện (2: not_damaged, damaged).
        weights_path (str): Đường dẫn đến file trọng số đã huấn luyện trước (nếu có).
        model_type (str): Phiên bản YOLO muốn sử dụng ('yolov8n', 'yolov8s', 'yolov8m', ...).

    Returns:
        ultralytics.YOLO: Đối tượng mô hình YOLOv8 sẵn sàng huấn luyện hoặc suy luận.
    """
    # 1️⃣ Tải mô hình cơ sở
    if weights_path and os.path.exists(weights_path):
        # Tải từ trọng số đã lưu (checkpoint)
        model = YOLO(weights_path)
        print(f"✅ Mô hình YOLOv8 đã tải từ file trọng số: {weights_path}")
    else:
        # Tải mô hình đã được huấn luyện trước trên tập lớn (COCO)
        model = YOLO(f'{model_type}.pt')
        print(f"📦 Đã tải mô hình cơ sở: {model_type}.pt (pretrained COCO)")

    # 2️⃣ Ghi chú về cấu hình lớp đầu ra
    # Ultralytics YOLOv8 sẽ tự động điều chỉnh số lớp đầu ra (output layer)
    # khi bạn huấn luyện với file YAML chứa "nc" và "names" tương ứng.
    # Vì vậy không cần thủ công sửa cấu trúc layer ở đây.

    print(f"⚙️ Cấu hình mô hình: {num_classes} lớp ({CLASS_NAMES})")
    return model
