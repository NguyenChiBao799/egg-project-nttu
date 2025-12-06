# File: src/data_processing/augmentation.py

import cv2
import numpy as np
import albumentations as A
from typing import List, Tuple, Dict

# Định nghĩa các lớp theo thứ tự YOLO: 0, 1, 2
CLASS_NAMES = ['intact', 'cracked', 'broken']

def get_train_augmentation(image_size: int) -> A.Compose:
    """
    Định nghĩa pipeline tăng cường dữ liệu (augmentation) cho tập huấn luyện.
    Sử dụng Albumentations với định dạng YOLO (Normalized bounding boxes).
    """
    return A.Compose(
        [
            # --- 1. Biến đổi Hình học (Geometric Transformations) ---
            A.HorizontalFlip(p=0.5),      # Lật ngang
            A.VerticalFlip(p=0.1),        # Lật dọc (ít)
            A.ShiftScaleRotate(
                shift_limit=0.05,         # Dịch chuyển 5%
                scale_limit=0.15,         # Thay đổi tỷ lệ 15%
                rotate_limit=15,          # Xoay 15 độ
                p=0.6, 
                border_mode=cv2.BORDER_CONSTANT 
            ),                            
            
            # --- 2. Biến đổi Màu sắc và Độ sáng (Color and Brightness) ---
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.ToGray(p=0.05),             
            A.ChannelShuffle(p=0.05),     
            
            # --- 3. Biến đổi Giả lập Môi trường ---
            A.GaussianBlur(blur_limit=(3, 7), p=0.1), 
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.15), 
            A.CLAHE(p=0.1),               
            
            # --- 4. Cắt và Resize ---
            A.Resize(image_size, image_size), 
        ], 
        # Cấu hình Bounding Box
        bbox_params=A.BboxParams(
            format='yolo',  # Định dạng [x_center, y_center, width, height] (Normalized)
            label_fields=['class_labels'] 
        )
    )

def get_val_transform(image_size: int) -> A.Compose:
    """
    Chỉ định pipeline biến đổi chuẩn hóa cho tập validation/test.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size), 
        ],
        bbox_params=A.BboxParams(
            format='yolo', 
            label_fields=['class_labels']
        )
    )

# Hàm này cần thiết để giữ nguyên các import cũ (nếu có)
def get_test_transforms(image_size: int) -> A.Compose:
    return get_val_transform(image_size)