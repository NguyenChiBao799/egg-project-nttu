# File: src/data_processing/data_loader.py (Đã Refactor cho Object Detection)

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# Giữ nguyên import tương đối (Nếu trước đây là .augmentation)
from .augmentation import get_train_augmentation, get_test_transforms, get_val_transform 
from typing import Tuple, List

# Định nghĩa các đường dẫn cố định dựa trên cấu trúc mô hình
PROCESSED_DATA_PATH = 'data/processed'

class EggDetectionDataset(Dataset):
    """
    Dataset cho bài toán Phát hiện Vật thể Trứng vỡ.
    Đọc ảnh và nhãn (bounding box) theo định dạng YOLO (.txt).
    """
    def __init__(self, split='train', image_size=416):
        
        # Cấu trúc: data/processed/images/train hoặc data/processed/images/val
        self.image_dir = os.path.join(PROCESSED_DATA_PATH, 'images', split)
        self.label_dir = os.path.join(PROCESSED_DATA_PATH, 'labels', split)
        # Hỗ trợ cả .jpg và .png
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg') or f.endswith('.png')] 
        self.image_size = image_size
        self.split = split
        
        # Chọn Augmentation/Transforms phù hợp
        if split == 'train':
            self.transform = get_train_augmentation(image_size)
        else:
            # Sử dụng get_val_transform (tương đương get_test_transforms)
            self.transform = get_val_transform(image_size) 

    def __len__(self):
        return len(self.image_files)

    def load_yolo_labels(self, label_path: str) -> np.ndarray:
        """Đọc nhãn YOLO (.txt) và chuyển thành numpy array: [class_id, x_c, y_c, w, h]"""
        if not os.path.exists(label_path):
            return np.empty((0, 5), dtype=np.float32) # Trả về mảng rỗng
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            boxes = [list(map(float, line.strip().split())) for line in lines]
            # Đảm bảo tất cả các nhãn đều có 5 cột (class_id + 4 tọa độ)
            boxes = [b for b in boxes if len(b) == 5] 
            return np.array(boxes, dtype=np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_filename = self.image_files[idx]
        label_filename = img_filename.replace('.jpg', '.txt').replace('.png', '.txt')
        
        img_path = os.path.join(self.image_dir, img_filename)
        label_path = os.path.join(self.label_dir, label_filename)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albumentations dùng RGB
        
        # Tải nhãn
        labels_yolo = self.load_yolo_labels(label_path)
        
        if labels_yolo.size == 0:
            bboxes = []
            class_labels = []
        else:
            # Tách class_labels và bboxes [x_c, y_c, w, h] (normalized)
            class_labels = labels_yolo[:, 0]
            bboxes = labels_yolo[:, 1:] 
            
        # Áp dụng Augmentation/Transform
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        transformed_image = transformed['image']
        transformed_bboxes = np.array(transformed['bboxes'], dtype=np.float32)
        transformed_class_labels = np.array(transformed['class_labels'], dtype=np.float32)

        # Chuyển đổi về định dạng Tensor PyTorch
        if transformed_bboxes.size > 0:
            # Tạo tensor [class_id, x_c, y_c, w, h] (normalized)
            final_target = np.concatenate((
                transformed_class_labels[:, None],
                transformed_bboxes
            ), axis=1)
            final_target = torch.from_numpy(final_target)
        else:
            final_target = torch.empty((0, 5), dtype=torch.float32)

        # Chuẩn hóa ảnh và chuyển định dạng (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(transformed_image).permute(2, 0, 1).float() / 255.0
        
        return image_tensor, final_target


def custom_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hàm Collate tùy chỉnh cần thiết cho Object Detection.
    Nó thêm chỉ số batch vào nhãn để biết nhãn thuộc ảnh nào trong batch.
    """
    images = []
    targets = []
    
    for i, (img, boxes) in enumerate(batch):
        images.append(img)
        
        # Thêm cột đầu tiên là chỉ số batch (giá trị 'i')
        if boxes.numel() > 0:
            batch_index = torch.full((boxes.shape[0], 1), i, dtype=torch.float32)
            # targets là: [batch_idx, class_id, x_c, y_c, w, h]
            targets.append(torch.cat((batch_index, boxes), dim=1))
        
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0) if targets else torch.empty((0, 6), dtype=torch.float32) 
    
    return images, targets


def create_yolo_dataloaders(batch_size: int = 16, image_size: int = 640, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Tạo DataLoader cho tập huấn luyện và kiểm thử theo định dạng YOLO.
    """
    train_dataset = EggDetectionDataset(split='train', image_size=image_size)
    val_dataset = EggDetectionDataset(split='val', image_size=image_size)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True 
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Hàm này được định nghĩa để giữ nguyên logic nếu nó từng tồn tại
    def get_classification_dataloaders(batch_size=16, image_size=640):
         print("CẢNH BÁO: Hàm get_classification_dataloaders bị bỏ qua vì đây là bài toán Object Detection.")
         return train_loader, val_loader, ['intact', 'cracked', 'broken']
    
    return train_loader, val_loader