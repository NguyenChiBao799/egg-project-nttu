# File: src/utils/metrics.py

import torch
import numpy as np
# Cài đặt thư viện torchmetrics để tính toán mAP chính xác
# pip install torchmetrics
# from torchmetrics.detection.mean_ap import MeanAveragePrecision 

# Giả định: CLASS_NAMES đã được định nghĩa
CLASS_NAMES = ['intact', 'cracked', 'broken']
NUM_CLASSES = len(CLASS_NAMES)

def calculate_metrics_for_detection(all_predictions: list, all_targets: list) -> dict:
    """
    Tính toán các chỉ số đánh giá hiệu suất (mAP, F1-score) cho mô hình Object Detection.

    Args:
        all_predictions (list): Danh sách các dự đoán của mô hình. 
                                Mỗi phần tử chứa tensor/array: [x, y, w, h, confidence, class_id].
        all_targets (list): Danh sách các nhãn gốc (ground truths). 
                            Mỗi phần tử chứa tensor/array: [class_id, cx, cy, w, h].
                            
    Returns:
        dict: Chứa các chỉ số như mAP@0.5, mAP@[0.5:0.95], Precision, Recall, F1.
    """
    
    # ----------------------------------------------------------------------
    # TRONG THỰC TẾ, BẠN SẼ SỬ DỤNG THƯ VIỆN CHUYÊN DỤNG:
    # ----------------------------------------------------------------------
    
    # metric = MeanAveragePrecision(box_format="xywh", iou_thresholds=[0.5, 0.75])
    # Tích lũy các batch dự đoán và nhãn
    # metric.update(all_predictions, all_targets)
    # final_metrics = metric.compute() 
    
    # ----------------------------------------------------------------------
    # DỮ LIỆU GIẢ LẬP VÀ MỘT VÀI CHỈ SỐ CƠ BẢN
    # ----------------------------------------------------------------------
    
    # Số lượng mục tiêu (trứng) và dự đoán
    total_targets = sum([t.shape[0] for t in all_targets if t.numel() > 0])
    total_detections = sum([p.shape[0] for p in all_predictions if p.size > 0])
    
    print(f"Tổng số trứng gốc (Ground Truths): {total_targets}")
    print(f"Tổng số dự đoán (Detections): {total_detections}")

    # Giả lập các chỉ số mAP, đây là các giá trị ngẫu nhiên cho mục đích cấu trúc code
    # Bạn sẽ thay thế phần này bằng logic tính mAP thực tế
    metrics = {
        'mAP@0.5': np.random.uniform(0.70, 0.95), # Mean Average Precision tại IoU 0.5
        'mAP': np.random.uniform(0.60, 0.85),     # mAP trung bình trên các ngưỡng IoU (0.5:0.95)
        'Precision': np.random.uniform(0.75, 0.90),
        'Recall': np.random.uniform(0.80, 0.95),
        'F1-score': np.random.uniform(0.77, 0.92),
        'Total_Targets': total_targets,
        'Total_Detections': total_detections,
    }
    
    return metrics
    
# --- Ví dụ về Hàm tính toán cho bài toán Classification (chỉ tham khảo) ---

def calculate_metrics_for_classification(y_true, y_pred):
    """
    Tính toán các chỉ số hiệu suất chính cho Classification (Accuracy, Precision, Recall).
    (Sử dụng Scikit-learn)
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Tính Precision, Recall, F1-score theo phương pháp weighted
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    return metrics