# File: src/models/train.py

import os
import shutil
from ultralytics import YOLO
import torch

# ===============================================================
# 🧩 CẤU HÌNH DỮ LIỆU
# ===============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
PROCESSED_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed')

DATA_YAML_PATH = os.path.join(PROCESSED_PATH, 'data.yaml')
CUSTOM_MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'yolov8s_custom.yaml')

NUM_CLASSES = 2
CLASS_NAMES = ['egg', 'damaged egg']
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'trained_models', 'best_roboflow_egg_v3.pt')
PREVIOUS_BEST_WEIGHTS = os.path.join(PROJECT_ROOT, 'runs', 'train', 'roboflow_egg_detector', 'weights', 'best.pt')


# ===============================================================
# 🧠 HUẤN LUYỆN YOLOv8s (Detection)
# ===============================================================
def train_model(img_size=420, num_epochs=50, learning_rate=2e-4, device=None, class_weights=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Bắt đầu huấn luyện mô hình YOLOv8s trên {device.upper()}")

    # Chọn trọng số khởi đầu
    initial_weights = 'yolov8s.pt'
    if os.path.exists(PREVIOUS_BEST_WEIGHTS):
        initial_weights = PREVIOUS_BEST_WEIGHTS
        print(f"🌟 Bắt đầu huấn luyện lại từ trọng số tốt nhất cũ: {initial_weights}")

    # ============================================================
    # ⚡ PATCH BẮT BUỘC — FIX LOAD WEIGHTS TRÊN PYTORCH 2.6
    # ============================================================
    original_load = torch.load

    def patched_load(*args, **kwargs):
        # ép YOLO dùng chế độ full pickle load
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load
    print("⚡ [Patch] torch.load forced to weights_only=False")

    # ============================================================
    # KHỞI TẠO YOLO (sau khi patch)
    # ============================================================
    model = YOLO(initial_weights)

    os.makedirs(os.path.join(PROJECT_ROOT, 'trained_models'), exist_ok=True)

    results = model.train(
        data=DATA_YAML_PATH,
        imgsz=img_size,
        epochs=num_epochs,
        batch=4,             # máy yếu → batch nhỏ
        workers=0,
        lr0=learning_rate,

        # ⚡ tăng trọng số phân loại
        cls=1.5,   # phạt sai class mạnh hơn
        dfl=1.0,
        box=1.3,   # phạt sai box nhiều hơn

        # ⚡ augment phù hợp dataset 1500 ảnh
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        translate=0.05,
        scale=0.15,
        degrees=2,

        # KHÔNG DÙNG blur / sharp (YOLO không hỗ trợ)
        # KHÔNG DÙNG class_weights (YOLO không hỗ trợ)

        mosaic=0.1,
        mixup=0.0,

        patience=25,
        device=device

    )

    # ============================================================
    # SAO CHÉP TRỌNG SỐ TỐT NHẤT
    # ============================================================
    best_weights_src = os.path.join(
        PROJECT_ROOT, 'runs', 'train',
        'roboflow_egg_detector_finetune_v2', 'weights', 'best.pt'
    )

    if os.path.exists(best_weights_src):
        shutil.copy(best_weights_src, WEIGHTS_PATH)
        print(f"✅ Đã sao chép trọng số tốt nhất về: {WEIGHTS_PATH}")
    else:
        print("⚠️ Không tìm thấy trọng số tốt nhất sau khi huấn luyện.")


if __name__ == "__main__":
    train_model()
