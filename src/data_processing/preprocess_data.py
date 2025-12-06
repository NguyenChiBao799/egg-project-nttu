# ==========================================================
# ⚡ PyTorch 2.6 bypass (giữ nguyên như trước)
# ==========================================================
import sys
sys.stdout.reconfigure(encoding='utf-8')

import torch
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel

_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)

torch.load = _patched_load
add_safe_globals([DetectionModel])
print("⚡ PyTorch 2.6 bypass applied")


# ==========================================================
# 📁 IMPORT
# ==========================================================
import os
import random
import shutil
import yaml


# ==========================================================
# 📁 PATH
# ==========================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_YAML_PATH = os.path.join(PROJECT_ROOT, "egg_dataset.yaml")

RAW_IMAGES = os.path.join(PROJECT_ROOT, "data", "dataset_all", "images")
RAW_LABELS = os.path.join(PROJECT_ROOT, "data", "dataset_all", "labels")

# Nếu không có RAW → dùng processed làm nguồn
if not os.path.exists(RAW_IMAGES):
    RAW_IMAGES = os.path.join(PROJECT_ROOT, "data", "processed", "images")
    RAW_LABELS = os.path.join(PROJECT_ROOT, "data", "processed", "labels")
    print("⚠️ Không tìm thấy data/dataset_all → dùng data/processed làm nguồn.")


# ==========================================================
# 📁 TẠO FOLDER THEO ĐÚNG CẤU TRÚC BẠN YÊU CẦU
# ==========================================================
def ensure_structure():
    folders = [
        "train/images", "train/labels",
        "val/images", "val/labels",
        "test/images", "test/labels"
    ]
    for f in folders:
        os.makedirs(os.path.join(PROCESSED, f), exist_ok=True)
    print("📂 Đã tạo cấu trúc dataset train/val/test")


# ==========================================================
# 🔀 CHIA 60/20/20
# ==========================================================
def auto_split_dataset():
    print("🔀 Đang chia dataset 60/20/20...")

    images = [f for f in os.listdir(RAW_IMAGES)
              if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if not images:
        raise RuntimeError(f"❌ Không tìm thấy ảnh trong {RAW_IMAGES}")

    random.shuffle(images)

    total = len(images)
    train_end = int(total * 0.6)
    val_end = int(total * 0.8)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        print(f"  {split}: {len(files)} ảnh")

        for img in files:
            src_img = os.path.join(RAW_IMAGES, img)
            dst_img = os.path.join(PROCESSED, f"{split}/images/{img}")
            shutil.copy(src_img, dst_img)

            # label tương ứng
            label_file = os.path.splitext(img)[0] + ".txt"
            src_label = os.path.join(RAW_LABELS, label_file)
            dst_label = os.path.join(PROCESSED, f"{split}/labels/{label_file}")
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)

    print("🎉 Chia dataset thành công.")


# ==========================================================
# 📝 TẠO YAML THEO ĐÚNG MẪU BẠN YÊU CẦU
# ==========================================================
def create_data_yaml():
    yaml_content = {
        "train": f"{PROCESSED}/train/images",
        "val": f"{PROCESSED}/val/images",
        "test": f"{PROCESSED}/test/images",

        "nc": 2,
        "names": ["normal", "crack"]
    }

    with open(DATA_YAML_PATH, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, sort_keys=False, allow_unicode=True)

    print("📝 Đã tạo egg_dataset.yaml")
    print("👉 Path:", DATA_YAML_PATH)


# ==========================================================
# 🚀 RUN
# ==========================================================
if __name__ == "__main__":
    ensure_structure()
    auto_split_dataset()
    create_data_yaml()
    print("✅ Preprocess hoàn tất.")
