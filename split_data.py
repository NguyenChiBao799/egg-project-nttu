# File: split_data.py (Đã sửa lỗi cú pháp và cập nhật đường dẫn)

import os
import random
import shutil
from glob import glob

# ===============================================
# 📌 CẤU HÌNH ĐƯỜNG DẪN & TỶ LỆ
# ===============================================
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

# ĐƯỜNG DẪN NGUỒN VÀ ĐÍCH ĐỀU LÀ data/processed
SOURCE_BASE_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed') 
SOURCE_DIR = os.path.join(SOURCE_BASE_DIR, 'images') # Ảnh gốc nằm trong data/processed/images
SOURCE_LABELS_DIR = os.path.join(SOURCE_BASE_DIR, 'labels') # Nhãn gốc nằm trong data/processed/labels

# Thư mục đích sẽ được tạo bên trong data/processed (VD: data/processed/images/train)
TARGET_BASE = SOURCE_BASE_DIR 

# Tỷ lệ phân chia: 60% Training / 20% Validation / 20% Test
TRAIN_RATIO = 0.60  # 60%
VAL_RATIO = 0.20    # 20%
TEST_RATIO = 0.20   # 20%
assert TRAIN_RATIO + VAL_RATIO + TEST_RATIO == 1.0, "Tỷ lệ phân chia phải bằng 1.0"


# ===============================================
# 🔨 HÀM PHÂN CHIA VÀ DI CHUYỂN
# ===============================================
def split_data():
    # 1. Lấy danh sách tất cả các ảnh gốc (.jpg, .png, etc.)
    image_files = glob(os.path.join(SOURCE_DIR, '*.jpg')) + \
                  glob(os.path.join(SOURCE_DIR, '*.png'))
    
    random.seed(42) # Đảm bảo kết quả phân chia giống nhau qua các lần chạy
    random.shuffle(image_files) # Trộn ngẫu nhiên (quan trọng!)
    N = len(image_files)
    
    if N == 0:
        print("❌ Lỗi: Không tìm thấy ảnh trong thư mục nguồn:", SOURCE_DIR)
        print("Vui lòng đảm bảo các ảnh gốc (trước Augmentation) đang nằm trực tiếp trong thư mục này.")
        return

    # Tính toán số lượng cho từng tập
    n_train = int(N * TRAIN_RATIO)
    n_val = int(N * VAL_RATIO)
    n_test = N - n_train - n_val # Phần còn lại

    # Phân chia danh sách files
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]

    print(f"✅ Tìm thấy {N} ảnh gốc. Phân chia tỷ lệ 60/20/20:")
    print(f"   - Training: {len(train_files)} ảnh")
    print(f"   - Validation: {len(val_files)} ảnh")
    print(f"   - Test: {len(test_files)} ảnh")

    # 💡 ĐỊNH NGHĨA BIẾN 'splits' ĐÃ BỊ THIẾU TRƯỚC ĐÓ
    splits = {
        'train': train_files, 
        'val': val_files, 
        'test': test_files
    }

    # Tạo thư mục đích nếu chưa có
    for split_name in splits.keys():
        os.makedirs(os.path.join(TARGET_BASE, 'images', split_name), exist_ok=True)
        os.makedirs(os.path.join(TARGET_BASE, 'labels', split_name), exist_ok=True)


    # 2. Di chuyển ảnh và nhãn
    for split_name, file_list in splits.items():
        target_img_dir = os.path.join(TARGET_BASE, 'images', split_name)
        target_label_dir = os.path.join(TARGET_BASE, 'labels', split_name)

        print(f"\nDi chuyển {len(file_list)} file tới {split_name}...")
        
        for img_path in file_list:
            base_name = os.path.basename(img_path)
            stem = os.path.splitext(base_name)[0]
            label_name = stem + '.txt'
            
            label_path = os.path.join(SOURCE_LABELS_DIR, label_name)
            
            # Xử lý Hard Negative: Nếu không có file nhãn, tạo file rỗng trước khi di chuyển
            if not os.path.exists(label_path):
                # Tạo file .txt rỗng (0 KB) cho ảnh Hard Negative
                with open(label_path, 'w') as f:
                    pass

            # Di chuyển ảnh và nhãn
            shutil.move(img_path, os.path.join(target_img_dir, base_name))
            shutil.move(label_path, os.path.join(target_label_dir, label_name))
            
    print("\n✅ Quá trình phân chia dữ liệu hoàn tất!")


if __name__ == "__main__":
    split_data()