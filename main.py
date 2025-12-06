# File: main.py

import os
import argparse
import torch
import shutil
import sys
from importlib import util
import subprocess

# ============================================================
# 1) PATCH TORCH LOAD — BẮT BUỘC (PHẢI CHẠY TRƯỚC YOLO)
# ============================================================
import torch

_original_torch_load = torch.load

def _patched_load(*args, **kwargs):
    # ép YOLO load full pickle (bắt buộc cho PyTorch 2.6 trở lên)
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_load
print("⚡ [Patch] torch.load patched (weights_only=False enforced)")

# --- Import các hàm chính từ module ---
from src.models.train import train_model 
from src.models.model_architectures import NUM_CLASSES


# --- Cấu hình Chung (Hyperparameters) ---
HYPERPARAMS = {
    'BATCH_SIZE': 16,
    'IMAGE_SIZE': 512,       # Kích thước ảnh
    'NUM_EPOCHS': 50,        # Số epochs
    'LEARNING_RATE': 5e-5,   # Learning rate
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'DATA_YAML_PATH': os.path.join('data', 'processed', 'data.yaml'),
    'BEST_WEIGHTS_PATH': os.path.join('trained_models', 'best_roboflow_egg.pt'),
    
    # ✅ ĐIỀU CHỈNH TRỌNG SỐ LOSS MỚI (Dựa trên data.yaml)
    # Lớp 0: damaged egg (Trọng số thấp hơn: 0.4)
    # Lớp 1: egg (Trọng số cao hơn: 0.6)
    # Tỷ lệ 40:60 để làm cho mô hình nghiêm khắc hơn khi dự đoán sai lớp 'egg'.
    'CLASS_WEIGHTS': [0.55, 0.45] 
}


# ===============================================================
# 🥚 TIỀN XỬ LÝ DỮ LIỆU
# ===============================================================
def run_preprocess():
    """Chạy script tiền xử lý dữ liệu nếu dùng dataset nội bộ."""
    print("\n============================")
    print("🚀 BẮT ĐẦU: TIỀN XỬ LÝ DỮ LIỆU")
    print("============================")

    preprocess_script_path = 'src/data_processing/preprocess_data.py'

    if os.path.exists(preprocess_script_path):
        try:
            result = subprocess.run(
                [sys.executable, preprocess_script_path],
                check=False,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            print(result.stdout)
            if result.returncode == 0:
                print("\n✅ [HOÀN TẤT] Tiền xử lý dữ liệu đã hoàn thành thành công.")
                print("⚙️ File YAML Roboflow sẽ được dùng trong huấn luyện, không cần tạo lại.")
            else:
                print(f"\n❌ [LỖI] Tiền xử lý thất bại (mã lỗi {result.returncode})")
                print(result.stderr)
        except Exception as e:
            print(f"⚠️ Lỗi khi chạy tiền xử lý: {e}")
    else:
        print(f"❌ [LỖI] Không tìm thấy script tiền xử lý tại: {preprocess_script_path}")

    print("-" * 60)


# ===============================================================
# 🧠 HUẤN LUYỆN MÔ HÌNH
# ===============================================================
def run_training_pipeline():
    """
    Thực hiện luồng huấn luyện: Tải mô hình -> Huấn luyện bằng YOLOv8.
    """
    print("\n============================")
    print("🧠 BẮT ĐẦU HUẤN LUYỆN MÔ HÌNH")
    print("============================")

    # Kiểm tra file YAML
    if not os.path.exists(HYPERPARAMS['DATA_YAML_PATH']):
        print(f"⚠️ [CẢNH BÁO] Không tìm thấy dataset config: {HYPERPARAMS['DATA_YAML_PATH']}")
        print("👉 Vui lòng đảm bảo Roboflow export có file data.yaml.")
        return

    print(f"Thiết bị: {HYPERPARAMS['DEVICE'].upper()}")
    print(f"Batch Size: {HYPERPARAMS['BATCH_SIZE']} | Epochs: {HYPERPARAMS['NUM_EPOCHS']} | LR: {HYPERPARAMS['LEARNING_RATE']}")
    print(f"📄 Dataset YAML: {HYPERPARAMS['DATA_YAML_PATH']}")
    print(f"⚖️ Trọng số lớp: {HYPERPARAMS['CLASS_WEIGHTS']} (0: damaged egg, 1: egg)")

    try:
        # ✅ TRUYỀN THAM SỐ TRỌNG SỐ LỚP MỚI
        train_model(
            img_size=HYPERPARAMS['IMAGE_SIZE'],
            num_epochs=HYPERPARAMS['NUM_EPOCHS'],
            learning_rate=HYPERPARAMS['LEARNING_RATE'],
            device=HYPERPARAMS['DEVICE'],
            class_weights=HYPERPARAMS['CLASS_WEIGHTS'] 
        )
        print("\n✅ [HOÀN TẤT] Huấn luyện mô hình thành công.")
    except KeyboardInterrupt:
        print("\n🛑 Huấn luyện bị hủy bởi người dùng.")
    except Exception as e:
        print(f"\n❌ Lỗi khi huấn luyện mô hình: {e}")

    print("-" * 60)


# ===============================================================
# 🧩 CHẠY ỨNG DỤNG STREAMLIT DEMO
# ===============================================================
def run_streamlit_app():
    """Khởi động ứng dụng demo bằng Streamlit."""
    print("\n============================")
    print("🎛️ KHỞI ĐỘNG ỨNG DỤNG DEMO")
    print("============================")

    app_script_path = 'src/deployment/app.py'
    project_root = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(app_script_path):
        print(f"❌ [LỖI] Không tìm thấy app.py tại: {app_script_path}")
        return

    # KIỂM TRA TỆP TRỌNG SỐ TỐT NHẤT
    best_weights_path = HYPERPARAMS['BEST_WEIGHTS_PATH']
    if not os.path.exists(best_weights_path):
        print(f"❌ [LỖI] Không tìm thấy mô hình tốt nhất tại: {best_weights_path}")
        print("👉 Vui lòng chạy hành động 'train' trước để có file trọng số.")
        return

    if not util.find_spec("streamlit"):
        print("⚠️ [LỖI] Chưa cài 'streamlit'. Vui lòng chạy: pip install streamlit")
        return

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(project_root, "src")
        # TRUYỀN ĐƯỜNG DẪN TRỌNG SỐ TỐT NHẤT VÀO BIẾN MÔI TRƯỜNG
        env["BEST_WEIGHTS_PATH"] = os.path.join(project_root, best_weights_path)

        print(f"💡 Đang tải mô hình: {best_weights_path}")
        print("💡 Đang khởi động Streamlit...")
        print("-" * 50)
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", app_script_path],
            env=env,
            check=False
        )
    except Exception as e:
        print(f"❌ Lỗi khi khởi động Streamlit: {e}")

    print("-" * 60)


# ===============================================================
# ⚙️ HÀM MAIN ĐIỀU PHỐI
# ===============================================================
def main():
    parser = argparse.ArgumentParser(description="🥚 Egg Crack Detector AI Pipeline (YOLOv8)")
    parser.add_argument(
        'action',
        choices=['preprocess', 'train', 'run_app'],
        help="Chọn hành động: preprocess (tiền xử lý dữ liệu), train (huấn luyện mô hình), hoặc run_app (chạy ứng dụng demo)."
    )
    args = parser.parse_args()

    if args.action == 'preprocess':
        run_preprocess()
    elif args.action == 'train':
        run_training_pipeline()
    elif args.action == 'run_app':
        run_streamlit_app()


if __name__ == "__main__":
    main()