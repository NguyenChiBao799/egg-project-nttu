"""
===============================================================
🎓 HỆ THỐNG KIỂM ĐỊNH CHẤT LƯỢNG TRỨNG BẰNG AI – ỨNG DỤNG STREAMLIT
Phục vụ Chương 3 & 4 của luận văn tốt nghiệp
===============================================================

📌 MÔ TẢ TỔNG QUAN (PHÙ HỢP CHƯƠNG 3 – PHÂN TÍCH THIẾT KẾ)

Ứng dụng được thiết kế theo kiến trúc 3 lớp:

1) Presentation Layer (UI/UX – Streamlit):
    - Cung cấp giao diện web trực quan.
    - Cho phép người dùng tải ảnh hoặc sử dụng camera.
    - Xử lý trải nghiệm người dùng bằng CSS, layout.

2) Application Layer (YOLO Inference + ViT Verification):
    - Thực hiện pipeline gồm:
        + Phát hiện trứng và trứng nứt bằng YOLOv8.
        + Xác minh lại độ chính xác nhờ ViT-B/16 (Vision Transformer).
    - Đảm bảo tốc độ xử lý nhanh, mô hình nhẹ phù hợp môi trường không GPU.

3) Data Processing Layer:
    - Chuyển đổi ảnh đầu vào sang định dạng tensor.
    - Tối ưu hoá pipeline tiền xử lý.
    - Hỗ trợ kiểm thử và phân tích kết quả.

📌 YÊU CẦU PHI CHỨC NĂNG (CHƯƠNG 3 – Non-functional):
    - Thời gian phản hồi < 1.2s trên CPU (i7-13620H).
    - Giao diện trực quan, dễ dùng.
    - Có khả năng mở rộng tích hợp camera và API sau này.
    - Đảm bảo mô hình chạy ổn định không GPU.

📌 CHƯƠNG 4 – TRIỂN KHAI VÀ KIỂM THỬ:
    - Mã nguồn dưới đây thể hiện đầy đủ:
        + Quá trình load mô hình
        + Pipeline dự đoán
        + Giao diện hiển thị kết quả
        + Sơ đồ xử lý được mô tả trong luận văn

===============================================================
"""

# ===============================================================
# 🔗 IMPORT – Giữ nguyên theo yêu cầu
# ===============================================================

import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import torch
import json
from ultralytics import YOLO
from torch.serialization import add_safe_globals
from ultralytics.nn.modules.conv import Concat
from datetime import datetime
import platform
import uuid

# ===============================================================
# 🔥 FIX PYTORCH 2.6 — KHÔNG ĐƯỢC XOÁ
# ===============================================================

_original_torch_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_load
add_safe_globals([Concat])

# ===============================================================
# 🤖 IMPORT VIT VALIDATOR (Local module)
# ===============================================================

from src.ai_modules.vit_classifier import VITEggValidator


# ===============================================================
# 🧩 CONFIG
# ===============================================================

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

MODEL_PATH_ENV = os.environ.get("BEST_WEIGHTS_PATH")
MODEL_PATH = MODEL_PATH_ENV if MODEL_PATH_ENV else os.path.join(
    PROJECT_ROOT, "trained_models", "best_egg_detector.pt"
)

st.set_page_config(
    page_title="Egg Inspector AI - Professional Edition",
    layout="wide",
    page_icon="🥚",
    initial_sidebar_state="expanded"
)


# ===============================================================
# 🎨 CUSTOM CSS (UI Layer) - Enhanced Professional Design
# ===============================================================

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main { 
            background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
        }
        
        .header-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
            border-radius: 20px;
            margin-bottom: 30px;
            box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        }
        
        .header-title {
            font-size: 42px;
            font-weight: 800;
            color: white;
            margin-bottom: 8px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        .sub-header { 
            font-size: 18px; 
            color: rgba(255,255,255,0.9);
            font-weight: 500;
        }
        
        .metric-card {
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
            margin-bottom: 20px;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.12);
        }
        
        .metric-value {
            font-size: 36px;
            font-weight: 800;
            margin: 10px 0;
        }
        
        .metric-label {
            font-size: 14px;
            color: #6c757d;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .success-metric { color: #00c853; }
        .danger-metric { color: #d32f2f; }
        .info-metric { color: #1e88e5; }
        .warning-metric { color: #ff6f00; }
        
        .result-card {
            padding: 20px;
            border-radius: 15px;
            background: white;
            margin-bottom: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            border-left: 5px solid #e0e0e0;
            transition: all 0.3s;
        }
        
        .result-card:hover {
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .egg { 
            border-left-color: #00c853;
            background: linear-gradient(to right, rgba(0,200,83,0.05), white);
        }
        
        .damaged { 
            border-left-color: #d32f2f;
            background: linear-gradient(to right, rgba(211,47,47,0.05), white);
        }
        
        .info-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
        }
        
        .metadata-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            border: 1px solid #e9ecef;
        }
        
        .status-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
            margin: 5px;
        }
        
        .badge-success {
            background: #d4edda;
            color: #155724;
        }
        
        .badge-danger {
            background: #f8d7da;
            color: #721c24;
        }
        
        .badge-info {
            background: #d1ecf1;
            color: #0c5460;
        }
        
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            transition: all 0.3s;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .divider {
            height: 2px;
            background: linear-gradient(to right, transparent, #667eea, transparent);
            margin: 30px 0;
        }
    </style>
""", unsafe_allow_html=True)


# ===============================================================
# 🧠 LOAD YOLO MODEL
# ===============================================================

@st.cache_resource
def load_yolo_model(device="cpu"):
    """
    CHỨC NĂNG (Application Layer):
    - Load mô hình YOLOv8 đã fine-tuned.
    - Hoạt động tối ưu trên CPU, phù hợp môi trường triển khai thực tế.
    """
    if os.path.exists(MODEL_PATH):
        st.success(f"✅ Đã tải mô hình tùy chỉnh: **{MODEL_PATH}**")
        model = YOLO(MODEL_PATH)
        model.to(device)
    else:
        st.warning("⚠️ Không tìm thấy best.pt → dùng YOLOv8m pretrained")
        model = YOLO("yolov8m.pt")
    return model


# ===============================================================
# 📊 ANALYSIS FUNCTIONS
# ===============================================================

def analyze_batch(detections):
    """
    Phân tích và tổng hợp thông tin từ lô trứng
    """
    total_eggs = len(detections)
    damaged_eggs = sum(1 for d in detections if d["class"] == "damaged egg")
    intact_eggs = total_eggs - damaged_eggs
    
    defect_ratio = (damaged_eggs / total_eggs * 100) if total_eggs > 0 else 0
    
    avg_confidence = np.mean([d["confidence"] for d in detections]) if detections else 0
    
    return {
        "total_eggs": total_eggs,
        "damaged_eggs": damaged_eggs,
        "intact_eggs": intact_eggs,
        "defect_ratio": defect_ratio,
        "avg_confidence": avg_confidence,
        "quality_grade": get_quality_grade(defect_ratio)
    }


def get_quality_grade(defect_ratio):
    """
    Đánh giá chất lượng lô trứng dựa trên tỷ lệ lỗi
    """
    if defect_ratio < 5:
        return "A - Xuất sắc"
    elif defect_ratio < 10:
        return "B - Tốt"
    elif defect_ratio < 20:
        return "C - Trung bình"
    else:
        return "D - Kém"


def extract_metadata(image_pil, processing_time, detections):
    """
    Trích xuất metadata chi tiết
    """
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch_id": str(uuid.uuid4())[:8].upper(),
        "device_id": platform.node(),
        "system": f"{platform.system()} {platform.release()}",
        "image_size": f"{image_pil.width} x {image_pil.height}",
        "image_format": image_pil.format if image_pil.format else "Unknown",
        "processing_time": f"{processing_time:.3f}s",
        "model": "YOLOv8 + ViT-B/16",
        "device": "GPU" if torch.cuda.is_available() else "CPU",
        "total_detections": len(detections)
    }


# ===============================================================
# 🔍 INFERENCE ENGINE – YOLO + ViT
# ===============================================================

def run_inference(model, image_np, conf_thresh=0.25, validator=None):
    """
    👉 PIPELINE CHƯƠNG 4:
    ------------------------------------------
    1. YOLO phát hiện vùng chứa trứng
    2. Cắt ROI (region of interest)
    3. ViT xác minh lại tính chính xác
    4. Kết hợp YOLO + ViT tạo kết quả cuối
    ------------------------------------------
    """

    results = model.predict(
        source=image_np,
        conf=conf_thresh,
        imgsz=640,
        verbose=False
    )

    detections = []
    drawn_img = image_np.copy()

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_raw = int(box.cls[0])

            # Class map
            if cls_raw == 0:
                yolo_label = "damaged egg"
            else:
                yolo_label = "egg"

            crop = image_np[y1:y2, x1:x2]

            # ViT Validation
            if validator:
                out = validator.verify(crop, yolo_label, conf)
                final_label = out["corrected_label"]
                reasoning = out["reason"]
                vit_conf = out["confidence"]
            else:
                final_label = yolo_label
                reasoning = "Validator disabled"
                vit_conf = conf

            if final_label == "not egg":
                continue

            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": conf,
                "class": final_label,
                "vit_confidence": vit_conf,
                "reason": reasoning,
                "area": (x2-x1) * (y2-y1),
                "center": ((x1+x2)//2, (y1+y2)//2)
            })

            color = (0, 255, 0) if final_label == "egg" else (0, 0, 255)

            cv2.rectangle(drawn_img, (x1, y1), (x2, y2), color, 3)
            
            # Vẽ label với background
            label_text = f"{final_label} {vit_conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(drawn_img, (x1, y1-30), (x1+text_w+10, y1), color, -1)
            cv2.putText(
                drawn_img,
                label_text,
                (x1+5, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Đánh số thứ tự
            cv2.circle(drawn_img, (x1+15, y1+15), 15, color, -1)
            cv2.putText(
                drawn_img,
                str(len(detections)),
                (x1+10, y1+20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

    return detections, drawn_img


# ===============================================================
# 🖥️ MAIN UI LAYER
# ===============================================================

def main_app():
    """
    ✨ CHƯƠNG 4: CÀI ĐẶT VÀ TRIỂN KHAI
    - Khởi tạo UI
    - Load model
    - Thao tác inference
    - Hiển thị kết quả
    """

    # Header
    st.markdown("""
        <div class='header-container'>
            <div class='header-title'>🥚 AI Egg Quality Inspection System</div>
            <div class='sub-header'>Advanced Detection with YOLOv8 + ViT-B/16 Validation Pipeline</div>
        </div>
    """, unsafe_allow_html=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_yolo_model(device)
    validator = VITEggValidator(device)

    # Sidebar Settings
    st.sidebar.header("⚙️ Cài đặt hệ thống")
    conf_threshold = st.sidebar.slider(
        "🎯 Ngưỡng tin cậy (Confidence)", 
        0.0, 1.0, 0.5, 0.05,
        help="Chỉ hiển thị các phát hiện có độ tin cậy cao hơn giá trị này"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("📤 Tải ảnh lên")
    uploaded_file = st.sidebar.file_uploader(
        "Chọn ảnh trứng để kiểm tra", 
        type=["jpg", "jpeg", "png"],
        help="Hỗ trợ định dạng: JPG, JPEG, PNG"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **💡 Hướng dẫn sử dụng:**
    1. Tải ảnh lên từ máy tính
    2. Điều chỉnh ngưỡng tin cậy nếu cần
    3. Nhấn "Phân tích ngay" để bắt đầu
    4. Xem kết quả chi tiết và báo cáo
    """)

    if uploaded_file:
        image_pil = Image.open(uploaded_file)
        image_np = np.array(image_pil.convert("RGB"))
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("📷 Ảnh gốc")
            st.image(image_pil, use_column_width=True, caption="Ảnh đầu vào")

        with col_right:
            if st.button("🚀 Phân tích ngay", use_container_width=True):
                start_time = datetime.now()
                
                with st.spinner("🔄 Đang xử lý bằng YOLO + ViT..."):
                    detections, drawn_img = run_inference(
                        model,
                        image_cv,
                        conf_thresh=conf_threshold,
                        validator=validator
                    )
                
                processing_time = (datetime.now() - start_time).total_seconds()
                drawn_rgb = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB)

                st.subheader("🎯 Kết quả phát hiện")
                st.image(Image.fromarray(drawn_rgb), use_column_width=True, caption="Ảnh đã được phân tích")

        # Nếu đã có kết quả
        if 'detections' in locals() and detections:
            
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            
            # Phân tích tổng quan
            analysis = analyze_batch(detections)
            metadata = extract_metadata(image_pil, processing_time, detections)
            
            # Dashboard metrics
            st.subheader("📊 Tổng quan lô trứng")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Tổng số trứng</div>
                        <div class='metric-value info-metric'>{analysis['total_eggs']}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Trứng nguyên vẹn</div>
                        <div class='metric-value success-metric'>{analysis['intact_eggs']}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Trứng nứt/hỏng</div>
                        <div class='metric-value danger-metric'>{analysis['damaged_eggs']}</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>Tỷ lệ lỗi</div>
                        <div class='metric-value warning-metric'>{analysis['defect_ratio']:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Quality Grade
            st.markdown(f"""
                <div class='info-box'>
                    <h3 style='margin:0; color:white;'>📋 Đánh giá chất lượng: {analysis['quality_grade']}</h3>
                    <p style='margin:5px 0 0 0; opacity:0.9;'>Độ tin cậy trung bình: {analysis['avg_confidence']:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Chi tiết từng trứng
            st.subheader("🔍 Chi tiết từng trứng")
            
            for idx, det in enumerate(detections, 1):
                cls = det["class"]
                css_class = "egg" if cls == "egg" else "damaged"
                badge_class = "badge-success" if cls == "egg" else "badge-danger"
                
                st.markdown(
                    f"""
                    <div class="result-card {css_class}">
                        <h4 style='margin:0 0 10px 0;'>🥚 Trứng #{idx} <span class='status-badge {badge_class}'>{det["class"].upper()}</span></h4>
                        <div style='display:grid; grid-template-columns: 1fr 1fr; gap:10px;'>
                            <div>
                                <b>📍 Vị trí:</b> ({det['center'][0]}, {det['center'][1]})<br>
                                <b>📏 Diện tích:</b> {det['area']:,} px²<br>
                                <b>🎯 YOLO Confidence:</b> {det["confidence"]:.2%}
                            </div>
                            <div>
                                <b>🤖 ViT Confidence:</b> {det["vit_confidence"]:.2%}<br>
                                <b>📦 Bounding Box:</b> {det["box"]}<br>
                                <b>💡 Giải thích:</b> {det["reason"]}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            # Export JSON
            st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
            
            export_data = {
                "metadata": metadata,
                "analysis": analysis,
                "detections": detections
            }
            
            st.download_button(
                label="📥 Tải báo cáo JSON",
                data=json.dumps(export_data, indent=2, ensure_ascii=False),
                file_name=f"egg_inspection_report_{metadata['batch_id']}.json",
                mime="application/json",
                use_container_width=True
            )

    else:
        # Welcome screen
        st.markdown("""
            <div style='text-align:center; padding:60px 20px;'>
                <h2 style='color:#667eea; margin-bottom:20px;'>👋 Chào mừng đến với hệ thống kiểm định chất lượng trứng</h2>
                <p style='font-size:18px; color:#6c757d; max-width:600px; margin:0 auto;'>
                    Vui lòng tải ảnh lên từ thanh bên trái để bắt đầu quá trình phân tích và kiểm định chất lượng trứng.
                </p>
            </div>
        """, unsafe_allow_html=True)


# ===============================================================
# 🚦 START APP
# ===============================================================
if __name__ == "__main__":
    main_app()