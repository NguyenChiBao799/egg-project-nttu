import base64
import google.generativeai as genai
from PIL import Image
import io
import json


class GeminiEggValidator:
    """
    Bộ xác minh trứng dùng Gemini.
    Hỗ trợ mọi API key (dùng gemini-pro-vision).
    """

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)

        # Bản ổn định nhất, AI Studio nào cũng chạy:
        self.model = genai.GenerativeModel("gemini-pro-vision")

    def encode_image(self, image_np):
        """Convert numpy → JPEG base64."""
        pil_img = Image.fromarray(image_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    def verify_egg(self, image_np, yolo_label: str):
        """
        Xác minh lại YOLO bằng Gemini.

        Trả về JSON dạng:
        {
            "corrected_label": "egg" | "damaged egg",
            "confidence_reasoning": "..."
        }
        """

        img_b64 = self.encode_image(image_np)

        prompt = f"""
        Bạn là chuyên gia kiểm định chất lượng trứng.

        YOLO dự đoán: {yolo_label}

        Hãy phân loại lại ảnh trứng trong hình này theo 2 nhãn:
        - "egg" (trứng tốt)
        - "damaged egg" (trứng bị nứt/vỡ)

        Trả về duy nhất JSON với 2 trường:
        {{
            "corrected_label": "egg" hoặc "damaged egg",
            "confidence_reasoning": "giải thích ngắn gọn"
        }}
        """

        try:
            response = self.model.generate_content(
                [
                    {"mime_type": "image/jpeg", "data": img_b64},
                    prompt
                ]
            )
            return response.text

        except Exception as e:
            return json.dumps({
                "corrected_label": yolo_label,
                "confidence_reasoning": f"Gemini lỗi: {str(e)}"
            })
