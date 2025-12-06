import base64
import json
from groq import Groq
from PIL import Image
import io


class GroqEggValidator:
    """
    Xác minh YOLO bằng Groq Vision (không cần URL, chỉ dùng base64).
    """

    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model_name = "llama-3.2-90b-vision"

    def encode_image(self, image_np):
        """Chuyển numpy → base64."""
        pil_img = Image.fromarray(image_np)
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def verify_egg(self, image_np, yolo_label: str, yolo_conf: float):
        img_b64 = self.encode_image(image_np)

        # CHUẨN FORMAT GROQ: text + image_url
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"YOLO dự đoán: {yolo_label} (confidence={yolo_conf:.2f}). "
                            "Hãy nhìn ảnh này và phân loại lại: 'egg' hoặc 'damaged egg'. "
                            "Trả về JSON với 2 trường: corrected_label, confidence_reasoning."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{img_b64}"
                    }
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=200,
            )
            return response.choices[0].message.content

        except Exception as e:
            return json.dumps({
                "corrected_label": yolo_label,
                "confidence_reasoning": f"Groq Vision lỗi: {e}"
            })
