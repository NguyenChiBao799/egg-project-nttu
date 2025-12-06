import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
import json
import numpy as np


class QwenVLEggValidator:
    """
    Sử dụng Qwen2-VL-2B-Instruct đúng chuẩn multimodal model.
    """

    def __init__(self, device="cpu"):
        self.device = device

        self.model_name = "Qwen/Qwen2-VL-2B-Instruct"

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load model đúng class
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(device)

    def verify_egg(self, image_np, yolo_label: str, confidence: float):
        img = Image.fromarray(image_np)

        prompt = (
            f"YOLO dự đoán: {yolo_label} (conf={confidence:.2f}). "
            "Hãy xem kỹ ảnh trứng và phân loại chính xác: 'egg' hoặc 'damaged egg'. "
            "Chỉ trả về JSON dạng: "
            "{'corrected_label':'egg|damaged egg', 'confidence_reasoning':'...'}"
        )

        # Chuẩn hóa input
        inputs = self.processor(
            text=prompt,
            images=img,
            return_tensors="pt"
        ).to(self.device)

        # Generate
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=128
        )

        text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Extract JSON
        try:
            return text[text.index("{"): text.rindex("}") + 1]
        except:
            return json.dumps({
                "corrected_label": yolo_label,
                "confidence_reasoning": "Qwen2 trả về không đúng JSON"
            })
