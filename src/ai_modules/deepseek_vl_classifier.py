import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
import json


class DeepSeekEggValidator:
    """
    Sử dụng DeepSeek-VL-7B (Vision + Chat) để xác minh trứng offline.
    """

    def __init__(self, device="cpu"):
        self.device = device

        # MODEL CHUẨN: DeepSeek tốc độ cao, có Vision
        self.model_name = "deepseek-ai/deepseek-vl-7b-chat"

        # Load tokenizer và processor
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        # Load Vision2Seq model
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(device)

    def verify_egg(self, image_np, yolo_label: str, confidence: float):
        # Convert numpy → PIL
        img = Image.fromarray(image_np)

        prompt = (
            f"YOLO dự đoán: {yolo_label} (conf={confidence:.2f}). "
            "Hãy nhìn kỹ quả trứng trong ảnh và phân loại chính xác: "
            "'egg' hoặc 'damaged egg'. "
            "Chỉ trả về JSON: "
            "{'corrected_label':'egg|damaged egg', 'confidence_reasoning':'...'}"
        )

        # Chuẩn bị input vision
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)

        # Generate
        output = self.model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False
        )

        text = self.processor.decode(output[0], skip_special_tokens=True)

        # Extract JSON nếu có
        try:
            return text[text.index("{"): text.rindex("}") + 1]
        except:
            return json.dumps({
                "corrected_label": yolo_label,
                "confidence_reasoning": "DeepSeek không trả JSON chuẩn"
            })
