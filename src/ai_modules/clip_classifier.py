import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import json


class CLIPEggValidator:
    """
    Phân loại egg / damaged egg bằng CLIP ViT-B/32.
    - Rất nhẹ
    - Rất nhanh
    - Không cần sentencepiece
    - Chạy tốt trên Windows + Python 3.12
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.model_name = "openai/clip-vit-base-patch32"

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(device)

        self.labels = ["egg", "damaged egg"]

    def verify_egg(self, image_np, yolo_label, confidence):
        img = Image.fromarray(image_np)

        # chuẩn bị input
        inputs = self.processor(
            text=self.labels,
            images=img,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # forward
        outputs = self.model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1)[0]

        idx = probs.argmax().item()
        label = self.labels[idx]

        return json.dumps({
            "corrected_label": label,
            "confidence_reasoning": f"CLIP confidence={probs[idx].item():.3f}"
        })
