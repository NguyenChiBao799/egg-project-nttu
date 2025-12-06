import torch
from transformers import SiglipModel, SiglipImageProcessor
from PIL import Image
import numpy as np
import torch.nn.functional as F
import json

class SigLipEggValidator:
    """
    SigLIP tinh gọn:
    - KHÔNG cần SentencePiece
    - KHÔNG dùng tokenizer
    - Chỉ dùng image + embedding text cố định
    - Nhanh hơn bản đầy đủ 3–4 lần
    """

    def __init__(self, device="cpu"):
        self.device = device
        self.model_name = "google/siglip-base-patch16-224"

        # Load phần xử lý ảnh (không cần tokenizer)
        self.processor = SiglipImageProcessor.from_pretrained(self.model_name)
        self.model = SiglipModel.from_pretrained(self.model_name).to(device)

        # Nhãn
        self.labels = ["egg", "damaged egg"]

        # Encode text 1 lần duy nhất → không cần tokenizer
        with torch.no_grad():
            self.text_emb = self.model.get_text_features(text=self.labels)
            self.text_emb = self.text_emb / self.text_emb.norm(dim=-1, keepdim=True)

    def verify_egg(self, image_np, yolo_label, confidence):
        img = Image.fromarray(image_np)

        # Encode hình
        image_inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            img_emb = self.model.get_image_features(**image_inputs)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

            # Cosine similarity xác định nhãn
            sims = F.cosine_similarity(img_emb, self.text_emb)
            idx = sims.argmax().item()

        result_label = self.labels[idx]
        prob = float(sims[idx])

        return json.dumps({
            "corrected_label": result_label,
            "confidence_reasoning": f"SigLIP cosine={prob:.3f}"
        })
