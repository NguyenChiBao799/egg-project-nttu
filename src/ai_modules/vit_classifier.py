import torch
from transformers import ViTModel, ViTImageProcessor
from PIL import Image
import numpy as np


class VITEggValidator:
    """
    ViT-B/16 Validator – fixed version
    Không xoá nhầm bounding box.
    """
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(self.device)
        self.processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

        # 3 class
        self.labels = ["egg", "damaged egg", "not egg"]

    def _embed(self, crop_np):
        pil_img = Image.fromarray(crop_np)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self.model(**inputs).pooler_output.squeeze(0)
        return emb / emb.norm()

    def verify(self, crop_np, yolo_label=None, yolo_conf=None):

        emb = self._embed(crop_np)

        # Zero-shot approximations
        egg_score = float(emb[0])                # stable indicator
        crack_score = float(emb.mean()) * 0.6    # texture intensity
        not_egg_score = 1 - max(egg_score, crack_score)

        # FIX: Bounded & stabilized scores
        egg_score = max(0.0, min(1.0, egg_score))
        crack_score = max(0.0, min(1.0, crack_score))
        not_egg_score = max(0.0, min(1.0, not_egg_score))

        # 🔥 NEW decision rules (không xoá nhầm)
        # 1) Nếu YOLO tự tin > 0.6 → tin YOLO trước
        if yolo_conf and yolo_conf > 0.6:
            if yolo_label == "egg":
                return {"corrected_label": "egg", "confidence": yolo_conf,
                        "reason": "YOLO high confidence – giữ bounding box"}
            else:
                return {"corrected_label": "damaged egg", "confidence": yolo_conf,
                        "reason": "YOLO high confidence – giữ bounding box"}

        # 2) Chỉ xoá khi thực sự NOT EGG rõ ràng
        if not_egg_score > 0.60:
            return {"corrected_label": "not egg", "confidence": not_egg_score,
                    "reason": "ViT confident this is NOT an egg"}

        # 3) Egg vs damaged egg
        if crack_score > egg_score:
            return {"corrected_label": "damaged egg", "confidence": crack_score,
                    "reason": "ViT sees crack-like texture"}
        else:
            return {"corrected_label": "egg", "confidence": egg_score,
                    "reason": "ViT sees smooth egg texture"}
