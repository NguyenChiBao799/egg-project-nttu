import torch
import pickle


# ============================================================
# ⚡ PATCH: EP torch.load về chế độ cũ (full pickle)
# ============================================================
def patched_torch_load(*args, **kwargs):
    # Gỡ weights_only nếu YOLO chỉnh
    kwargs["weights_only"] = False
    return original_torch_load(*args, **kwargs)


original_torch_load = torch.load
torch.load = patched_torch_load

print("⚡ [TorchPatch] torch.load patched (weights_only=False enforced)")