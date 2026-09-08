"""Model loaders and linear probe segmentation heads."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def load_dinov3_backbone(checkpoint_path: Optional[str], model_id: str, device: torch.device, dtype: torch.dtype):
    """Loads DINOv3 backbone from a .ckpt (EMA teacher) or raw Hugging Face baseline."""
    model = AutoModel.from_pretrained(model_id, attn_implementation="sdpa", dtype=dtype)
    epoch, train_loss = None, None
    benthic_norm = False

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("teacher_state_dict", ckpt.get("teacher", ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))))
        cleaned_sd = {k.replace("_orig_mod.", "").replace("backbone.", "").replace("teacher.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(cleaned_sd, strict=False)
        epoch = ckpt.get("epoch", None)
        train_loss = ckpt.get("loss", ckpt.get("ssl_loss", None))

        cfg = ckpt.get("config", {})
        if isinstance(cfg, dict):
            benthic_norm = cfg.get("benthic_norm", False)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model, epoch, train_loss, benthic_norm


class LinearSegmenter(nn.Module):
    """Frozen ViT backbone with a 1x1 Conv linear segmentation head."""
    def __init__(self, backbone, feat_dim: int, num_classes: int, patch_size: int = 16):
        super().__init__()
        self.backbone = backbone
        self.patch_size = patch_size
        self.head = nn.Conv2d(feat_dim, num_classes, kernel_size=1)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def forward(self, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        B, C, H, W = x.shape
        grid_h, grid_w = H // self.patch_size, W // self.patch_size
        num_patches = grid_h * grid_w

        outputs = self.backbone(x)
        tokens = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]

        # Extract only spatial patch tokens (skipping [CLS] and 4 register tokens)
        patch_tokens = tokens[:, -num_patches:, :]
        feat_map = patch_tokens.transpose(1, 2).reshape(B, -1, grid_h, grid_w)

        logits = self.head(feat_map.float())
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
