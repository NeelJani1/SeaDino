"""Model loaders, baseline adapters, and linear probe segmentation heads."""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

BASELINE_MAP = {
    "sam3.1": "facebook/sam3.1",
    "sam3.1-encoder": "facebook/sam3.1",
    "facebook/sam3.1": "facebook/sam3.1",
    "marineinst": "marineinst-vit-h-stage1",
    "marineinst-encoder": "marineinst-vit-h-stage1",
    "marineinst-vit-h-stage1": "marineinst-vit-h-stage1",
    "sam3": "facebook/sam3",
    "sam3-encoder": "facebook/sam3",
    "facebook/sam3": "facebook/sam3",
    "bioclip": "imageomics/bioclip",
    "imageomics/bioclip": "imageomics/bioclip",
    "dinov2-small": "facebook/dinov2-small",
    "dinov2_small": "facebook/dinov2-small",
    "dinov2-s": "facebook/dinov2-small",
    "facebook/dinov2-small": "facebook/dinov2-small",
    "dinov2-base": "facebook/dinov2-base",
    "dinov2_base": "facebook/dinov2-base",
    "dinov2-b": "facebook/dinov2-base",
    "facebook/dinov2-base": "facebook/dinov2-base",
    "clip": "openai/clip-vit-base-patch16",
    "openai-clip": "openai/clip-vit-base-patch16",
    "openai/clip-vit-base-patch16": "openai/clip-vit-base-patch16",
}

# Standard normalization statistics
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class VisionBackboneWrapper(nn.Module):
    """Unified wrapper around ViT backbones (DINOv3, DINOv2, CLIP, BioCLIP)
    providing standard output interfaces for classification probes, attention rollout,
    and dense segmentation.
    """
    def __init__(
        self,
        raw_model: nn.Module,
        hidden_size: int,
        patch_size: int = 16,
        model_type: str = "generic",
        norm_mean: Optional[list] = None,
        norm_std: Optional[list] = None,
    ):
        super().__init__()
        self.raw_model = raw_model
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.model_type = model_type
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        class ConfigProxy:
            pass
        self.config = getattr(raw_model, "config", ConfigProxy())
        self.config.hidden_size = hidden_size
        self.config.patch_size = patch_size

    def forward(self, *args, **kwargs):
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None and len(args) > 0:
            pixel_values = args[0]
            extra_args = args[1:]
        else:
            extra_args = args

        output_attentions = kwargs.pop("output_attentions", False)
        output_hidden_states = kwargs.pop("output_hidden_states", False)
        interpolate_pos = kwargs.pop("interpolate_pos_encoding", True)

        call_kwargs = {}
        if output_attentions:
            call_kwargs["output_attentions"] = True
        if output_hidden_states:
            call_kwargs["output_hidden_states"] = True

        if self.model_type in ("clip", "bioclip"):
            model_target = getattr(self.raw_model, "vision_model", self.raw_model)
            try:
                return model_target(pixel_values=pixel_values, interpolate_pos_encoding=interpolate_pos, **call_kwargs)
            except TypeError:
                return model_target(pixel_values=pixel_values, **call_kwargs)

        elif self.model_type == "dinov2":
            return self.raw_model(pixel_values=pixel_values, **call_kwargs)

        else:
            try:
                return self.raw_model(pixel_values=pixel_values, interpolate_pos_encoding=interpolate_pos, **call_kwargs)
            except TypeError:
                return self.raw_model(pixel_values=pixel_values, **call_kwargs)


class OpenCLIPVisionWrapper(nn.Module):
    """Wraps OpenCLIP VisionTransformer to conform to standard Hugging Face
    last_hidden_state interface for dual probes and visualization.
    """
    def __init__(self, visual, hidden_size: int = 768, patch_size: int = 16):
        super().__init__()
        self.visual = visual
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.model_type = "bioclip"
        self.norm_mean = CLIP_MEAN
        self.norm_std = CLIP_STD

        class ConfigProxy:
            pass
        self.config = ConfigProxy()
        self.config.hidden_size = hidden_size
        self.config.patch_size = patch_size

    def forward(self, pixel_values, *args, **kwargs):
        v = self.visual
        x = v.conv1(pixel_values)
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)
        cls_tok = v.class_embedding.to(x.dtype) + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_tok, x], dim=1)

        pos_emb = v.positional_embedding.to(x.dtype)
        if x.shape[1] != pos_emb.shape[0]:
            import math
            cls_pos = pos_emb[:1]
            patch_pos = pos_emb[1:]
            num_orig = int(math.isqrt(patch_pos.shape[0]))
            patch_pos = patch_pos.reshape(1, num_orig, num_orig, -1).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(H, W), mode="bicubic", align_corners=False)
            patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(-1, pos_emb.shape[-1])
            pos_emb = torch.cat([cls_pos, patch_pos], dim=0)

        x = x + pos_emb
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = v.transformer(x)
        x = x.permute(1, 0, 2)
        if hasattr(v, "ln_post"):
            tokens = v.ln_post(x)
        else:
            tokens = x

        class Output:
            def __init__(self, lhs):
                self.last_hidden_state = lhs
        return Output(tokens)


def load_dinov3_backbone(
    checkpoint_path: Optional[str],
    model_id: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str = "sdpa",
):
    """Loads DINOv3 backbone from a .ckpt (EMA teacher) or raw Hugging Face baseline."""
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id, attn_implementation=attn_implementation, dtype=dtype)
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


def load_baseline_backbone(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str = "sdpa",
    *, sam3_revision=None, sam3_model_path=None, sam3_microbatch_size=2,
    marineinst_checkpoint=None, marineinst_microbatch_size=4,
):
    """Loads comparative frozen image-encoder backbones, including SAM 3."""
    canonical_id = BASELINE_MAP.get(model_name.lower().strip(), model_name)
    from transformers import AutoModel, CLIPVisionModel
    print(f"  [Baseline Loader] Loading foundation baseline '{canonical_id}'...")

    if canonical_id == "marineinst-vit-h-stage1":
        from eval_suite.marineinst_backbone import load_marineinst_backbone
        return load_marineinst_backbone(device, dtype, checkpoint_path=marineinst_checkpoint,
                                       microbatch_size=marineinst_microbatch_size)

    if canonical_id in ("facebook/sam3", "facebook/sam3.1"):
        from eval_suite.sam3_backbone import load_sam3_backbone
        return load_sam3_backbone(device, dtype, model_id=canonical_id, revision=sam3_revision,
                                 model_path=sam3_model_path, microbatch_size=sam3_microbatch_size,
                                 attn_implementation=attn_implementation)

    if "bioclip" in canonical_id.lower():
        try:
            import open_clip
            oc_model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
            wrapped = OpenCLIPVisionWrapper(oc_model.visual, hidden_size=768, patch_size=16)
        except ImportError:
            raise ImportError(
                "BioCLIP ('imageomics/bioclip') is hosted as an OpenCLIP architecture and requires 'open_clip_torch'. "
                "Please install it via: pip install open_clip_torch"
            )
        except Exception:
            try:
                raw_model = CLIPVisionModel.from_pretrained(canonical_id, dtype=dtype)
            except Exception:
                raw_model = AutoModel.from_pretrained(canonical_id, dtype=dtype)
            vis = getattr(raw_model, "vision_model", raw_model)
            hidden_size = getattr(vis.config, "hidden_size", 768)
            patch_size = getattr(vis.config, "patch_size", 16)
            wrapped = VisionBackboneWrapper(raw_model, hidden_size=hidden_size, patch_size=patch_size, model_type="bioclip")

    elif "clip" in canonical_id.lower() and "dinov2" not in canonical_id.lower():
        raw_model = CLIPVisionModel.from_pretrained(canonical_id, dtype=dtype)
        hidden_size = getattr(raw_model.config, "hidden_size", 768)
        patch_size = getattr(raw_model.config, "patch_size", 16)
        wrapped = VisionBackboneWrapper(
            raw_model, hidden_size=hidden_size, patch_size=patch_size,
            model_type="clip", norm_mean=CLIP_MEAN, norm_std=CLIP_STD
        )

    elif "dinov2" in canonical_id.lower():
        try:
            raw_model = AutoModel.from_pretrained(canonical_id, attn_implementation=attn_implementation, dtype=dtype)
        except Exception:
            raw_model = AutoModel.from_pretrained(canonical_id, dtype=dtype)
        hidden_size = getattr(raw_model.config, "hidden_size", 384 if "small" in canonical_id else 768)
        patch_size = getattr(raw_model.config, "patch_size", 14)
        wrapped = VisionBackboneWrapper(raw_model, hidden_size=hidden_size, patch_size=patch_size, model_type="dinov2")

    else:
        raw_model = AutoModel.from_pretrained(canonical_id, attn_implementation=attn_implementation, dtype=dtype)
        hidden_size = getattr(raw_model.config, "hidden_size", 768)
        patch_size = getattr(raw_model.config, "patch_size", 16)
        wrapped = VisionBackboneWrapper(raw_model, hidden_size=hidden_size, patch_size=patch_size, model_type="generic")

    wrapped = wrapped.to(device=device, dtype=dtype)
    wrapped.eval()
    return wrapped, None, None, False


class LinearSegmenter(nn.Module):
    """Frozen ViT backbone with a 1x1 Conv linear segmentation head."""
    def __init__(self, backbone, feat_dim: int, num_classes: int, patch_size: Optional[int] = None):
        super().__init__()
        self.backbone = backbone
        if patch_size is None:
            patch_size = getattr(getattr(backbone, "config", None), "patch_size", 16)
        self.patch_size = patch_size
        self.head = nn.Conv2d(feat_dim, num_classes, kernel_size=1)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def forward(self, x: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
        B, C, H, W = x.shape
        pad_h = (self.patch_size - (H % self.patch_size)) % self.patch_size
        pad_w = (self.patch_size - (W % self.patch_size)) % self.patch_size
        x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h > 0 or pad_w > 0) else x

        grid_h = x_in.shape[2] // self.patch_size
        grid_w = x_in.shape[3] // self.patch_size
        num_patches = grid_h * grid_w

        with torch.no_grad():
            outputs = self.backbone(x_in)
            if hasattr(outputs, "last_hidden_state"):
                tokens = outputs.last_hidden_state
            elif isinstance(outputs, dict) and "last_hidden_state" in outputs:
                tokens = outputs["last_hidden_state"]
            else:
                tokens = outputs[0]

            # Extract only spatial patch tokens (skipping [CLS] and register tokens)
            patch_tokens = tokens[:, -num_patches:, :]
            feat_map = patch_tokens.transpose(1, 2).reshape(B, -1, grid_h, grid_w)

        logits = self.head(feat_map.float())
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
