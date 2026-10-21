"""Frozen MarineInst Stage 1 ViT-H encoder for SeaDino's existing probes."""
from functools import partial
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from eval_suite.vendor.marineinst.image_encoder import ImageEncoderViT


MARINEINST_REVISION = "62a5cba254b52339ac93cfd0293b8b6bc820b46b"
MARINEINST_FILENAME = "marineinst_vit-h_stage1.pth"
MARINEINST_URL = (
    "https://www.dropbox.com/scl/fi/nixpd3w67y4jy4r8xbp1m/marineinst_vit-h_stage1.pth"
    "?dl=1&rlkey=onkpaobmq0jtv8nycmiuwnvyl&st=etxnqesp"
)


class MarineInstBackboneWrapper(nn.Module):
    """Mean-pooled pre-neck features plus the spatial ViT patch grid.

Absolute position embeddings are bicubically interpolated for the benchmark
resolution. At the encoder's native resolution, its original positions are used
unchanged. Native relative-position interpolation remains in the upstream code.
"""

    def __init__(self, encoder, microbatch_size=4, identity=None):
        super().__init__()
        if microbatch_size < 1:
            raise ValueError("MarineInst microbatch size must be positive")
        self.raw_model = encoder.eval().requires_grad_(False)
        self.hidden_size = encoder.patch_embed.proj.out_channels
        self.patch_size = encoder.patch_embed.proj.kernel_size[0]
        self.config = SimpleNamespace(hidden_size=self.hidden_size, patch_size=self.patch_size,
                                      _commit_hash=MARINEINST_REVISION)
        self.microbatch_size = microbatch_size
        self.model_type = "marineinst-encoder"
        self.norm_mean = [.485, .456, .406]
        self.norm_std = [.229, .224, .225]
        self.norm_label = "MarineInst/ImageNet"
        vendor = Path(__file__).parent / "vendor" / "marineinst"
        self.feature_cache_source_files = tuple(str(p) for p in
            [Path(__file__).resolve(), vendor / "image_encoder.py", vendor / "common.py"])
        self.benchmark_identity = dict(
            extractor="marineinst-vit-h-pre-neck-mean-pool-v1", microbatch_size=microbatch_size,
            position_resize="bicubic-align_corners_false", code_revision=MARINEINST_REVISION,
            possible_pretraining_overlap=["brackish", "deepfish", "fathomnet"],
            **(identity or {}),
        )

    @torch.no_grad()
    def forward(self, pixel_values, *args, **kwargs):
        if args:
            raise TypeError("MarineInst frozen-feature adapter accepts image tensors only")
        if kwargs.get("output_attentions") or kwargs.get("output_hidden_states"):
            raise NotImplementedError("MarineInst adapter exposes final features, not CLS attention rollout")
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("MarineInst expects RGB images (N, 3, H, W)")
        batch, _, height, width = pixel_values.shape
        if not batch or height % self.patch_size or width % self.patch_size:
            raise ValueError("MarineInst image dimensions must be positive multiples of patch size")
        self.raw_model.eval()
        outputs = []
        for images in pixel_values.split(self.microbatch_size):
            features = self.raw_model.patch_embed(images)
            position = self.raw_model.pos_embed
            if position is not None:
                if position.shape[1:3] != features.shape[1:3]:
                    position = F.interpolate(position.permute(0, 3, 1, 2).float(),
                        size=features.shape[1:3], mode="bicubic", align_corners=False).permute(0, 2, 3, 1)
                features = features + position.to(features.dtype)
            for block in self.raw_model.blocks:
                features = block(features)
            patches = features.reshape(len(images), -1, self.hidden_size)
            pooled = patches.float().mean(1, keepdim=True).to(patches.dtype)
            outputs.append(torch.cat((pooled, patches), dim=1))
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(last_hidden_state=torch.cat(outputs, dim=0))


def load_marineinst_backbone(device, dtype, *, checkpoint_path=None, microbatch_size=4):
    checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path else (
        Path.home() / ".cache" / "seadino" / "checkpoints" / MARINEINST_FILENAME
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"MarineInst checkpoint missing: {checkpoint}. Download the official Stage 1 model "
            f"from {MARINEINST_URL} and pass --marineinst_checkpoint /path/to/{MARINEINST_FILENAME}"
        )
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    # The release is a tensor state dictionary. Never fall back to arbitrary
    # pickle execution if a supplied file is not compatible with weights_only.
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, dict):
        raise ValueError("MarineInst checkpoint must contain a state dictionary")
    for key in ("state_dict", "model"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    weights = {key.removeprefix("image_encoder."): value for key, value in state.items()
               if key.startswith("image_encoder.")}
    if not weights:
        raise ValueError("No MarineInst image_encoder weights found")
    with torch.device("meta"):
        encoder = ImageEncoderViT(img_size=1024, patch_size=16, embed_dim=1280,
            depth=32, num_heads=16, mlp_ratio=4, out_chans=256, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), use_rel_pos=True,
            global_attn_indexes=[7, 15, 23, 31], window_size=14)
    encoder.load_state_dict(weights, strict=True, assign=True)
    wrapped = MarineInstBackboneWrapper(encoder, microbatch_size=microbatch_size, identity={
        "checkpoint_path": str(checkpoint.resolve()), "checkpoint_sha256": digest.hexdigest(),
    }).to(device=device, dtype=dtype).eval()
    print("  [MarineInst] Stage 1 ViT-H; frozen 1280-dimensional pre-neck patch features.")
    print("  [MarineInst] Resolution-matched probes; native prompted model uses 1024px.")
    print("  [MarineInst] Pretraining sources include Brackish, DeepFish and FathomNet; overlap is unresolved.")
    return wrapped, None, None, False
