"""SAM 3 image-encoder adapter for frozen-feature benchmarks.

This deliberately evaluates the image encoder with the suite's learned probes,
not SAM 3's text/box/point-prompted segmentation decoder. See docs/sam3-baseline.md.
"""
import hashlib
from pathlib import Path
import re

import torch
from torch import nn


SAM3_MODEL_ID = "facebook/sam3"
SAM3_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
SAM31_MODEL_ID = "facebook/sam3.1"
SAM31_REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
SAM3_CACHE_DIR = Path.home() / ".cache" / "seadino" / "huggingface"
SAM3_MEAN = [0.5, 0.5, 0.5]
SAM3_STD = [0.5, 0.5, 0.5]


class SAM3BackboneWrapper(nn.Module):
    """Expose mean-pooled global features followed by unmodified patch tokens.

The first output token is a pooling adapter for the existing probe interface;
it is not a learned SAM CLS token. Only the following spatial tokens enter the
segmentation head. Encoder microbatches do not change probe optimizer batches.
"""

    def __init__(self, backbone, microbatch_size=2, identity=None):
        super().__init__()
        if microbatch_size < 1:
            raise ValueError("SAM 3 microbatch size must be positive")
        self.raw_model = backbone.eval().requires_grad_(False)
        self.config = backbone.config
        self.hidden_size = self.config.hidden_size
        self.patch_size = self.config.patch_size
        if not isinstance(self.patch_size, int):
            raise ValueError("SAM 3 adapter requires a scalar patch size")
        self.microbatch_size = microbatch_size
        self.model_type = "sam3-encoder"
        self.norm_mean, self.norm_std = SAM3_MEAN, SAM3_STD
        self.norm_label = "SAM3"
        self.feature_cache_source_files = (str(Path(__file__).resolve()),)
        self.benchmark_identity = dict(
            extractor="sam3-vit-patches-mean-pool-v2", microbatch_size=microbatch_size,
            rotary_layout="row-major-x-mod-width-y-div-width; isotropic-window-over-height",
            **(identity or {}),
        )
        self._rotary_grid = None

    def _set_rotary_grid(self, height, width, device):
        # Transformers 5.3 builds fixed RoPE buffers at construction. Rebuild
        # only those buffers. For NHWC row-major patch tokens, x wraps at the
        # actual grid WIDTH. HF 5.3's rectangular constructor passes height as
        # end_x, which only has the correct stride for square inputs. Preserve
        # the existing isotropic scale and the exact square-input behavior.
        # Absolute position tiling stays inside the native model.
        grid = (height // self.patch_size, width // self.patch_size, str(device))
        if grid == self._rotary_grid:
            return
        from transformers.models.sam3.modeling_sam3 import Sam3ViTRotaryEmbedding

        for layer in self.raw_model.layers:
            h, w = (layer.window_size, layer.window_size) if layer.window_size else grid[:2]
            layer.rotary_emb = Sam3ViTRotaryEmbedding(
                self.config, end_x=w, end_y=h, scale=self.config.window_size / h,
            ).to(device=device)
        self._rotary_grid = grid

    @torch.no_grad()
    def forward(self, pixel_values, *args, **kwargs):
        if args:
            raise TypeError("SAM 3 frozen-feature adapter accepts image tensors only")
        if kwargs.get("output_attentions") or kwargs.get("output_hidden_states"):
            raise NotImplementedError("SAM 3 adapter exposes final features, not CLS attention rollout")
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("SAM 3 expects a batch of RGB images (N, 3, H, W)")
        batch, _, height, width = pixel_values.shape
        if not batch or height % self.patch_size or width % self.patch_size:
            raise ValueError("SAM 3 input height and width must be positive multiples of its patch size")
        self.raw_model.eval()
        self._set_rotary_grid(height, width, pixel_values.device)
        chunks = []
        for images in pixel_values.split(self.microbatch_size):
            patches = self.raw_model(pixel_values=images).last_hidden_state
            expected = (height // self.patch_size) * (width // self.patch_size)
            if patches.shape[1:] != (expected, self.hidden_size):
                raise RuntimeError(f"Unexpected SAM 3 patch layout: {tuple(patches.shape)}")
            pooled = patches.float().mean(dim=1, keepdim=True).to(patches.dtype)
            chunks.append(torch.cat((pooled, patches), dim=1))
        from transformers.modeling_outputs import BaseModelOutput
        return BaseModelOutput(last_hidden_state=torch.cat(chunks, dim=0))


def _vision_config(config):
    # Official HF SAM 3 currently packages the image detector and video tracker.
    # Older image-only and local vision-only exports are also supported.
    if hasattr(config, "detector_config"):
        config = config.detector_config
    if hasattr(config, "vision_config"):
        config = config.vision_config
    if hasattr(config, "backbone_config"):
        config = config.backbone_config
    if getattr(config, "model_type", None) != "sam3_vit_model":
        raise ValueError(f"Expected SAM 3 ViT config, received {getattr(config, 'model_type', None)!r}")
    return config


def _weights_digest(weights):
    digest = hashlib.sha256()
    for key, tensor in sorted(weights.items()):
        digest.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}".encode())
        digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _convert_native_encoder_weights(state, config):
    """Map the official native ViT trunk exactly; reject unknown learned tensors."""
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    prefix = "detector.backbone.vision_backbone.trunk."
    native = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
    if not native:
        raise ValueError("No official SAM 3 detector ViT trunk found in native checkpoint")
    converted = {}
    for key, value in native.items():
        if key == "pos_embed":
            expected = (config.pretrain_image_size // config.patch_size) ** 2
            if value.shape != (1, expected + 1, config.hidden_size):
                raise ValueError(f"Unexpected native CLS position layout: {tuple(value.shape)}")
            converted["embeddings.position_embeddings"] = value[:, 1:].contiguous()
        elif key == "patch_embed.proj.weight":
            converted["embeddings.patch_embeddings.projection.weight"] = value
        elif key.startswith("ln_pre."):
            converted["layer_norm." + key.split(".")[-1]] = value
        else:
            match = re.fullmatch(r"blocks\.(\d+)\.(.+)", key)
            if not match:
                raise ValueError(f"Unknown native SAM 3 encoder tensor: {key}")
            index, suffix = int(match[1]), match[2]
            if not 0 <= index < config.num_hidden_layers:
                raise ValueError(f"Native encoder layer out of range: {index}")
            target = f"layers.{index}."
            if suffix == "attn.freqs_cis":
                # Serialized trigonometric buffers are not learned weights.
                # Check geometry against the native configuration before using
                # the installed model's equivalent regenerated rotary buffers.
                side = config.image_size // config.patch_size if index in config.global_attn_indexes else config.window_size
                if value.shape != (side * side, config.hidden_size // config.num_attention_heads // 2):
                    raise ValueError(f"Unexpected native RoPE layout for {key}: {tuple(value.shape)}")
                if not torch.is_complex(value) or not torch.isfinite(value).all():
                    raise ValueError(f"Invalid native RoPE buffer: {key}")
                continue
            if suffix in ("attn.qkv.weight", "attn.qkv.bias"):
                if value.shape[0] != 3 * config.hidden_size:
                    raise ValueError(f"Unexpected native QKV tensor: {key}")
                for name, tensor in zip(("q", "k", "v"), value.chunk(3, dim=0)):
                    converted[target + f"attention.{name}_proj." + suffix.split(".")[-1]] = tensor.contiguous()
            else:
                replacements = {"norm1.": "layer_norm1.", "norm2.": "layer_norm2.",
                                "attn.proj.": "attention.o_proj.", "mlp.fc1.": "mlp.fc1.",
                                "mlp.fc2.": "mlp.fc2."}
                replacement = next(((old, new) for old, new in replacements.items() if suffix.startswith(old)), None)
                if replacement is None:
                    raise ValueError(f"Unknown native SAM 3 encoder tensor: {key}")
                old, new = replacement
                converted[target + new + suffix[len(old):]] = value
    return converted


def _read_encoder_weights(filename, config=None):
    """Read only image-encoder tensors, and reject unrecognized checkpoint layouts."""
    if Path(filename).suffix == ".pt":
        if config is None:
            raise ValueError("Native SAM 3 checkpoints require their ViT configuration")
        state = torch.load(filename, map_location="cpu", weights_only=True, mmap=True)
        weights = _convert_native_encoder_weights(state, config)
        return weights, _weights_digest(weights)
    from safetensors import safe_open

    prefixes = ("detector_model.vision_encoder.backbone.", "vision_encoder.backbone.", "backbone.", "")
    digest = hashlib.sha256()
    with safe_open(str(filename), framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        candidates = [prefix for prefix in prefixes if prefix + "embeddings.position_embeddings" in keys]
        if len(candidates) != 1:
            raise ValueError("Cannot uniquely identify SAM 3 image-encoder weights in this safetensors file")
        prefix = candidates[0]
        weights = {}
        for full_key in sorted(keys):
            if not full_key.startswith(prefix):
                continue
            key = full_key[len(prefix):]
            tensor = checkpoint.get_tensor(full_key)
            weights[key] = tensor
            digest.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}".encode())
            digest.update(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return weights, digest.hexdigest()


def load_sam3_backbone(device, dtype, *, model_id=SAM3_MODEL_ID, revision=None, model_path=None,
                       microbatch_size=2, attn_implementation="sdpa"):
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, Sam3ViTModel

    if model_id not in (SAM3_MODEL_ID, SAM31_MODEL_ID):
        raise ValueError(f"Unsupported SAM release: {model_id}")
    revision = revision or (SAM31_REVISION if model_id == SAM31_MODEL_ID else SAM3_REVISION)
    release = "SAM3.1" if model_id == SAM31_MODEL_ID else "SAM3"
    source = str(Path(model_path).expanduser().resolve()) if model_path else model_id
    kwargs = {} if model_path else {"revision": revision, "cache_dir": str(SAM3_CACHE_DIR)}
    weight_name = "sam3.1_multiplex.pt" if model_id == SAM31_MODEL_ID else "model.safetensors"
    try:
        config = AutoConfig.from_pretrained(source, **kwargs)
        filename = Path(source) / weight_name if model_path else Path(
            hf_hub_download(source, weight_name, revision=revision, cache_dir=str(SAM3_CACHE_DIR))
        )
    except OSError as exc:
        raise RuntimeError(
            f"Could not load official {release} weights. Confirm access at "
            f"https://huggingface.co/{model_id} and authenticate locally with hf auth login, "
            "or pass --sam3_model_path pointing to an authorized Hugging Face-format local export. "
            f"Underlying error: {exc}"
        ) from exc
    resolved_revision = getattr(config, "_commit_hash", None) or ("" if model_path else revision)
    vision_config = _vision_config(config)
    if filename.suffix == ".pt":
        # The native Meta factory constructs torch.nn.LayerNorm with eps=1e-5;
        # the repository's reused HF config states1e-6. Follow the native factory
        # for native checkpoint conversion and include the choice in identity.
        vision_config.layer_norm_eps = 1e-5
    vision_config._attn_implementation = attn_implementation
    vision_config._commit_hash = resolved_revision
    weights, weights_sha256 = _read_encoder_weights(filename, vision_config)
    # Meta initialization avoids holding a second 450M-parameter random encoder.
    with torch.device("meta"):
        encoder = Sam3ViTModel(vision_config)
    encoder.load_state_dict(weights, strict=True, assign=True)
    del weights
    wrapped = SAM3BackboneWrapper(encoder, microbatch_size=microbatch_size, identity={
        "source": source, "revision": resolved_revision, "encoder_sha256": weights_sha256,
        "model_id": model_id, "weight_format": filename.suffix, "layer_norm_eps": vision_config.layer_norm_eps,
    })
    # The only nonpersistent meta buffers are native RoPE, reconstructed here.
    size = vision_config.image_size
    height, width = (size, size) if isinstance(size, int) else size
    wrapped._set_rotary_grid(height, width, torch.device("cpu"))
    wrapped = wrapped.to(device=device, dtype=dtype).eval()
    wrapped._rotary_grid = None  # rebuild FP32 trigonometric buffers on first use
    wrapped.model_type = release.lower() + "-encoder"
    wrapped.norm_label = release
    print(f"  [{release}] Frozen image encoder; mean-pool classification; spatial patch segmentation.")
    print(f"  [{release}] Uses benchmark resolutions. Native prompted model operates at 1008px.")
    return wrapped, None, None, False
