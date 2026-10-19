"""Benchmark-compatible loading with an additional strict checkpoint audit."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from .common import digest, file_sha, normalize_features


def registry(path=None):
    return json.loads(Path(path or Path(__file__).with_name("models.json")).read_text())


def selected_specs(keys, config=None):
    entries = registry(config)
    requested = list(entries) if keys == ["all"] else keys
    if "all" in requested or len(set(requested)) != len(requested):
        raise ValueError("Use --models all or a list of unique model keys")
    for key in requested:
        if key not in entries:
            raise ValueError(f"Unknown model {key}; choose {', '.join(entries)}")
    return [(key, entries[key]) for key in requested]


def source_identity(spec):
    """Hash local weight bytes, including the foundation weights beneath custom checkpoints."""
    from eval_suite import config  # establishes the same HF cache root as benchmarks
    from huggingface_hub import try_to_load_from_cache
    sources = []
    if spec.get("checkpoint"):
        sources.append(Path(spec["checkpoint"]))
    if spec.get("model_path"):
        folder = Path(spec["model_path"])
        sources.extend(p for p in folder.iterdir() if p.is_file() and p.suffix in (".json", ".pt", ".safetensors", ".bin"))
    elif spec["model_id"] != "marineinst-vit-h-stage1":
        for filename in ("config.json", "model.safetensors", "pytorch_model.bin", "model.safetensors.index.json"):
            found = try_to_load_from_cache(spec["model_id"], filename)
            if isinstance(found, str):
                sources.append(Path(found))
                if filename.endswith("index.json"):
                    for shard in set(json.loads(Path(found).read_text())["weight_map"].values()):
                        shard_path = try_to_load_from_cache(spec["model_id"], shard)
                        if not isinstance(shard_path, str):
                            raise FileNotFoundError(f"Missing cached shard {shard}")
                        sources.append(Path(shard_path))
    if not sources or not any(p.suffix in (".ckpt", ".pth", ".pt", ".bin", ".safetensors") for p in sources):
        raise FileNotFoundError(f"No cached weights found for {spec['model_id']}; supply a local path in models.json")
    entries = []
    for path in sorted(set(sources)):
        if not path.is_file():
            raise FileNotFoundError(f"Required model file is missing: {path}")
        entries.append(dict(path=str(path), resolved=str(path.resolve()), bytes=path.stat().st_size, sha256=file_sha(path)))
    return dict(spec=spec, files=entries)


def checkpoint_state(checkpoint):
    for key in ("teacher_state_dict", "teacher", "model_state_dict", "state_dict"):
        if key in checkpoint:
            return key, checkpoint[key]
    return "raw", checkpoint


def audit_backbone(model, checkpoint_path, checkpoint=None):
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    selected, original = checkpoint_state(checkpoint)
    # Match the shared benchmark loader exactly, rather than inventing another interpretation.
    cleaned = {k.replace("_orig_mod.", "").replace("backbone.", "").replace("teacher.", ""): v for k, v in original.items() if torch.is_tensor(v)}
    loaded = model.state_dict()
    missing = [k for k in loaded if k not in cleaned]
    wrong = [k for k in loaded if k in cleaned and tuple(loaded[k].shape) != tuple(cleaned[k].shape)]
    if missing or wrong:
        raise ValueError(f"Incomplete backbone checkpoint: missing={missing[:8]}, wrong_shape={wrong[:8]}")
    unequal = [k for k, value in loaded.items() if not torch.equal(value.detach().cpu(), cleaned[k].to(dtype=value.dtype))]
    if unequal:
        raise ValueError(f"Backbone differs from selected checkpoint tensors: {unequal[:8]}")
    ignored = sorted(set(cleaned) - set(loaded))
    allowed = ("dino_head.", "ibot_head.", "head.", "projection_head.")
    unexpected = [k for k in ignored if not k.startswith(allowed)]
    if unexpected:
        raise ValueError(f"Unexpected non-head checkpoint tensors: {unexpected[:8]}")
    return dict(selected_state=selected, matched_tensors=len(loaded), missing=[], mismatched=[], ignored_head_tensors=ignored, epoch=checkpoint.get("epoch"))


def audit_foundation(model, spec):
    """Reject randomly initialized/missing HF backbone tensors, including baseline models."""
    from huggingface_hub import try_to_load_from_cache
    from safetensors import safe_open
    paths = {}
    for filename in ("model.safetensors", "pytorch_model.bin"):
        cached = try_to_load_from_cache(spec["model_id"], filename)
        if isinstance(cached, str):
            paths[filename] = cached
    binary = None
    reader = None
    if "model.safetensors" in paths:
        reader = safe_open(paths["model.safetensors"], framework="pt", device="cpu")
        available = set(reader.keys())
    elif "pytorch_model.bin" in paths:
        binary = torch.load(paths["pytorch_model.bin"], map_location="cpu", weights_only=True)
        available = set(binary)
    else:
        raise FileNotFoundError(f"Strict HF audit needs cached model.safetensors or pytorch_model.bin: {spec['model_id']}")
    missing, unequal = [], []
    count = 0
    for name, value in model.named_parameters():
        source_name = name.removeprefix("raw_model.")
        if source_name not in available:
            missing.append(source_name)
            continue
        reference = reader.get_tensor(source_name) if reader is not None else binary[source_name]
        if tuple(reference.shape) != tuple(value.shape) or not torch.equal(value.detach().cpu(), reference.to(dtype=value.dtype)):
            unequal.append(source_name)
        count += 1
    if missing or unequal:
        raise ValueError(f"Incomplete official backbone: missing={missing[:8]}, mismatched={unequal[:8]}")
    return dict(selected_state="official cached backbone", matched_parameter_tensors=count, missing=[], mismatched=[])


def load_backbone(spec, device, attn_implementation="sdpa"):
    # Establish the project's HF cache before importing Transformers (which snapshots
    # cache environment variables at import time). Attention-only runs skip source_identity.
    from eval_suite.config import IMAGENET_MEAN, IMAGENET_STD, BENTHIC_MEAN, BENTHIC_STD
    from eval_suite.models import load_dinov3_backbone, load_baseline_backbone
    if spec["kind"] == "dinov3":
        # Reuse the benchmark architecture, then load its exact teacher selection with
        # memory-mapped checkpoint storage. This avoids reading unused optimizer/student
        # tensors from the large training files a second time for the strict audit.
        model, epoch, loss, benthic = load_dinov3_backbone(None, spec["model_id"], device, torch.float32, attn_implementation=attn_implementation)
        checkpoint = None
        if spec.get("checkpoint"):
            checkpoint = torch.load(spec["checkpoint"], map_location="cpu", weights_only=False, mmap=True)
            _, state = checkpoint_state(checkpoint)
            cleaned = {k.replace("_orig_mod.", "").replace("backbone.", "").replace("teacher.", ""): v for k, v in state.items()}
            expected = model.state_dict()
            missing = [name for name in expected if name not in cleaned]
            if missing:
                raise ValueError(f"Incomplete backbone checkpoint: missing={missing[:8]}")
            model.load_state_dict({name: cleaned[name] for name in expected}, strict=True)
            epoch = checkpoint.get("epoch")
            loss = checkpoint.get("loss", checkpoint.get("ssl_loss"))
            cfg = checkpoint.get("config", {})
            benthic = cfg.get("benthic_norm", False) if isinstance(cfg, dict) else False
    else:
        kwargs = {}
        if spec.get("model_path"):
            kwargs["sam3_model_path"] = spec["model_path"]
        if spec["model_id"] == "marineinst-vit-h-stage1":
            kwargs["marineinst_checkpoint"] = spec["checkpoint"]
        model, epoch, loss, benthic = load_baseline_backbone(spec["model_id"], device, torch.float32, attn_implementation=attn_implementation, **kwargs)
    if attn_implementation == "eager":
        # The shared CLIP branch loads its default implementation. Configure the
        # underlying HF encoder explicitly for visualization without changing weights.
        target = getattr(model, "raw_model", model)
        if hasattr(target, "set_attn_implementation"):
            target.set_attn_implementation("eager")
    if spec["kind"] == "dinov3" and spec.get("checkpoint"):
        audit = audit_backbone(model, spec["checkpoint"], checkpoint)
    elif spec["model_id"] in ("facebook/sam3.1", "marineinst-vit-h-stage1"):
        audit = {"selected_state": "official cached backbone", "strict_native_adapter": True}
    else:
        audit = audit_foundation(model, spec)
    model.eval().requires_grad_(False)
    mean = (BENTHIC_MEAN if benthic else IMAGENET_MEAN) if spec["kind"] == "dinov3" else getattr(model, "norm_mean", None) or IMAGENET_MEAN
    std = (BENTHIC_STD if benthic else IMAGENET_STD) if spec["kind"] == "dinov3" else getattr(model, "norm_std", None) or IMAGENET_STD
    patch = getattr(model, "patch_size", getattr(model.config, "patch_size", 16))
    if isinstance(patch, (tuple, list)):
        if patch[0] != patch[1]:
            raise ValueError("Only square patches are supported")
        patch = patch[0]
    metadata = dict(epoch=epoch, ssl_loss=loss, normalization=dict(mean=list(mean), std=list(std)),
                    patch_size=int(patch), hidden_size=int(model.config.hidden_size),
                    parameters=sum(p.numel() for p in model.parameters()), frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
                    pooling="mean pre-neck patches" if spec["model_id"] == "marineinst-vit-h-stage1" else "mean patches" if spec["model_id"] == "facebook/sam3.1" else "CLS",
                    checkpoint_audit=audit)
    return model, metadata


def transform(mean, std, image_size=224, patch=1):
    steps = [T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True), T.ToTensor()]
    padding = (-image_size) % patch
    if padding:
        steps.append(T.Pad((0, 0, padding, padding), padding_mode="reflect" if image_size > padding else "edge"))
    steps.append(T.Normalize(mean, std))
    return T.Compose(steps)


def forward_tokens(model, images):
    try:
        output = model(images, interpolate_pos_encoding=True)
    except TypeError:
        output = model(images)
    tokens = output.last_hidden_state if hasattr(output, "last_hidden_state") else output[0]
    if tokens.ndim != 3 or not torch.isfinite(tokens).all():
        raise ValueError("Backbone returned invalid token features")
    return tokens


def spatial_tokens(tokens, height, width, patch):
    if height % patch or width % patch:
        raise ValueError("Patch input must be padded to a patch multiple")
    gh, gw = height // patch, width // patch
    count = gh * gw
    if tokens.shape[1] < count:
        raise ValueError(f"Expected {count} patches; got {tokens.shape[1]} tokens")
    return tokens[:, -count:, :], gh, gw


def patch_input(image, longest, patch, mean, std):
    scale = longest / max(image.size)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    resized = T.functional.resize(image, [height, width], interpolation=T.InterpolationMode.BICUBIC, antialias=True)
    tensor = T.functional.to_tensor(resized)
    pad_w, pad_h = (-width) % patch, (-height) % patch
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect" if width > pad_w and height > pad_h else "replicate")
    tensor = T.functional.normalize(tensor, mean, std)
    return tensor, dict(width=width, height=height, padded_width=width + pad_w, padded_height=height + pad_h)


@torch.inference_mode()
def extract(model, metadata, records, patch_indices, batch_size, patch_longest, device, image_size=224):
    norm = metadata["normalization"]
    convert = transform(norm["mean"], norm["std"], image_size, metadata["patch_size"])
    features = []
    for start in range(0, len(records), batch_size):
        images = []
        for row in records[start:start + batch_size]:
            with Image.open(row["staged_image"]) as image:
                images.append(convert(image.convert("RGB")))
        tokens = forward_tokens(model, torch.stack(images).to(device))
        features.append(tokens[:, 0].float().cpu().numpy())
        print(f"  Global features {min(start + batch_size, len(records))}/{len(records)}", flush=True)
    patches, geometries = [], []
    for count, index in enumerate(patch_indices):
        with Image.open(records[index]["staged_image"]) as image:
            tensor, geometry = patch_input(image.convert("RGB"), patch_longest, metadata["patch_size"], norm["mean"], norm["std"])
        tokens = forward_tokens(model, tensor.unsqueeze(0).to(device))
        spatial, gh, gw = spatial_tokens(tokens, *tensor.shape[-2:], metadata["patch_size"])
        patches.append(spatial[0].float().cpu().numpy())
        geometries.append(dict(**geometry, grid_h=gh, grid_w=gw, index=index))
        print(f"  Patch features {count + 1}/{len(patch_indices)}", flush=True)
    raw = np.concatenate(features).astype(np.float32)
    return raw, normalize_features(raw), patches, geometries
