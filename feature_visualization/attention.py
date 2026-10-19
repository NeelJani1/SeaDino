"""Genuine encoder attention, with explicit CLS and patch-only semantics.

Never substitutes a pooled adapter token for a CLS query. Native windowed
encoders expose their last *global* received patch attention, not CLS rollout.
"""
from contextlib import contextmanager
import html
import json
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .backbones import patch_input
from .common import digest, file_sha, write_json
from .dense import legend_html, mask_view


def feature_parity(actual, reference):
    """Compare entire patch vectors across FP32 eager/SDPA accumulation orders.

    A fixed near-zero element tolerance is unstable for CLIP: identical weights
    and a bitwise-reproducible SDPA forward can differ by 0.0033 in one eager
    feature element. Bound every patch vector as well as the whole tensor.
    """
    if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("Attention extraction changed the cached patch geometry or produced invalid features")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    norms = np.linalg.norm(reference.astype(np.float64), axis=1)
    relative = np.linalg.norm(delta, axis=1) / np.maximum(norms, 1e-12)
    maximum = float(relative.max())
    overall = float(np.linalg.norm(delta) / max(np.linalg.norm(reference.astype(np.float64)), 1e-12))
    if maximum > 5e-4 or overall > 1e-4:
        raise ValueError(f"Attention-enabled forward differs from cached encoder features: maximum patch relative L2={maximum:.6g}, overall relative L2={overall:.6g}")
    return dict(maximum_absolute_error=float(np.abs(delta).max()), maximum_patch_relative_l2=maximum,
                global_relative_l2=overall, maximum_patch_relative_l2_limit=5e-4,
                global_relative_l2_limit=1e-4, passed=True,
                reference="previous frozen-feature patch cache", criterion="FP32 eager/SDPA feature-vector agreement")


def validate_attention(weights):
    if weights is None or weights.ndim != 4 or weights.shape[0] != 1 or weights.shape[-1] != weights.shape[-2]:
        raise ValueError("Expected one image's attention with shape 1 x heads x tokens x tokens")
    if not torch.isfinite(weights).all() or weights.min().item() < -1e-7:
        raise ValueError("Attention contains non-finite or negative probabilities")
    error = float((weights.float().sum(-1) - 1).abs().max())
    if error > 2e-4:
        raise ValueError(f"Attention rows are not probability distributions: maximum error={error}")
    return error


def summarize_cls(attentions, patches):
    if not attentions:
        raise ValueError("The model did not return attention weights; eager attention is required")
    rollout = None
    errors = []
    for weights in attentions:
        errors.append(validate_attention(weights))
        fused = weights[0].float().mean(0)
        if fused.shape[0] <= patches:
            raise ValueError("CLS attention requires at least one non-spatial token")
        fused = fused + torch.eye(len(fused), device=fused.device)
        fused = fused / fused.sum(-1, keepdim=True)
        rollout = fused if rollout is None else fused @ rollout
    heads = attentions[-1][0, :, 0, -patches:].float()
    last = heads.mean(0)
    rolled = rollout[0, -patches:]
    return dict(last=last.cpu().numpy(), heads=heads.cpu().numpy(), rollout=rolled.cpu().numpy()), dict(
        query="learned CLS token", method="last-layer CLS-to-patch attention; residual attention rollout",
        layers=len(attentions), heads=int(heads.shape[0]), spatial_tokens=patches,
        register_tokens_excluded_from_maps=True, registers_retained_in_rollout=True,
        last_layer_spatial_mass=float(last.sum()), rollout_spatial_mass=float(rolled.sum()), maximum_row_sum_error=max(errors))


def summarize_received(weights, patches, layer, family):
    error = validate_attention(weights)
    if weights.shape[-1] != patches:
        raise ValueError(f"Expected {patches} global spatial keys; received {weights.shape[-1]}")
    heads = weights[0].float().mean(1)
    last = heads.mean(0)
    return dict(last=last.cpu().numpy(), heads=heads.cpu().numpy()), dict(
        query="uniform average over all spatial patch queries", method="last global layer received patch attention",
        family=family, layer_index=layer, heads=int(heads.shape[0]), spatial_tokens=patches,
        last_layer_spatial_mass=float(last.sum()), maximum_row_sum_error=error,
        rollout_unavailable="No learned CLS token; global CLS rollout is not defined across native windowed layers")


def marine_attention_weights(module, x):
    """Exact upstream QK/relative-position/softmax expression, without changing output."""
    from eval_suite.vendor.marineinst.image_encoder import add_decomposed_rel_pos
    batch, height, width, _ = x.shape
    qkv = module.qkv(x).reshape(batch, height * width, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.reshape(3, batch * module.num_heads, height * width, -1).unbind(0)
    scores = (q * module.scale) @ k.transpose(-2, -1)
    if module.use_rel_pos:
        scores = add_decomposed_rel_pos(scores, q, module.rel_pos_h, module.rel_pos_w, (height, width), (height, width))
    return scores.softmax(-1).reshape(batch, module.num_heads, height * width, height * width)


def sam_attention_weights(module, x, position_embeddings):
    """The installed HF SAM3 QK + 2D RoPE formula; preserve SDPA encoder output."""
    from transformers.models.sam3.modeling_sam3 import apply_rotary_pos_emb_2d
    batch, height, width, _ = x.shape
    shape = (batch, height * width, module.num_attention_heads, module.head_dim)
    query = module.q_proj(x).view(*shape).transpose(1, 2)
    key = module.k_proj(x).view(*shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb_2d(query, key, cos=cos, sin=sin)
    return (torch.matmul(query, key.transpose(2, 3)) * module.scaling).softmax(-1)


@contextmanager
def native_attention_capture(model, spec, patches):
    captured = {}
    if spec["model_id"] == "facebook/sam3.1":
        layers = model.raw_model.layers
        layer_index = max(i for i, layer in enumerate(layers) if layer.window_size == 0)
        module = layers[layer_index].attention
        def hook(module, arguments):
            weights = sam_attention_weights(module, arguments[0], arguments[1])
            arrays, metadata = summarize_received(weights, patches, layer_index, "SAM3.1")
            captured.update(arrays=arrays, metadata=metadata)
        handle = module.register_forward_pre_hook(hook)
    elif spec["model_id"] == "marineinst-vit-h-stage1":
        layers = model.raw_model.blocks
        layer_index = max(i for i, layer in enumerate(layers) if layer.window_size == 0)
        module = layers[layer_index].attn
        def hook(module, arguments):
            weights = marine_attention_weights(module, arguments[0])
            arrays, metadata = summarize_received(weights, patches, layer_index, "MarineInst")
            captured.update(arrays=arrays, metadata=metadata)
        handle = module.register_forward_pre_hook(hook)
    else:
        raise ValueError("Unknown native attention family")
    try:
        yield captured
    finally:
        handle.remove()


@torch.inference_mode()
def extract_image_attention(model, spec, image, normalization, patch, longest, device, reference_patches=None):
    tensor, geometry = patch_input(image, longest, patch, normalization["mean"], normalization["std"])
    images = tensor.unsqueeze(0).to(device)
    gh, gw = tensor.shape[-2] // patch, tensor.shape[-1] // patch
    count = gh * gw
    if spec["model_id"] in ("facebook/sam3.1", "marineinst-vit-h-stage1"):
        with native_attention_capture(model, spec, count) as captured:
            output = model(images)
        if not captured:
            raise ValueError("The native global attention hook did not run")
        arrays, metadata = captured["arrays"], captured["metadata"]
    else:
        try:
            output = model(images, output_attentions=True, interpolate_pos_encoding=True)
        except TypeError:
            output = model(images, output_attentions=True)
        arrays, metadata = summarize_cls(getattr(output, "attentions", None), count)
    hidden = output.last_hidden_state
    if hidden.shape[1] < count or not torch.isfinite(hidden).all():
        raise ValueError("Invalid encoder output during attention extraction")
    if reference_patches is not None:
        actual = hidden[0, -count:].float().cpu().numpy()
        metadata["feature_parity"] = feature_parity(actual, reference_patches)
    geometry.update(grid_h=gh, grid_w=gw)
    implementation = "native QK plus 2D RoPE; SDPA encoder unchanged" if spec["model_id"] == "facebook/sam3.1" else "native QK plus decomposed relative position" if spec["model_id"] == "marineinst-vit-h-stage1" else "eager"
    metadata.update(geometry=geometry, precision="float32", attn_implementation=implementation,
                    class_conditional=False, note="Attention describes internal mixing, not a segmentation prediction or causal explanation")
    return arrays, metadata


def dense_map(values, geometry):
    gh, gw = geometry["grid_h"], geometry["grid_w"]
    grid = torch.as_tensor(values).float().reshape(1, 1, gh, gw)
    # Bilinear interpolation preserves non-negativity; crop right/bottom patch padding.
    dense = F.interpolate(grid, size=(geometry["padded_height"], geometry["padded_width"]), mode="bilinear", align_corners=False)
    return dense[0, 0, :geometry["height"], :geometry["width"]].numpy()


def color_attention(values, geometry):
    dense = dense_map(values, geometry)
    low, high = np.percentile(dense, [2, 98])
    constant = high - low <= 1e-6 * max(abs(low), abs(high), 1e-12)
    # Do not turn FP32 interpolation roundoff in a uniform map into fake contrast.
    scaled = np.zeros_like(dense) if constant else np.clip((dense - low) / max(high - low, 1e-12), 0, 1)
    return colormaps["magma"](scaled)[..., :3], dict(percentiles=[2, 98], low=float(low), high=float(high), constant=bool(constant))


def render_attention_image(directory, record, arrays, metadata, model_name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = record["sample_id"][:20]
    geometry = metadata["geometry"]
    with Image.open(record["staged_image"]) as image:
        original = np.asarray(image.convert("RGB").resize((geometry["width"], geometry["height"]), Image.Resampling.BICUBIC)) / 255.
    original_path = directory / f"{stem}_original.png"
    Image.fromarray((original * 255).round().astype(np.uint8)).save(original_path)
    panels = [("Original", original)]
    if record.get("mask_path"):
        mask, _ = mask_view(record, (geometry["width"], geometry["height"]))
        panels.append(("Ground-truth semantic mask", mask))
        metadata["semantic_reference"] = dict(mask_sha256=record["mask_sha256"], resize="nearest neighbor", used_as_model_input=False)
    display_limits = {}
    for name in ("last", "rollout"):
        if name not in arrays:
            continue
        heat, limits = color_attention(arrays[name], geometry)
        display_limits[name] = limits
        overlay = .55 * original + .45 * heat
        for suffix, value in (("heatmap", heat), ("overlay", overlay)):
            Image.fromarray((value * 255).round().clip(0, 255).astype(np.uint8)).save(directory / f"{stem}_{name}_{suffix}.png")
        panels.append(("Last-layer CLS attention" if name == "last" and "rollout" in arrays else "Last global received attention" if name == "last" else "CLS attention rollout", overlay))
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5), squeeze=False)
    for ax, (label, image) in zip(axes[0], panels):
        ax.imshow(image); ax.set_title(label); ax.axis("off")
    box = None
    if metadata.get("bbox_format") == "COCO xywh" and record.get("bbox"):
        values = ast.literal_eval(record["bbox"]) if isinstance(record["bbox"], str) else record["bbox"]
        x, y, w, h = map(float, values)
        sx, sy = geometry["width"] / record["width"], geometry["height"] / record["height"]
        box = (x * sx, y * sy, w * sx, h * sy)
        from matplotlib.patches import Rectangle
        for ax in axes[0]:
            ax.add_patch(Rectangle(box[:2], box[2], box[3], linewidth=1.2, edgecolor="cyan", facecolor="none", linestyle="--"))
    fig.suptitle(f"{model_name} | {record.get('label_name') or 'Unlabeled'}\n{metadata['query']}; display contrast scaled per image")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}_comparison.{extension}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    heads = arrays["heads"]
    cols = min(4, len(heads)); rows = (len(heads) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 2.6 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (ax, values) in enumerate(zip(axes.flat, heads)):
        heat, _ = color_attention(values, geometry)
        ax.imshow(.55 * original + .45 * heat); ax.set_title(f"Head {i+1}")
    fig.suptitle(f"{model_name} | Last-layer individual heads\n{metadata['query']}; each head independently contrast scaled")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(directory / f"{stem}_heads.{extension}", dpi=140, bbox_inches="tight")
    plt.close(fig)
    metadata = dict(metadata, display_limits=display_limits, sample_id=record["sample_id"], source_image=record["image"], label=record.get("label_name"), displayed_annotation_box=box)
    write_json(directory / f"{stem}.json", metadata)
    np.savez_compressed(directory / f"{stem}.npz", **arrays)
    return dict(sample_id=record["sample_id"], comparison=f"{stem}_comparison.png", heads=f"{stem}_heads.png",
                last_overlay=str(directory / f"{stem}_last_overlay.png"), original=str(original_path), metadata=f"{stem}.json", arrays=f"{stem}.npz", query=metadata["query"], has_rollout="rollout" in arrays,
                arrays_sha256=file_sha(directory / f"{stem}.npz"), metadata_sha256=file_sha(directory / f"{stem}.json"), feature_parity=metadata.get("feature_parity"),
                mask_color=record.get("mask_color"), semantic_legend=legend_html(record) if record.get("mask_path") else "")


def render_attention_page(directory, name, items):
    directory = Path(directory)
    cards = []
    for n, item in enumerate(items):
        with Image.open(directory / item["comparison"]) as image:
            width, height = image.size
        cards.append(f'<article><a href="{item["comparison"]}"><img width="{width}" height="{height}" style="width:100%;height:auto" loading="{"eager" if n == 0 else "lazy"}" src="{item["comparison"]}" alt="Attention overlays"></a>{item.get("semantic_legend", "")}<p><a href="{item["heads"]}">Individual heads</a> · <a href="{item["metadata"]}">Metadata</a> · <a href="{item["arrays"]}">Raw attention arrays</a></p></article>')
    query = items[0]["query"] if items else "Unavailable"
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(name)} attention</title><style>body{{font:16px system-ui;background:#f4f7fa;color:#18283c;margin:24px auto;max-width:1350px;padding:0 20px}}article{{background:white;border-radius:12px;padding:15px;margin:22px 0}}img{{width:100%}}a{{color:#0874a7}}p{{line-height:1.6}}</style></head><body><a href="../../index.html">Back to all models</a><h1>{html.escape(name)} attention maps</h1><p><strong>Query:</strong> {html.escape(query)}. Maps are not class-conditioned. Attention is internal token mixing, not a fish/coral prediction or proof of causal importance.</p><p>DINO/SeaDino/CLIP show learned CLS-to-patch attention and residual attention rollout. SAM3.1/MarineInst have no learned CLS token: their maps show attention received by keys, averaged over spatial queries in the last global layer. These are different views and their colors are not quantitatively comparable.</p><p>Overlays use per-image 2nd–98th percentile contrast. Each head is scaled independently. Raw, unscaled probabilities and geometry are downloadable. Padded borders are removed; CLS/register tokens are excluded from spatial maps but retained during rollout. Where available, the dashed cyan box is the dataset annotation, not a model prediction.</p>{''.join(cards)}</body></html>'''
    (directory / "index.html").write_text(page)


def render_attention_gallery(root, results):
    root = Path(root)
    by_sample = {}
    for key, result in results.items():
        for item in result.get("items", []):
            by_sample.setdefault(item["sample_id"], []).append((result["name"], item))
    directory = root / "attention_comparisons"
    directory.mkdir(exist_ok=True)
    images = []
    for sample_id, entries in by_sample.items():
        panels = [("Original", entries[0][1]["original"])]
        if entries[0][1].get("mask_color"):
            panels.append(("Ground-truth semantic mask", entries[0][1]["mask_color"]))
        panels += [(name + (" | CLS" if item["has_rollout"] else " | received patch"), item["last_overlay"]) for name, item in entries]
        cols = min(4, len(panels)); rows = (len(panels) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, (name, path) in zip(axes.flat, panels):
            with Image.open(path) as image:
                ax.imshow(image)
            ax.set_title(name, fontsize=10)
        fig.suptitle("Last-layer attention | each image independently contrast scaled\nCLS-query and received-patch views have different semantics")
        fig.tight_layout()
        name = sample_id[:20]
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"{name}.{extension}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        images.append(f'<a href="{name}.png"><img loading="lazy" src="{name}.png" alt="All-model attention comparison"></a>')
    (directory / "index.html").write_text('<!doctype html><html><head><meta charset="utf-8"><title>All-model attention comparisons</title><style>body{font:16px system-ui;max-width:1500px;margin:24px auto}img{width:100%;margin:20px 0}</style></head><body><a href="../index.html">Back to report</a><h1>Attention comparisons</h1><p>Different query types are labeled. Overlays use independent contrast and do not constitute segmentation scores.</p>' + ''.join(images) + '</body></html>')


def run_attention(root, run, keys, longest=512, maximum=20, device_name="auto", cached_only=False):
    """Attach attention to a saved run without touching image embeddings or scores."""
    from .backbones import load_backbone
    from .common import code_identity
    import gc
    root = Path(root)
    saved_path = root / "attention_run.json"
    state = json.loads(saved_path.read_text()) if saved_path.exists() else dict(schema_version=1, results={})
    # Keep the producing code identities of earlier, complete model outputs when
    # a resumed subset uses a newer validation/rendering implementation.
    for previous in state["results"].values():
        previous.setdefault("extraction_code", state.get("extraction_code", {}))
    selected_indices = run["patch_indices"][:maximum]
    if not selected_indices:
        selected_indices = list(range(min(maximum, len(run["records"]))))
    if maximum < 1 or longest < 1:
        raise ValueError("attention_images and attention_longest must be positive")
    code = code_identity()
    import importlib.metadata
    runtime = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy")}
    fingerprint_code = {k: v for k, v in code.items() if k.startswith("eval_suite/") or Path(k).name in ("attention.py", "backbones.py", "common.py", "dense.py")}
    import eval_suite
    vendor = Path(eval_suite.__file__).parent / "vendor/marineinst"
    for filename in ("image_encoder.py", "common.py"):
        fingerprint_code["eval_suite/vendor/marineinst/" + filename] = file_sha(vendor / filename)
    state.pop("error", None)
    state.update(state="running", code=code, extraction_code=fingerprint_code, runtime=runtime, longest_side=longest, requested_models=keys, sample_indices=selected_indices)
    write_json(saved_path, state)
    def refresh_index():
        from .render import render_index
        results = [run["results"][k] for k in run["models"] if k in run["results"]]
        comparisons = [str(p.relative_to(root)) for p in sorted((root / "patch_comparisons").glob("*.png"))]
        render_index(root, run, results, comparisons)
    refresh_index()
    try:
        for key in keys:
            spec = run["models"][key]
            sources = run["caches"][key]["source"]
            records = [run["records"][i] for i in selected_indices]
            identity = digest(dict(source=sources, images=[(r["sample_id"], r["image_sha256"], r.get("mask_sha256")) for r in records], longest=longest, code=fingerprint_code, runtime=runtime, precision="float32"))
            old = state["results"].get(key)
            if old and old.get("identity") == identity and old.get("state") == "complete":
                for item in old["items"]:
                    directory = root / key / "attention"
                    for field in ("arrays", "metadata"):
                        if file_sha(directory / item[field]) != item[field + "_sha256"]:
                            raise ValueError(f"Cached attention {field} checksum failed: {key}")
                print(f"  Attention cache hit: {spec['name']}", flush=True)
                continue
            if cached_only:
                raise ValueError(f"No matching cached attention maps for {key}; run --attention_only first")
            # Check every saved source byte identity before loading new attention weights.
            for source in sources["files"]:
                if file_sha(source["path"]) != source["sha256"]:
                    raise ValueError(f"Model source changed since feature extraction: {source['path']}")
            for record in records:
                if file_sha(record["staged_image"]) != record["image_sha256"]:
                    raise ValueError(f"Staged image changed: {record['sample_id']}")
                if record.get("mask_path") and file_sha(record["mask_path"]) != record["mask_sha256"]:
                    raise ValueError(f"Semantic mask changed: {record['sample_id']}")
            device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else "cpu" if device_name == "auto" else device_name)
            print(f"\nAttention: {spec['name']}", flush=True)
            model, metadata = load_backbone(spec, device, attn_implementation="sdpa" if spec["model_id"] == "facebook/sam3.1" else "eager")
            items = []
            directory = root / key / "attention"
            reference = np.load(run["caches"][key]["arrays"], allow_pickle=False)
            try:
                for count, record in enumerate(records):
                    previous = None
                    original_index = selected_indices[count]
                    if longest == run.get("arguments", {}).get("patch_longest", 512) and original_index in run["patch_indices"]:
                        previous = reference[f"patch_{run['patch_indices'].index(original_index)}"]
                    with Image.open(record["staged_image"]) as image:
                        arrays, details = extract_image_attention(model, spec, image.convert("RGB"), metadata["normalization"], metadata["patch_size"], longest, device, previous)
                    if run["source_name"] == "fathomnet":
                        details["bbox_format"] = "COCO xywh"
                    items.append(render_attention_image(directory, record, arrays, details, spec["name"]))
                    print(f"  Attention image {count + 1}/{len(records)}", flush=True)
            finally:
                reference.close()
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            render_attention_page(directory, spec["name"], items)
            state["results"][key] = dict(name=spec["name"], state="complete", identity=identity, items=items, page=f"{key}/attention/index.html", extraction_code=fingerprint_code)
            write_json(saved_path, state)
            refresh_index()
        render_attention_gallery(root, state["results"])
        state["state"] = "complete"
        write_json(saved_path, state)
        refresh_index()
    except Exception as exc:
        state.update(state="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(saved_path, state)
        raise
    return state
