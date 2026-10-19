#!/usr/bin/env python3
"""SeaDino Feature & Attention Visualizer.

Generates qualitative interpretability proofs for self-supervised vision models:
1. Patch Feature PCA: Maps top-3 principal components of spatial patch tokens to RGB
   with sign canonicalization to visualize fine-grained semantic boundary emergence.
2. [CLS] Attention Rollout: Accumulates multi-head self-attention matrices across all
   transformer layers (Abnar & Zuidema, 2020) and projects organism saliency heatmaps
   directly onto seafloor and fauna imagery.

Supports simultaneous comparison across:
- SeaDino checkpoints (.ckpt)
- Off-the-shelf DINOv3 baseline
- External foundation baselines (BioCLIP, DINOv2, OpenAI CLIP)
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from eval_suite.config import (
    BENTHIC_MEAN,
    BENTHIC_STD,
    DATA_DIR,
    GDRIVE_CANDIDATE_PATHS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    REPO_ROOT,
)
from eval_suite.models import (
    CLIP_MEAN,
    CLIP_STD,
    load_baseline_backbone,
    load_dinov3_backbone,
)


def canonicalize_pca_signs(components: np.ndarray) -> np.ndarray:
    """Canonicalize eigenvector signs to prevent arbitrary color inversions across models.
    Enforces positive skewness so high-magnitude activations consistently map to bright channels.
    """
    for i in range(components.shape[1]):
        comp = components[:, i]
        # If skewness is negative, flip the eigenvector
        skew = np.mean((comp - np.mean(comp)) ** 3)
        if skew < 0:
            components[:, i] *= -1.0
    return components


@torch.no_grad()
def extract_patch_features_and_attentions(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
    """Runs forward pass with eager attention tracking.
    Returns:
        patch_tokens: (N_patches, Hidden_Dim)
        attention_matrices: List of (Num_Heads, Seq_Len, Seq_Len) for each layer
    """
    model.eval()
    pixel_values = pixel_values.to(device)

    try:
        outputs = model(pixel_values, output_attentions=True, output_hidden_states=True)
    except TypeError:
        try:
            outputs = model(pixel_values, output_attentions=True)
        except TypeError:
            outputs = model(pixel_values)

    hidden = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    # Squeeze batch dimension (assuming batch size 1)
    tokens = hidden.squeeze(0)  # (Seq_Len, D)

    # Determine patch size and sequence length
    patch_size = getattr(model, "patch_size", 16)
    grid_h = pixel_values.shape[2] // patch_size
    grid_w = pixel_values.shape[3] // patch_size
    num_patches = grid_h * grid_w

    # Spatial patch tokens (skip [CLS] and register tokens from front)
    patch_tokens = tokens[-num_patches:]

    attentions = getattr(outputs, "attentions", None)
    if attentions is not None:
        attentions = [a.squeeze(0).cpu() for a in attentions]

    return patch_tokens.cpu(), attentions


def compute_patch_pca_rgb(patch_tokens: torch.Tensor, grid_h: int, grid_w: int) -> np.ndarray:
    """Projects spatial patch tokens into top-3 principal components rendered as RGB (0..1)."""
    feats = patch_tokens.float().numpy()  # (N, D)
    pca = PCA(n_components=3)
    pca_proj = pca.fit_transform(feats)  # (N, 3)
    pca_proj = canonicalize_pca_signs(pca_proj)

    # Min-max normalization per channel with 2nd/98th percentile clipping for contrast
    rgb = np.zeros_like(pca_proj)
    for c in range(3):
        low, high = np.percentile(pca_proj[:, c], 2), np.percentile(pca_proj[:, c], 98)
        clipped = np.clip(pca_proj[:, c], low, high)
        denom = (high - low) if (high - low) > 1e-6 else 1.0
        rgb[:, c] = (clipped - low) / denom

    rgb_map = rgb.reshape(grid_h, grid_w, 3)
    return rgb_map


def compute_attention_rollout(
    attentions: List[torch.Tensor],
    grid_h: int,
    grid_w: int,
    head_fusion: str = "mean",
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """Computes attention rollout across all transformer layers (Abnar & Zuidema 2020).
    Args:
        attentions: List of (Num_Heads, Seq_Len, Seq_Len) tensors
        head_fusion: 'mean', 'max', or 'min'
    Returns:
        saliency_map: (grid_h, grid_w) normalized to [0, 1]
    """
    if not attentions:
        return np.zeros((grid_h, grid_w), dtype=np.float32)

    seq_len = attentions[0].shape[-1]
    rollout = torch.eye(seq_len)

    for attn in attentions:
        # 1. Head fusion: (H, S, S) -> (S, S)
        if head_fusion == "mean":
            attn_fused = attn.mean(dim=0)
        elif head_fusion == "max":
            attn_fused = attn.max(dim=0).values
        else:
            attn_fused = attn.mean(dim=0)

        # 2. Add identity residual connection and re-normalize rows
        # A_hat = 0.5 * A + 0.5 * I
        attn_hat = 0.5 * attn_fused + 0.5 * torch.eye(seq_len)
        attn_hat = attn_hat / attn_hat.sum(dim=-1, keepdim=True)

        # 3. Recursive rollout
        rollout = torch.matmul(attn_hat, rollout)

    # Extract [CLS] token attention to spatial patches (row 0)
    cls_attn = rollout[0]
    num_patches = grid_h * grid_w
    spatial_attn = cls_attn[-num_patches:].numpy()

    # Normalize to [0, 1]
    low, high = np.percentile(spatial_attn, 1), np.percentile(spatial_attn, 99)
    spatial_attn = np.clip(spatial_attn, low, high)
    denom = (high - low) if (high - low) > 1e-6 else 1.0
    spatial_attn = (spatial_attn - low) / denom

    return spatial_attn.reshape(grid_h, grid_w)


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """Overlays a 2D saliency heatmap onto a PIL RGB image."""
    orig_np = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    h, w = orig_np.shape[:2]

    # Resize heatmap bicubic to image size
    heat_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
    heat_upsampled = F.interpolate(heat_tensor, size=(h, w), mode="bicubic", align_corners=False).squeeze().clamp(0, 1).numpy()

    # Apply colormap (jet / magma)
    colormap = cm.get_cmap("jet")
    heat_rgb = colormap(heat_upsampled)[:, :, :3]

    blended = (1.0 - alpha) * orig_np + alpha * heat_rgb
    return np.clip(blended, 0.0, 1.0)


def upsample_pca_map(pca_map: np.ndarray, out_size: Tuple[int, int]) -> np.ndarray:
    """Upsamples low-res PCA patch map (GridH, GridW, 3) to full image size (H, W, 3)."""
    t = torch.from_numpy(pca_map).permute(2, 0, 1).unsqueeze(0).float()
    up = F.interpolate(t, size=out_size, mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    return up


def find_candidate_images(candidate_dir: Optional[str] = None, max_images: int = 4) -> List[Path]:
    """Finds high-quality marine test images from candidate directories."""
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    search_dirs = []

    if candidate_dir:
        search_dirs.append(Path(candidate_dir))

    for gd in GDRIVE_CANDIDATE_PATHS:
        search_dirs.extend([
            gd / "benchmarks" / "deepfish" / "DeepFish" / "Classification",
            gd / "benchmarks" / "brackish" / "test",
            gd / "benchmarks" / "fathomnet",
            gd / "benchmarks",
        ])

    search_dirs.extend([
        DATA_DIR / "benchmarks" / "deepfish" / "DeepFish" / "Classification",
        DATA_DIR / "benchmarks" / "brackish" / "test",
        REPO_ROOT / "data" / "benchmarks" / "deepfish",
        REPO_ROOT / "data" / "benchmarks" / "brackish",
        REPO_ROOT / "data",
    ])

    found = []
    for d in search_dirs:
        if d.exists() and d.is_dir():
            for ext in exts:
                found.extend(list(d.rglob(ext)))
                if len(found) >= max_images:
                    break
        if len(found) >= max_images:
            break

    return found[:max_images]


def main():
    p = argparse.ArgumentParser(description="SeaDino Feature & Attention Visualizer")
    p.add_argument("--image_path", type=str, default=None,
                   help="Path to a single image to visualize.")
    p.add_argument("--image_dir", type=str, default=None,
                   help="Directory of images to visualize.")
    p.add_argument("--max_images", type=int, default=4,
                   help="Max images to process if directory is scanned (default: 4).")
    p.add_argument("--checkpoints", nargs="+", default=[],
                   help="Checkpoint (.ckpt) paths to evaluate.")
    p.add_argument("--include_off_the_shelf", action="store_true", default=False,
                   help="Include un-pretrained DINOv3 foundation model.")
    p.add_argument("--baselines", nargs="+", default=[],
                   help="External foundation baselines (e.g. bioclip dinov2-base clip).")
    p.add_argument("--model_id", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    p.add_argument("--image_size", type=int, default=224,
                   help="ViT input resolution (default: 224).")
    p.add_argument("--output_dir", type=str, default="eval_output/visualizations",
                   help="Directory to save visual proof figures.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve Target Images
    image_paths: List[Path] = []
    if args.image_path:
        p_img = Path(args.image_path)
        if p_img.exists():
            image_paths = [p_img]
        else:
            sys.exit(f"Error: Specified image does not exist: {args.image_path}")
    else:
        image_paths = find_candidate_images(args.image_dir, max_images=args.max_images)

    if not image_paths:
        sys.exit("Error: No images found. Provide --image_path or verify data directory.")

    print(f"Found {len(image_paths)} target images for visualization:")
    for ip in image_paths:
        print(f"  • {ip}")

    # 2. Build Model Queue
    models_to_test = []

    if args.include_off_the_shelf:
        print("  • Loading Off-The-Shelf DINOv3 baseline...")
        m, _, _, _ = load_dinov3_backbone(None, args.model_id, device, torch.float32, attn_implementation="eager")
        models_to_test.append(("Off-The-Shelf DINOv3", m, IMAGENET_MEAN, IMAGENET_STD, 16))

    for ckpt in args.checkpoints:
        name = Path(ckpt).stem
        print(f"  • Loading SeaDino Checkpoint: {name}...")
        m, epoch, _, is_benthic = load_dinov3_backbone(ckpt, args.model_id, device, torch.float32, attn_implementation="eager")
        mean = BENTHIC_MEAN if is_benthic else IMAGENET_MEAN
        std = BENTHIC_STD if is_benthic else IMAGENET_STD
        label = f"SeaDino ({name})"
        models_to_test.append((label, m, mean, std, 16))

    for base in args.baselines:
        print(f"  • Loading External Baseline: {base}...")
        try:
            m, _, _, _ = load_baseline_backbone(base, device, torch.float32, attn_implementation="eager")
            mean = getattr(m, "norm_mean", IMAGENET_MEAN)
            std = getattr(m, "norm_std", IMAGENET_STD)
            patch_sz = getattr(m, "patch_size", 16)
            models_to_test.append((f"Baseline ({base})", m, mean, std, patch_sz))
        except Exception as e:
            print(f"    Warning: Failed to load baseline '{base}': {e}")

    if not models_to_test:
        sys.exit("Error: No models selected. Specify --include_off_the_shelf, --checkpoints, or --baselines.")

    # 3. Generate Visual Proofs
    print(f"\nProcessing {len(image_paths)} images across {len(models_to_test)} models...")

    for img_idx, img_path in enumerate(image_paths):
        raw_pil = Image.open(img_path).convert("RGB")
        img_stem = img_path.stem

        # Create multi-column comparison figure
        num_cols = 1 + len(models_to_test) * 2  # Original + (PCA, Attention) per model
        fig, axes = plt.subplots(1, num_cols, figsize=(4.2 * num_cols, 4.5), squeeze=False)
        axes = axes[0]

        # Column 0: Original Raw Seafloor Image
        resized_pil = raw_pil.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
        axes[0].imshow(resized_pil)
        axes[0].set_title(f"Input Frame\n{img_stem[:18]}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        col_idx = 1
        for model_label, model, norm_mean, norm_std, patch_size in models_to_test:
            # Transform
            t_eval = T.Compose([
                T.Resize((args.image_size, args.image_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                T.ToTensor(),
                T.Normalize(mean=norm_mean, std=norm_std),
            ])
            tensor_in = t_eval(raw_pil).unsqueeze(0)

            # Feature & Attention extraction
            grid_h = args.image_size // patch_size
            grid_w = args.image_size // patch_size
            patch_tokens, attentions = extract_patch_features_and_attentions(model, tensor_in, device)

            # 1. PCA Patch Feature Map
            pca_rgb = compute_patch_pca_rgb(patch_tokens, grid_h, grid_w)
            pca_upsampled = upsample_pca_map(pca_rgb, (args.image_size, args.image_size))

            axes[col_idx].imshow(pca_upsampled)
            axes[col_idx].set_title(f"{model_label}\nPatch PCA (Top-3)", fontsize=10)
            axes[col_idx].axis("off")
            col_idx += 1

            # 2. Attention Rollout Map
            if attentions is not None:
                rollout = compute_attention_rollout(attentions, grid_h, grid_w)
                attn_overlay = overlay_heatmap_on_image(resized_pil, rollout)
                axes[col_idx].imshow(attn_overlay)
                axes[col_idx].set_title(f"{model_label}\n[CLS] Attention Rollout", fontsize=10)
            else:
                axes[col_idx].text(0.5, 0.5, "Attention\nUnavailable", ha="center", va="center")
                axes[col_idx].set_title(f"{model_label}\nAttention", fontsize=10)
            axes[col_idx].axis("off")
            col_idx += 1

        plt.tight_layout()
        save_path = out_dir / f"visual_proof_{img_stem}.png"
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"  [✓] Saved comparison figure: {save_path}")

    print(f"\nAll visual proof figures saved in: {out_dir}")


if __name__ == "__main__":
    main()
