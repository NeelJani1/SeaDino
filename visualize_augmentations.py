#!/usr/bin/env python3
"""
Visualize DinoV3 augmentation pipelines --
all global + local crops for every augmentation mode, plus a plain
(no-augmentation) reference row.

Usage:
    python visualize_augmentations.py --image /path/to/sample.jpg --image_size 256
    python visualize_augmentations.py --data_dir /path/to/images/ --num_samples 3 --image_size 256
"""

import argparse
import inspect
import random
import sys
from pathlib import Path

# Use headless Agg backend to silence Qt/Wayland plugin warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# Robust import: try custom filename first, then standard training script name
try:
    from DinoV3_abalation_V3_clean import get_transform
except ImportError:
    try:
        from ssl_training import get_transform
    except ImportError:
        sys.exit(
            "Error: Could not import 'get_transform'. Ensure this script is in the "
            "same directory as your DINOv3 training script (e.g. DinoV3_abalation_V3_clean.py "
            "or ssl_training.py)."
        )

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def safe_get_transform(**kwargs):
    """Filters arguments to only pass parameters accepted by get_transform()."""
    sig = inspect.signature(get_transform)
    accepts_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_kwargs:
        return get_transform(**kwargs)

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return get_transform(**filtered_kwargs)


def to_display(tensor: torch.Tensor) -> np.ndarray:
    """[3, H, W] normalized float tensor -> un-normalize -> [H, W, 3] uint8."""
    tensor = tensor.clone().detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    arr = tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return (arr * 255).astype(np.uint8)


def get_reference_crops(img: Image.Image, global_size: int, local_size: int):
    """Plain resize + center-crop baseline without random augmentations."""
    ref_global = T.Compose([
        T.Resize(global_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(global_size),
    ])(img)
    ref_local = T.Compose([
        T.Resize(local_size, interpolation=T.InterpolationMode.BICICBIC if hasattr(T.InterpolationMode, 'BICICBIC') else T.InterpolationMode.BICUBIC),
        T.CenterCrop(local_size),
    ])(img)
    return np.array(ref_global), np.array(ref_local)


def build_modes():
    """Augmentation configurations to visualize."""
    return {
        "official_dinov3_aug": dict(
            marine_aug=False, underwater_orientation_aug=True, official_dinov3_aug=True,
        ),
        "marine_aug": dict(
            marine_aug=True, underwater_orientation_aug=True, official_dinov3_aug=False,
        ),
        "default (benthic)": dict(
            marine_aug=False, underwater_orientation_aug=True, official_dinov3_aug=False,
        ),
        "default, no orientation aug": dict(
            marine_aug=False, underwater_orientation_aug=False, official_dinov3_aug=False,
        ),
    }


def visualize_one_image(img_path: Path, args, out_dir: Path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"⚠️  Could not load image {img_path}: {e}")
        return

    modes = build_modes()
    n_cols = args.num_global_crops + args.num_local_crops
    n_rows = 1 + len(modes)  # 1 reference row + 1 row per aug mode

    # squeeze=False prevents 1D indexing bugs if n_cols or n_rows == 1
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.7 * max(n_cols, 2), 1.7 * n_rows),
        squeeze=False
    )
    fig.suptitle(
        f"{img_path.name}\nimage_size={args.image_size} | "
        f"{args.num_global_crops} global + {args.num_local_crops} local crops",
        fontsize=10,
        y=0.98,
    )

    # ---- Row 0: Reference crops (no random augmentations) ----
    local_ref_size = 112 if args.image_size == 256 else 96
    global_ref, local_ref = get_reference_crops(img, args.image_size, local_ref_size)

    for c in range(n_cols):
        axes[0, c].axis("off")

    if n_cols > 0:
        axes[0, 0].imshow(global_ref)
        axes[0, 0].set_title("ref global", fontsize=7)
        axes[0, 0].text(
            -0.15, 0.5, "no_aug", transform=axes[0, 0].transAxes,
            ha="right", va="center", fontsize=8, fontweight="bold",
        )
    if n_cols > 1:
        axes[0, 1].imshow(local_ref)
        axes[0, 1].set_title("ref local", fontsize=7)

    # ---- Rows 1..N: Real augmentation modes ----
    for row_idx, (label, kwargs) in enumerate(modes.items(), start=1):
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)

        transform = safe_get_transform(
            image_size=args.image_size,
            num_global_crops=args.num_global_crops,
            num_local_crops=args.num_local_crops,
            **kwargs,
        )
        crops = transform(img)  # List of [3, H, W] tensors

        for col_idx, crop in enumerate(crops):
            if col_idx >= n_cols:
                break
            ax = axes[row_idx, col_idx]
            ax.imshow(to_display(crop))
            ax.axis("off")
            is_global = col_idx < args.num_global_crops
            idx = col_idx if is_global else col_idx - args.num_global_crops
            ax.set_title(f"{'G' if is_global else 'L'}{idx} ({crop.shape[-1]}px)", fontsize=6)

        # Turn off any remaining unused subplots in this row
        for col_idx in range(len(crops), n_cols):
            axes[row_idx, col_idx].axis("off")

        axes[row_idx, 0].text(
            -0.15, 0.5, label, transform=axes[row_idx, 0].transAxes,
            ha="right", va="center", fontsize=8, fontweight="bold",
        )

    plt.tight_layout(rect=[0.08, 0.0, 1.0, 0.95])
    out_path = out_dir / f"aug_check_{img_path.stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved visualization to: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Visualize DinoV3 augmentation crops")
    p.add_argument("--image", type=str, default=None,
                   help="Path to one specific image file to visualize.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Folder to scan recursively; picks --num_samples random images.")
    p.add_argument("--num_samples", type=int, default=3,
                   help="Number of random images to pick when using --data_dir.")
    p.add_argument("--image_size", type=int, default=224,
                   help="Global crop size (e.g. 224 or 256).")
    p.add_argument("--num_global_crops", type=int, default=2)
    p.add_argument("--num_local_crops", type=int, default=8)
    p.add_argument("--output_dir", type=str, default="./aug_check")
    p.add_argument("--seed", type=int, default=None,
                   help="Seed for reproducible crops across visualization runs.")
    args = p.parse_args()

    if not args.image and not args.data_dir:
        p.error("Pass either --image or --data_dir")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        paths = [Path(args.image)]
    else:
        data_dir = Path(args.data_dir)
        all_paths = sorted({p for ext in IMAGE_EXTS for p in data_dir.rglob(f"*{ext}") if p.is_file()})
        if not all_paths:
            sys.exit(f"No images found matching {IMAGE_EXTS} under {data_dir}")
        rng = random.Random(args.seed)
        paths = rng.sample(all_paths, min(args.num_samples, len(all_paths)))

    for path in paths:
        visualize_one_image(path, args, out_dir)


if __name__ == "__main__":
    main()