#!/usr/bin/env python3
"""
Visualize DinoV3 augmentation pipelines --
all global + local crops for every augmentation mode, plus a plain
(no-augmentation) reference row, and (optionally) a look at what iBOT's
pixel-space masking actually feeds the model.

Usage:
    python visualize_augmentations.py --image /path/to/sample.jpg
    python visualize_augmentations.py --image /path/to/sample.jpg --global_crop_size 256 --local_crop_size 112
    python visualize_augmentations.py --data_dir /path/to/images/ --num_samples 3
    python visualize_augmentations.py --image /path/to/sample.jpg --show_masking

v2 changes from the original:
  - FIX (real bug): the original --image_size flag was passed straight into
    get_transform()'s `image_size` kwarg. In the current training script,
    `image_size` means the optional ONE-TIME RAW pre-resize -- a completely
    different knob from crop output size, which is now `global_crop_size`.
    Passing a "224 or 256" crop-size value into `image_size` meant crops were
    silently NOT rendered at the requested size (global_crop_size stayed at
    get_transform()'s own default of 224 regardless of what you passed), while
    ALSO silently enabling a raw pre-resize you never asked for. This is the
    exact compatibility trap the training script's own docstring warns about --
    this viz script was still on the old semantics. Fixed: --global_crop_size
    and --local_crop_size now map directly to their real get_transform()
    counterparts; --pre_resize is a new, separate, opt-in flag for the actual
    (different) pre-resize behavior, off by default, matching the training
    script.
  - Added marine_aug WITH and WITHOUT the physics augmentations (vignette /
    turbidity / channel attenuation / marine snow) as two separate rows,
    matching the --no_marine_physics_aug ablation flag.
  - Added --show_masking: a separate figure per image showing a global crop
    under iBOT's pixel-space masking two ways -- pixels genuinely zeroed
    (apply_pixel_masking=True, the default) vs. pixels left untouched with
    only an outline marking which patches the loss is computed on
    (--no_pixel_masking) -- using the training script's own real
    MaskingGenerator, not a separate re-implementation.
  - Fixed a dead-code typo (BICICBIC) in the reference-crop resize.
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
    from ssl_train import get_transform, MaskingGenerator
except ImportError:
    try:
        from ssl_training import get_transform, MaskingGenerator
    except ImportError:
        sys.exit(
            "Error: Could not import 'get_transform'/'MaskingGenerator'. Ensure this "
            "script is in the same directory as your DINOv3 training script (e.g. "
            "DinoV3_abalation_V3.py or ssl_training.py)."
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
    BICUBIC = T.InterpolationMode.BICUBIC  # FIX: was a BICICBIC typo (dead code,
                                            # harmlessly caught by hasattr(), but confusing)
    ref_global = T.Compose([
        T.Resize(global_size, interpolation=BICUBIC),
        T.CenterCrop(global_size),
    ])(img)
    ref_local = T.Compose([
        T.Resize(local_size, interpolation=BICUBIC),
        T.CenterCrop(local_size),
    ])(img)
    return np.array(ref_global), np.array(ref_local)


def build_modes():
    """Augmentation configurations to visualize."""
    return {
        "official_dinov3_aug": dict(
            marine_aug=False, underwater_orientation_aug=True, official_dinov3_aug=True,
        ),
        "marine_aug (physics ON)": dict(
            marine_aug=True, marine_physics_aug=True,
            underwater_orientation_aug=True, official_dinov3_aug=False,
        ),
        "marine_aug (physics OFF)": dict(
            marine_aug=True, marine_physics_aug=False,
            underwater_orientation_aug=True, official_dinov3_aug=False,
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
        f"{img_path.name}\nglobal={args.global_crop_size}px local={args.local_crop_size}px"
        f"{f' | pre_resize={args.pre_resize}px' if args.pre_resize else ''} | "
        f"{args.num_global_crops} global + {args.num_local_crops} local crops",
        fontsize=10,
        y=0.98,
    )

    # ---- Row 0: Reference crops (no random augmentations) ----
    global_ref, local_ref = get_reference_crops(img, args.global_crop_size, args.local_crop_size)

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

        # FIX: explicit global_crop_size/local_crop_size/image_size(pre_resize)
        # instead of the old single overloaded `image_size=args.image_size`.
        transform = safe_get_transform(
            global_crop_size=args.global_crop_size,
            local_crop_size=args.local_crop_size,
            image_size=args.pre_resize,
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

    if args.show_masking:
        visualize_masking(img, args, out_dir, img_path.stem)


def visualize_masking(img: Image.Image, args, out_dir: Path, stem: str):
    """Shows what the student ViT actually receives as input under iBOT's
    pixel-space masking, using the training script's REAL MaskingGenerator --
    not a re-implementation -- at a few mask ratios. Three columns: the plain
    crop, pixels genuinely zeroed (apply_pixel_masking=True, the default,
    what the student's forward pass actually sees), and the same crop with
    pixels left untouched but the same positions outlined in red
    (--no_pixel_masking -- this is ALSO exactly what the student's forward
    pass sees; the outline is only for this figure, the model gets no such
    hint). This is the concrete picture behind why --no_pixel_masking makes
    the iBOT objective close to trivial: in the middle column the model has
    to infer the outlined region from context; in the right column it never
    had to, because it could see it the whole time.
    """
    BICUBIC = T.InterpolationMode.BICUBIC
    crop = T.Compose([
        T.Resize(args.global_crop_size, interpolation=BICUBIC),
        T.CenterCrop(args.global_crop_size),
    ])(img)
    crop_arr = np.array(crop)

    ps = args.patch_size
    n_h = n_w = args.global_crop_size // ps
    ratios = [0.15, 0.30, 0.50]

    fig, axes = plt.subplots(len(ratios), 3, figsize=(6.5, 2.2 * len(ratios)), squeeze=False)
    fig.suptitle(
        f"Pixel-space masking (patch_size={ps}, grid={n_h}x{n_w}) -- middle and "
        f"right columns are BOTH what the student's ViT forward pass receives",
        fontsize=9,
    )

    gen = MaskingGenerator(n_h, n_w, max_num_patches=max(1, int(0.5 * n_h * n_w)))
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    col_titles = [
        "crop",
        "apply_pixel_masking=True\n(default -- student's real input)",
        "--no_pixel_masking\n(student's real input; outline is\nfor this figure only, not the model)",
    ]

    for row, ratio in enumerate(ratios):
        num_masking_patches = int(n_h * n_w * ratio)
        patch_mask = gen(num_masking_patches)  # (n_h, n_w) bool
        pixel_mask = np.repeat(np.repeat(patch_mask, ps, axis=0), ps, axis=1)

        axes[row, 0].imshow(crop_arr)
        axes[row, 0].axis("off")

        zeroed = crop_arr.copy()
        zeroed[pixel_mask] = 0
        axes[row, 1].imshow(zeroed)
        axes[row, 1].axis("off")

        overlay = crop_arr.astype(float).copy()
        red = np.array([255.0, 0.0, 0.0])
        overlay[pixel_mask] = 0.55 * overlay[pixel_mask] + 0.45 * red
        axes[row, 2].imshow(overlay.astype(np.uint8))
        axes[row, 2].axis("off")

        if row == 0:
            for c, title in enumerate(col_titles):
                axes[row, c].set_title(title, fontsize=7)

        axes[row, 0].text(
            -0.3, 0.5, f"ratio={ratio:.0%}", transform=axes[row, 0].transAxes,
            ha="right", va="center", fontsize=8, fontweight="bold",
        )

    plt.tight_layout(rect=[0.1, 0.0, 1.0, 0.90])
    out_path = out_dir / f"masking_check_{stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved masking visualization to: {out_path}")


def main():
    p = argparse.ArgumentParser(description="Visualize DinoV3 augmentation crops")
    p.add_argument("--image", type=str, default=None,
                   help="Path to one specific image file to visualize.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Folder to scan recursively; picks --num_samples random images.")
    p.add_argument("--num_samples", type=int, default=3,
                   help="Number of random images to pick when using --data_dir.")
    p.add_argument("--global_crop_size", type=int, default=224,
                   help="Global crop OUTPUT size -- matches the training script's "
                        "--global_crop_size exactly (e.g. 224 default, or 256 for "
                        "official parity). This is NOT the old --image_size.")
    p.add_argument("--local_crop_size", type=int, default=96,
                   help="Local crop OUTPUT size -- matches --local_crop_size. "
                        "Training script default is 96; pass 112 to preview the "
                        "true official-parity pairing (with --global_crop_size 256).")
    p.add_argument("--pre_resize", type=int, default=None,
                   help="Optional raw-image pre-resize (shorter side, bicubic) applied "
                        "before any crop is drawn -- matches the training script's real "
                        "--image_size meaning. Default None (disabled), matching the "
                        "training script's own default. Do NOT use this to mean crop "
                        "size -- see --global_crop_size / --local_crop_size for that.")
    p.add_argument("--num_global_crops", type=int, default=2)
    p.add_argument("--num_local_crops", type=int, default=8)
    p.add_argument("--patch_size", type=int, default=16,
                   help="Model patch size, only used for --show_masking's patch grid "
                        "math. 16 for vits16/vitb16/vitl16.")
    p.add_argument("--show_masking", action="store_true", default=False,
                   help="Also render a separate figure showing iBOT pixel-space masking "
                        "at a few ratios, using the training script's real "
                        "MaskingGenerator.")
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