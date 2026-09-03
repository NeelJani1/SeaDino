"""Fast utility functions, seed determinism, and GPU Bicubic preprocessing."""

import os
from pathlib import Path
import random
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def fast_resolve_path(root: str, dataset: str, site: str, img_name: str) -> Optional[str]:
    """Fast primary candidate checker with fallbacks."""
    p1 = os.path.join(root, dataset, site, f"{img_name}.jpg")
    if os.path.isfile(p1):
        return p1
    p2 = os.path.join(root, dataset, site, f"{img_name}.JPG")
    if os.path.isfile(p2):
        return p2
    p3 = os.path.join(root, dataset, site, f"{img_name}.png")
    if os.path.isfile(p3):
        return p3
    p4 = os.path.join(root, dataset, site, img_name)
    if os.path.isfile(p4):
        return p4
    return None


def gpu_bicubic_preprocess(
    images: torch.Tensor,
    target_size: Tuple[int, int] = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225],
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Performs Bicubic interpolation (with antialiasing) and normalization directly on CUDA."""
    # 1. Asynchronous transfer of compact uint8 tensors to GPU
    x = images.to(device=device, non_blocking=True).float()

    # 2. CUDA-accelerated Bicubic Interpolation with Antialiasing (<1ms on RTX 4090)
    x = F.interpolate(x, size=target_size, mode="bicubic", align_corners=False, antialias=True)

    # 3. Vectorized GPU Normalization
    mean_t = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    x = (x.div_(255.0).sub_(mean_t)).div_(std_t)

    return x.to(dtype=dtype)


def plot_and_save_confusion_matrix(cm_norm: np.ndarray, class_names: List[str], save_path: str, title: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_norm, annot=True, fmt=".1f", cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"size": 11, "weight": "bold"}, vmin=0, vmax=100
    )
    plt.title(title, fontsize=12, weight="bold", pad=12)
    plt.xlabel("Predicted", fontsize=11, weight="bold")
    plt.ylabel("Ground Truth", fontsize=11, weight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
