"""FathomNet benchmark probe runner (k-NN + Linear Probe with Bicubic Transforms)."""

import gc
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from eval_suite.datasets.fathomnet import FathomNetDataset
from eval_suite.utils import seed_worker, set_seed


def evaluate_fathomnet(
    model: nn.Module,
    train_samples: List[Dict],
    test_samples: List[Dict],
    num_classes: int,
    transform,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
    epochs: int = 20,
    k: int = 20,
    temp: float = 0.07,
) -> Dict[str, float]:
    """Evaluates representation quality on FathomNet via k-NN and Linear Probing.

    Parameters:
      model: ViT backbone (e.g. DINOv3)
      train_samples: List of sample dicts with 'image', 'label', optional 'bbox'
      test_samples: List of sample dicts with 'image', 'label', optional 'bbox'
      num_classes: Number of unique taxonomic organism classes
      transform: Validation transform (e.g., Resize 224 bicubic + normalize)
      device: torch device (e.g. cuda)
      dtype: torch dtype (e.g. bfloat16 or float32)
      batch_size: Extraction batch size
      num_workers: DataLoader workers
      seed: Random seed for determinism
      epochs: Linear probe training epochs (default: 20)
      k: Number of nearest neighbors (default: 20)
      temp: Cosine similarity temperature for k-NN weights (default: 0.07)

    Returns:
      Dict with accuracy and Macro-F1 for both k-NN and Linear Probe,
      plus confusion matrix array.
    """
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)

    train_ds = FathomNetDataset(train_samples, transform=transform)
    test_ds = FathomNetDataset(test_samples, transform=transform)

    loader_kwargs = {"pin_memory": True}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        worker_init_fn=seed_worker, generator=g, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        worker_init_fn=seed_worker, generator=g, **loader_kwargs
    )

    # 1. Feature Extraction (CLS tokens, L2-normalized)
    @torch.no_grad()
    def _extract_features(loader, desc="FathomNet"):
        feats, targets = [], []
        for images, y in tqdm(loader, desc=f"      Extracting {desc}", leave=False):
            images = images.to(device=device, dtype=dtype, non_blocking=True)
            out = model(images)
            f = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out[0][:, 0]
            feats.append(F.normalize(f.float(), dim=-1, p=2).cpu())
            targets.append(y)
        return torch.cat(feats, dim=0), torch.cat(targets, dim=0)

    train_f, train_y = _extract_features(train_loader, desc=f"Train ({len(train_ds):,} crops)")
    test_f, test_y = _extract_features(test_loader, desc=f"Test ({len(test_ds):,} crops)")

    # 2. k-NN Probe Evaluation (Cosine similarity weighted)
    train_f_dev = train_f.to(device)
    train_y_dev = train_y.to(device)
    all_knn_preds = []

    for i in range(0, test_f.size(0), 2048):
        chunk = test_f[i: i + 2048].to(device)
        sim = torch.mm(chunk, train_f_dev.t())
        topk_sim, topk_idx = torch.topk(sim, k=min(k, train_f_dev.size(0)), dim=1)
        topk_lbls = train_y_dev[topk_idx]
        weights = torch.exp(topk_sim / temp)

        probs = torch.zeros(chunk.size(0), num_classes, device=device)
        for c in range(num_classes):
            probs[:, c] = (weights * (topk_lbls == c).float()).sum(dim=1)
        preds = torch.argmax(probs, dim=1)
        all_knn_preds.append(preds.cpu())

    y_pred_knn = torch.cat(all_knn_preds, dim=0).numpy()
    y_true = test_y.numpy()

    knn_acc = (y_pred_knn == y_true).mean() * 100.0
    knn_macro = f1_score(y_true, y_pred_knn, average="macro", zero_division=0) * 100.0
    knn_micro = f1_score(y_true, y_pred_knn, average="micro", zero_division=0) * 100.0

    del train_f_dev, train_y_dev, all_knn_preds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 3. Linear Probe Evaluation
    set_seed(seed)
    feat_dim = train_f.size(1)
    linear = nn.Linear(feat_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    tensor_ds = TensorDataset(train_f, train_y)
    probe_loader = DataLoader(
        tensor_ds, batch_size=256, shuffle=True,
        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed)
    )

    linear.train()
    for _ in range(epochs):
        for x, y in probe_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(linear(x), y)
            loss.backward()
            optimizer.step()

    linear.eval()
    with torch.no_grad():
        test_logits = linear(test_f.to(device))
        lin_preds = torch.argmax(test_logits, dim=-1).cpu().numpy()

    lin_acc = (lin_preds == y_true).mean() * 100.0
    lin_macro = f1_score(y_true, lin_preds, average="macro", zero_division=0) * 100.0
    lin_micro = f1_score(y_true, lin_preds, average="micro", zero_division=0) * 100.0

    cm_norm = confusion_matrix(y_true, lin_preds, labels=list(range(num_classes)), normalize="true") * 100.0

    del train_f, train_y, test_f, test_y, linear, optimizer, probe_loader, tensor_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "fathomnet_knn_acc": knn_acc,
        "fathomnet_knn_macro_f1": knn_macro,
        "fathomnet_knn_micro_f1": knn_micro,
        "fathomnet_lin_acc": lin_acc,
        "fathomnet_lin_macro_f1": lin_macro,
        "fathomnet_lin_micro_f1": lin_micro,
        "confusion_matrix_norm": cm_norm,
        "y_true": y_true,
        "y_pred": lin_preds,
    }
