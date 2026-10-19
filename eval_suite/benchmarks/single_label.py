"""Single-label benchmark probe runner (k-NN + Linear Probe with Bicubic Transforms).

Evaluates representation quality on single-label benchmarks (Substrate Depth 2,
German Bank 2010, etc.) simultaneously via non-parametric k-NN and parametric linear probing.
"""

import gc
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from eval_suite.datasets.single_label import SingleLabelDataset
from eval_suite.utils import seed_worker, set_seed


def evaluate_single_label(
    model: nn.Module,
    train_samples: List[Tuple[str, int]],
    test_samples: List[Tuple[str, int]],
    num_classes: int,
    transform,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int = 256,
    num_workers: int = 8,
    k: int = 20,
    temp: float = 0.07,
    epochs: int = 10,
    seed: int = 42,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> Dict[str, Any]:
    """Evaluates representation quality on single-label classification via both
    k-NN and Linear Probing on frozen backbone features.
    """
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)

    train_ds = SingleLabelDataset(train_samples, transform=transform)
    test_ds = SingleLabelDataset(test_samples, transform=transform)

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
    def _extract(loader, desc="Features"):
        feats, targets = [], []
        for images, y in tqdm(loader, desc=f"      Extracting {desc}", leave=False):
            images = images.to(device=device, dtype=dtype, non_blocking=True)
            try:
                out = model(images, interpolate_pos_encoding=True)
            except TypeError:
                out = model(images)
            f = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out[0][:, 0]
            feats.append(F.normalize(f.float(), dim=-1, p=2).cpu())
            targets.append(y)
        return torch.cat(feats, dim=0), torch.cat(targets, dim=0)

    train_f, train_y = _extract(train_loader, desc=f"Train ({len(train_ds):,} imgs)")
    test_f, test_y = _extract(test_loader, desc=f"Test ({len(test_ds):,} imgs)")

    del train_loader, test_loader, train_ds, test_ds
    gc.collect()

    y_true = test_y.numpy()

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

    knn_acc = (y_pred_knn == y_true).mean() * 100.0
    knn_macro_f1 = f1_score(y_true, y_pred_knn, average="macro", zero_division=0) * 100.0
    knn_micro_f1 = f1_score(y_true, y_pred_knn, average="micro", zero_division=0) * 100.0

    cm_norm_knn = confusion_matrix(y_true, y_pred_knn, labels=list(range(num_classes)), normalize="true") * 100.0
    cm_norm_knn = np.nan_to_num(cm_norm_knn, nan=0.0)

    del train_f_dev, train_y_dev, all_knn_preds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 3. Linear Probe Evaluation (CrossEntropy with balanced weights on frozen CLS features)
    lin_acc, lin_macro_f1, lin_micro_f1 = 0.0, 0.0, 0.0
    lin_preds = y_pred_knn
    cm_norm = cm_norm_knn

    if epochs > 0:
        set_seed(seed)
        feat_dim = train_f.size(1)
        linear = nn.Linear(feat_dim, num_classes).to(device)
        optimizer = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=weight_decay)

        # Class balancing
        class_counts = torch.bincount(train_y, minlength=num_classes).float()
        total_samples = float(len(train_y))
        class_weights = torch.ones(num_classes, dtype=torch.float32)
        for c in range(num_classes):
            if class_counts[c] > 0:
                class_weights[c] = total_samples / (num_classes * class_counts[c])
            else:
                class_weights[c] = 1.0
        class_weights = (class_weights / class_weights.mean()).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        tensor_ds = TensorDataset(train_f, train_y)
        probe_loader = DataLoader(
            tensor_ds, batch_size=min(128, max(16, len(tensor_ds))), shuffle=True,
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
        lin_macro_f1 = f1_score(y_true, lin_preds, average="macro", zero_division=0) * 100.0
        lin_micro_f1 = f1_score(y_true, lin_preds, average="micro", zero_division=0) * 100.0

        cm_norm = confusion_matrix(y_true, lin_preds, labels=list(range(num_classes)), normalize="true") * 100.0
        cm_norm = np.nan_to_num(cm_norm, nan=0.0)

        del linear, optimizer, probe_loader, tensor_ds

    del train_f, train_y, test_f, test_y
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "lin_accuracy": lin_acc,
        "lin_macro_f1": lin_macro_f1,
        "lin_micro_f1": lin_micro_f1,
        "knn_accuracy": knn_acc,
        "knn_macro_f1": knn_macro_f1,
        "knn_micro_f1": knn_micro_f1,
        "accuracy": knn_acc,
        "macro_f1": knn_macro_f1,
        "micro_f1": knn_micro_f1,
        "confusion_matrix_norm": cm_norm,
        "confusion_matrix_norm_knn": cm_norm_knn,
        "y_true": y_true,
        "y_pred": lin_preds,
        "y_pred_knn": y_pred_knn,
        "y_pred_lin": lin_preds,
    }


evaluate_knn = evaluate_single_label
