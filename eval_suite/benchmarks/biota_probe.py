"""Multi-label Biota runner using GPU-accelerated Bicubic interpolation."""

import gc
from sklearn.metrics import average_precision_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from eval_suite.config import IMAGENET_MEAN, IMAGENET_STD
from eval_suite.datasets.biota import BiotaDataset
from eval_suite.utils import gpu_bicubic_preprocess, seed_worker, set_seed


def evaluate_biota(model, train_samples, test_samples, num_classes, device, dtype, batch_size=256, num_workers=8, seed=42):
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    train_ds = BiotaDataset(train_samples, num_classes)
    test_ds = BiotaDataset(test_samples, num_classes)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
        worker_init_fn=seed_worker, generator=g
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=2,
        worker_init_fn=seed_worker, generator=g
    )

    @torch.no_grad()
    def _extract(loader, desc="Biota"):
        feats, targets = [], []
        for images, y in tqdm(loader, desc=f"      Extracting {desc}", leave=False):
            images_gpu = gpu_bicubic_preprocess(images, target_size=(224, 224), mean=IMAGENET_MEAN, std=IMAGENET_STD, device=device, dtype=dtype)
            out = model(images_gpu)
            f = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out[0][:, 0]
            feats.append(F.normalize(f.float(), dim=-1, p=2).cpu())
            targets.append(y)
        return torch.cat(feats, dim=0), torch.cat(targets, dim=0)

    train_f, train_y = _extract(train_loader, desc=f"Train ({len(train_ds):,} imgs)")
    test_f, test_y = _extract(test_loader, desc=f"Test ({len(test_ds):,} imgs)")
    y_true = test_y.numpy()

    del train_loader, test_loader, train_ds, test_ds
    gc.collect()

    # 1. Multi-Label k-NN (k=20)
    train_f_dev, train_y_dev = train_f.to(device), train_y.to(device)
    all_knn_probs = []
    with torch.no_grad():
        for i in range(0, test_f.size(0), 2048):
            chunk = test_f[i: i + 2048].to(device)
            sim = torch.mm(chunk, train_f_dev.t())
            topk_sim, topk_idx = torch.topk(sim, k=min(20, train_f_dev.size(0)), dim=1)
            w = torch.exp(topk_sim / 0.07)
            w = w / w.sum(dim=1, keepdim=True)
            chunk_probs = (train_y_dev[topk_idx] * w.unsqueeze(-1)).sum(dim=1)
            all_knn_probs.append(chunk_probs.cpu())

    knn_probs = torch.cat(all_knn_probs, dim=0).numpy()
    knn_map = average_precision_score(y_true, knn_probs, average="macro") * 100.0
    knn_micro = f1_score(y_true, (knn_probs >= 0.5).astype(int), average="micro", zero_division=0) * 100.0
    knn_macro = f1_score(y_true, (knn_probs >= 0.5).astype(int), average="macro", zero_division=0) * 100.0

    del train_f_dev, train_y_dev, all_knn_probs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 2. Multi-Label Linear Probe
    set_seed(seed)
    linear = nn.Linear(train_f.size(1), num_classes).to(device)
    optimizer = torch.optim.AdamW(linear.parameters(), lr=1e-2, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    tensor_ds = torch.utils.data.TensorDataset(train_f, train_y)
    probe_loader = DataLoader(tensor_ds, batch_size=256, shuffle=True, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed))

    linear.train()
    for _ in range(20):
        for x, y in probe_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(linear(x), y)
            loss.backward()
            optimizer.step()

    linear.eval()
    with torch.no_grad():
        test_logits = linear(test_f.to(device))
        lin_probs = torch.sigmoid(test_logits).cpu().numpy()

    lin_map = average_precision_score(y_true, lin_probs, average="macro") * 100.0
    lin_micro = f1_score(y_true, (lin_probs >= 0.5).astype(int), average="micro", zero_division=0) * 100.0
    lin_macro = f1_score(y_true, (lin_probs >= 0.5).astype(int), average="macro", zero_division=0) * 100.0

    del train_f, train_y, test_f, test_y, linear, optimizer, probe_loader, tensor_ds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "biota_knn_mAP": knn_map, "biota_knn_micro_f1": knn_micro, "biota_knn_macro_f1": knn_macro,
        "biota_lin_mAP": lin_map, "biota_lin_micro_f1": lin_micro, "biota_lin_macro_f1": lin_macro
    }
