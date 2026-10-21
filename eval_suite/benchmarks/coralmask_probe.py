"""Linear segmentation probe runner for CoralMask."""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_suite.datasets.coralmask import CoralMaskDataset, prepare_coralmask_splits
from eval_suite.models import LinearSegmenter
from eval_suite.utils import seed_worker, set_seed


def _eval_loader(model, loader, out_size, device, dtype, desc=""):
    model.eval()
    conf_matrix = np.zeros((2, 2), dtype=np.int64)
    with torch.no_grad():
        for img, target in tqdm(loader, desc=desc, leave=False):
            img = img.to(device=device, dtype=dtype)
            logits = model(img, out_size=out_size)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            mask = (target.numpy() >= 0) & (target.numpy() < 2)
            np.add.at(conf_matrix, (target.numpy()[mask], preds[mask]), 1)

    intersection = np.diag(conf_matrix)
    union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - intersection
    valid = union > 0
    ious = np.zeros(2)
    ious[valid] = (intersection[valid] / union[valid]) * 100.0

    miou = float(np.mean(ious))
    coral_iou = float(ious[1])
    acc = float((intersection.sum() / conf_matrix.sum()) * 100.0 if conf_matrix.sum() > 0 else 0.0)
    return miou, coral_iou, acc


def evaluate_coralmask(backbone, data_dir: str, device: torch.device, dtype: torch.dtype,
                       batch_size: int = 16, num_workers: int = 4, seed: int = 42, epochs: int = 10,
                       lr: float = 3e-3, width: int = 512, height: int = 512, max_train_samples=None,
                       mean=None, std=None, test_manifest=None, protocol: str = "legacy",
                       val_ratio: float = 0.2):
    if epochs < 1:
        raise ValueError(f"CoralMask epochs must be >= 1, got {epochs}")

    set_seed(seed)
    kwargs = {}
    if mean is not None and std is not None:
        kwargs = {"mean": mean, "std": std}

    is_heldout = protocol in ("heldout-v1", "coralmask-heldout-v1")

    if is_heldout:
        if val_ratio <= 0.0 or val_ratio >= 1.0 or not np.isfinite(val_ratio):
            raise ValueError(f"CoralMask held-out protocol requires 0 < val_ratio < 1, got {val_ratio}")
        train_samples, val_samples, test_samples, provenance = prepare_coralmask_splits(
            data_dir=data_dir,
            val_ratio=val_ratio,
            split_seed=seed,
            test_manifest=test_manifest,
            max_train_samples=max_train_samples,
        )
        train_ds = CoralMaskDataset(width=width, height=height, samples=train_samples, **kwargs)
        val_ds = CoralMaskDataset(width=width, height=height, samples=val_samples, **kwargs)
        test_ds = CoralMaskDataset(width=width, height=height, samples=test_samples, **kwargs)
    else:
        train_samples, _, test_samples, provenance = prepare_coralmask_splits(
            data_dir=data_dir,
            val_ratio=0.0,
            split_seed=seed,
            test_manifest=test_manifest,
            max_train_samples=max_train_samples,
        )
        train_ds = CoralMaskDataset(width=width, height=height, samples=train_samples, **kwargs)
        val_ds = None
        test_ds = CoralMaskDataset(width=width, height=height, samples=test_samples, **kwargs)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g) if val_ds is not None else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    out_size = (height, width)
    model = LinearSegmenter(backbone, feat_dim=backbone.config.hidden_size, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    if is_heldout:
        print(f"      [CoralMask Held-Out Protocol] Training linear probe ({epochs} epochs, {len(train_ds):,} train samples, {len(val_ds):,} val samples, {len(test_ds):,} test samples)...")
    else:
        print(f"      [CoralMask Legacy Protocol] Training linear probe ({epochs} epochs, {len(train_ds):,} train samples, {len(test_ds):,} test samples)...")

    best_miou = -1.0
    best_coral_iou = 0.0
    best_acc = 0.0
    best_head_state = copy.deepcopy(model.head.state_dict())
    best_epoch = 1

    for ep in range(1, epochs + 1):
        model.head.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"      Epoch {ep:2d}/{epochs} [Train]", leave=False)
        for img, target in pbar:
            img, target = img.to(device=device, dtype=dtype), target.to(device=device)
            optimizer.zero_grad()
            loss = criterion(model(img, out_size=out_size), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        avg_loss = total_loss / max(1, len(train_loader))

        if is_heldout:
            val_miou, val_coral_iou, val_acc = _eval_loader(
                model, val_loader, out_size, device, dtype, desc=f"      Epoch {ep:2d}/{epochs} [Val]"
            )
            if val_miou > best_miou or ep == 1:
                best_miou, best_coral_iou, best_acc = val_miou, val_coral_iou, val_acc
                best_epoch = ep
                best_head_state = copy.deepcopy(model.head.state_dict())
            print(f"      Epoch {ep:2d}/{epochs} - Loss: {avg_loss:.4f} | Val mIoU: {val_miou:.2f}% | Coral-IoU: {val_coral_iou:.2f}%")
        else:
            test_miou, test_coral_iou, test_acc = _eval_loader(
                model, test_loader, out_size, device, dtype, desc=f"      Epoch {ep:2d}/{epochs} [Test Eval]"
            )
            if test_miou > best_miou or ep == 1:
                best_miou, best_coral_iou, best_acc = test_miou, test_coral_iou, test_acc
                best_epoch = ep
            print(f"      Epoch {ep:2d}/{epochs} - Loss: {avg_loss:.4f} | Test mIoU (exploratory legacy selection): {test_miou:.2f}% | Coral-IoU: {test_coral_iou:.2f}%")

    if is_heldout:
        if best_head_state is not None:
            model.head.load_state_dict(best_head_state)
        test_miou, test_coral_iou, test_acc = _eval_loader(
            model, test_loader, out_size, device, dtype, desc="      [Final Test Eval]"
        )
        print(f"      [CoralMask Held-Out Protocol] Selected probe Epoch {best_epoch} (Val mIoU: {best_miou:.2f}%)")
        print(f"      ==> Final Test CoralMask mIoU: {test_miou:.2f}% | Coral-IoU: {test_coral_iou:.2f}%")
        res = {
            "coralmask_mIoU": test_miou,
            "coralmask_coral_iou": test_coral_iou,
            "coralmask_pixel_acc": test_acc,
            "coralmask_val_mIoU": best_miou,
            "coralmask_selected_epoch": best_epoch,
            "coralmask_protocol": "coralmask-heldout-v1",
        }
        res.update(provenance)
        return res
    else:
        print(f"      ==> Best CoralMask mIoU (legacy selection): {best_miou:.2f}% | Coral-IoU: {best_coral_iou:.2f}%")
        return {
            "coralmask_mIoU": best_miou,
            "coralmask_coral_iou": best_coral_iou,
            "coralmask_pixel_acc": best_acc,
            "coralmask_protocol": "legacy",
        }
