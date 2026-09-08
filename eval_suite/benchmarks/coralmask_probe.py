"""Linear segmentation probe runner for CoralMask."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from eval_suite.datasets.coralmask import CoralMaskDataset
from eval_suite.models import LinearSegmenter
from eval_suite.utils import seed_worker, set_seed


def evaluate_coralmask(backbone, data_dir: str, device: torch.device, dtype: torch.dtype,
                       batch_size: int = 16, num_workers: int = 4, seed: int = 42, epochs: int = 10,
                       lr: float = 3e-3, width: int = 512, height: int = 512, max_train_samples=None,
                       mean=None, std=None, test_manifest=None):
    set_seed(seed)
    kwargs = {}
    if mean is not None and std is not None:
        kwargs = {"mean": mean, "std": std}
    train_ds = CoralMaskDataset(data_dir, split="train", width=width, height=height, max_samples=max_train_samples, **kwargs)
    test_ds = CoralMaskDataset(data_dir, split="test", width=width, height=height, test_manifest=test_manifest, **kwargs)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    out_size = (height, width)
    model = LinearSegmenter(backbone, feat_dim=backbone.config.hidden_size, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_miou, best_coral_iou, best_acc = 0.0, 0.0, 0.0
    for ep in range(1, epochs + 1):
        model.head.train()
        for img, target in train_loader:
            img, target = img.to(device=device, dtype=dtype), target.to(device=device)
            optimizer.zero_grad()
            loss = criterion(model(img, out_size=out_size), target)
            loss.backward()
            optimizer.step()

        model.eval()
        conf_matrix = np.zeros((2, 2), dtype=np.int64)
        with torch.no_grad():
            for img, target in test_loader:
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

        miou = np.mean(ious)
        coral_iou = ious[1]
        acc = (intersection.sum() / conf_matrix.sum()) * 100.0 if conf_matrix.sum() > 0 else 0.0

        if miou > best_miou:
            best_miou, best_coral_iou, best_acc = miou, coral_iou, acc

    return {"coralmask_mIoU": best_miou, "coralmask_coral_iou": best_coral_iou, "coralmask_pixel_acc": best_acc}
