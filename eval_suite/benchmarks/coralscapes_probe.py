"""Linear segmentation probe runner for EPFL Coralscapes."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from eval_suite.datasets.coralscapes import CoralscapesDataset
from eval_suite.models import LinearSegmenter
from eval_suite.utils import seed_worker, set_seed


def evaluate_coralscapes(backbone, train_split, val_split, num_classes: int, device: torch.device, dtype: torch.dtype,
                         batch_size: int = 8, num_workers: int = 4, seed: int = 42, epochs: int = 10,
                         lr: float = 3e-3, width: int = 896, height: int = 448):
    set_seed(seed)
    train_ds = CoralscapesDataset(train_split, width=width, height=height)
    val_ds = CoralscapesDataset(val_split, width=width, height=height)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    out_size = (height, width)
    model = LinearSegmenter(backbone, feat_dim=backbone.config.hidden_size, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_miou, best_pix_acc = 0.0, 0.0
    for ep in range(1, epochs + 1):
        model.head.train()
        for img, target in train_loader:
            img, target = img.to(device=device, dtype=dtype), target.to(device=device)
            optimizer.zero_grad()
            loss = criterion(model(img, out_size=out_size), target)
            loss.backward()
            optimizer.step()

        model.eval()
        conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        with torch.no_grad():
            for img, target in val_loader:
                img = img.to(device=device, dtype=dtype)
                logits = model(img, out_size=out_size)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                mask = (target.numpy() >= 0) & (target.numpy() < num_classes)
                np.add.at(conf_matrix, (target.numpy()[mask], preds[mask]), 1)

        intersection = np.diag(conf_matrix)
        union = conf_matrix.sum(axis=1) + conf_matrix.sum(axis=0) - intersection
        valid = union > 0
        ious = np.zeros(num_classes)
        ious[valid] = (intersection[valid] / union[valid]) * 100.0
        miou = np.mean(ious[valid]) if np.any(valid) else 0.0
        pix_acc = (intersection.sum() / conf_matrix.sum()) * 100.0 if conf_matrix.sum() > 0 else 0.0

        if miou > best_miou:
            best_miou, best_pix_acc = miou, pix_acc

    return {"coralscapes_mIoU": best_miou, "coralscapes_pixel_acc": best_pix_acc}
