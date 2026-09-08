"""High-throughput k-NN runner with Bicubic transforms."""

from sklearn.metrics import confusion_matrix, f1_score
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from eval_suite.datasets.single_label import SingleLabelDataset
from eval_suite.utils import seed_worker


@torch.no_grad()
def evaluate_knn(model, train_samples, test_samples, num_classes, transform, device, dtype, batch_size=256, num_workers=8, k=20, temp=0.07):
    g = torch.Generator().manual_seed(42)
    train_ds = SingleLabelDataset(train_samples, transform=transform)
    test_ds = SingleLabelDataset(test_samples, transform=transform)

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

    def _extract(loader, desc="Features"):
        feats, targets = [], []
        for images, y in tqdm(loader, desc=f"      Extracting {desc}", leave=False):
            images = images.to(device=device, dtype=dtype, non_blocking=True)
            out = model(images)
            f = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out[0][:, 0]
            feats.append(F.normalize(f.float(), dim=-1, p=2).cpu())
            targets.append(y)
        return torch.cat(feats, dim=0), torch.cat(targets, dim=0)

    train_f, train_y = _extract(train_loader, desc=f"Train ({len(train_ds):,} imgs)")
    test_f, test_y = _extract(test_loader, desc=f"Test ({len(test_ds):,} imgs)")

    train_f, train_y = train_f.to(device), train_y.to(device)
    all_preds = []

    for i in range(0, test_f.size(0), 2048):
        chunk = test_f[i: i + 2048].to(device)
        sim = torch.mm(chunk, train_f.t())
        topk_sim, topk_idx = torch.topk(sim, k=min(k, train_f.size(0)), dim=1)
        topk_lbls = train_y[topk_idx]
        weights = torch.exp(topk_sim / temp)

        probs = torch.zeros(chunk.size(0), num_classes, device=device)
        for c in range(num_classes):
            probs[:, c] = (weights * (topk_lbls == c).float()).sum(dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.append(preds.cpu())

    y_pred = torch.cat(all_preds, dim=0).numpy()
    y_true = test_y.numpy()

    acc = (y_pred == y_true).mean() * 100.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0) * 100.0
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0) * 100.0

    cm_norm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)), normalize="true") * 100.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "confusion_matrix_norm": cm_norm,
        "y_true": y_true,
        "y_pred": y_pred,
    }
