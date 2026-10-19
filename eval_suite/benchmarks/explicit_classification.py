"""Frozen-feature probes selected on validation, evaluated once on requested split."""
import copy
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

from eval_suite.datasets.brackish import _parse_yolo_bbox
from eval_suite.datasets.fathomnet import FathomNetDataset
from eval_suite.utils import seed_worker, set_seed
from eval_suite.fathomnet_cache import make_fathomnet_cache


class StrictImages(Dataset):
    """Unreadable records fail visibly; no substitution with a neighbouring image."""
    def __init__(self, records, transform, name, crop_boxes=False, cache=None):
        self.records, self.transform, self.name = records, transform, name
        self.crop_boxes = crop_boxes
        self.fathomnet = FathomNetDataset(records, transform=transform, crop_boxes=crop_boxes, cache=cache)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        item = self.records[index]
        try:
            if self.name == 'fathomnet':
                image = self.fathomnet._load_image(item)
            else:
                with Image.open(item['image']) as raw:
                    raw.load()
                    image = raw.convert('RGB')
                if self.name == 'brackish' and self.crop_boxes and item.get('bbox'):
                    image = image.crop(_parse_yolo_bbox(item['bbox'], *image.size))
            return self.transform(image), int(item['label'])
        except Exception as exc:
            raise RuntimeError(f"Unreadable {self.name} {item['partition']} sample {item['sample_id']}: {item['image']}") from exc


def classification_metrics(labels, predictions, num_classes):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    return dict(accuracy=float(np.mean(labels == predictions) * 100),
                macro_f1=float(f1_score(labels, predictions, labels=list(range(num_classes)), average='macro', zero_division=0) * 100))


@torch.no_grad()
def extract_features(model, dataset, device, dtype, batch_size, num_workers, seed, description, cache=None):
    model.eval()
    started = time.perf_counter()
    feature_path = cache.feature_path(model, dataset.records, dataset.crop_boxes, device, dtype, batch_size, 'explicit') if cache else None
    saved = cache.load_features(feature_path, dataset.records, description) if cache else None
    if saved is not None:
        return saved
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(seed))
    features, labels = [], []
    waiting, previous = 0., time.perf_counter()
    for images, y in tqdm(loader, desc=f'      Extracting {description}', leave=False):
        waiting += time.perf_counter() - previous
        images = images.to(device=device, dtype=dtype, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == 'cuda' and dtype == torch.bfloat16):
            try:
                result = model(images, interpolate_pos_encoding=True)
            except TypeError:
                result = model(images)
        f = result.last_hidden_state[:, 0] if hasattr(result, 'last_hidden_state') else result[0][:, 0]
        features.append(F.normalize(f.float(), dim=-1).cpu())
        labels.append(y.cpu())
        previous = time.perf_counter()
    if not features:
        raise ValueError(f'Empty feature split: {description}')
    result = torch.cat(features), torch.cat(labels)
    if cache:
        cache.save_features(feature_path, *result)
        print(f"      FathomNet {description}: {time.perf_counter()-started:.2f}s total | image/batch wait {waiting:.2f}s")
    return result


@torch.no_grad()
def knn_predictions(train_f, train_y, eval_f, num_classes, device, k=20, temperature=.07):
    reference, labels = train_f.to(device), train_y.to(device)
    result = []
    for query in eval_f.split(2048):
        similarities = query.to(device) @ reference.T
        values, indices = similarities.topk(min(k, len(reference)), dim=1)
        # Shift by each query's maximum for stable exponential weighting.
        weights = ((values - values.max(dim=1, keepdim=True).values) / temperature).exp()
        probabilities = torch.zeros(len(query), num_classes, device=device)
        probabilities.scatter_add_(1, labels[indices], weights)
        result.append(probabilities.argmax(dim=1).cpu())
    return torch.cat(result)


def fit_linear_probe(train_f, train_y, val_f, val_y, num_classes, device,
                     epochs, seed, balanced=True, batch_size=128, lr=.01):
    """Only validation labels select the probe state; test data is not an input."""
    if epochs < 1:
        raise ValueError('Explicit linear probing requires at least one epoch')
    set_seed(seed)
    head = torch.nn.Linear(train_f.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    weights = None
    if balanced:
        counts = torch.bincount(train_y, minlength=num_classes).float()
        weights = torch.where(counts > 0, len(train_y) / (num_classes * counts.clamp_min(1)), torch.ones_like(counts))
        weights = (weights / weights.mean()).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(TensorDataset(train_f, train_y), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best_score, best_state, best_epoch, best_metrics = -1., None, 0, None
    for epoch in range(1, epochs + 1):
        head.train()
        for features, labels in loader:
            optimizer.zero_grad()
            criterion(head(features.to(device)), labels.to(device)).backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            predictions = torch.cat([head(f.to(device)).argmax(dim=1).cpu() for f in val_f.split(2048)])
        metrics = classification_metrics(val_y.numpy(), predictions.numpy(), num_classes)
        print(f"      Probe epoch {epoch}/{epochs}: Validation macro-F1={metrics['macro_f1']:.2f}%")
        if metrics['macro_f1'] > best_score:
            best_score, best_epoch, best_metrics = metrics['macro_f1'], epoch, metrics
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    return head, best_epoch, best_metrics


def evaluate_explicit_classification(model, bundle, transform, device, dtype,
                                    eval_split, epochs, batch_size, num_workers,
                                    seed, probe_path, crop_boxes=False, cache_enabled=True, cache_dir=None):
    nc = len(bundle.class_names)
    cache = make_fathomnet_cache(bundle.records('train') + bundle.records('validation'), transform, cache_dir, cache_enabled) if bundle.name == 'fathomnet' else None
    build = lambda part: StrictImages(bundle.records(part), transform, bundle.name, crop_boxes, cache)
    train_f, train_y = extract_features(model, build('train'), device, dtype, batch_size, num_workers, seed, 'train', cache)
    val_f, val_y = extract_features(model, build('validation'), device, dtype, batch_size, num_workers, seed, 'validation', cache)
    probe_started = time.perf_counter()
    probe_batch = 64 if bundle.name == 'brackish' else 256 if bundle.name == 'fathomnet' else 128
    head, best_epoch, val_linear = fit_linear_probe(train_f, train_y, val_f, val_y, nc, device,
                                                  epochs, seed, bundle.name != 'fathomnet', probe_batch)
    if bundle.name == 'fathomnet':
        print(f"      FathomNet linear probe ({epochs} epochs): {time.perf_counter()-probe_started:.2f}s")
    knn_started = time.perf_counter()
    val_knn_pred = knn_predictions(train_f, train_y, val_f, nc, device)
    val_knn = classification_metrics(val_y.numpy(), val_knn_pred.numpy(), nc)
    if bundle.name == 'fathomnet':
        print(f"      FathomNet validation k-NN: {time.perf_counter()-knn_started:.2f}s")
    Path(probe_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(head_state_dict=head.cpu().state_dict(), best_epoch=best_epoch,
                    selection_split='validation', selection_metric='fixed-taxonomy macro-F1',
                    class_names=bundle.class_names, manifest_hash=bundle.metadata['manifest_hash']), probe_path)
    head.to(device)
    if eval_split == 'validation':
        linear, knn = val_linear, val_knn
    else:
        # Final test features are not even extracted until selection is finished.
        test_f, test_y = extract_features(model, build('test'), device, dtype, batch_size, num_workers, seed, 'FINAL TEST', cache)
        with torch.no_grad():
            predictions = torch.cat([head(f.to(device)).argmax(dim=1).cpu() for f in test_f.split(2048)])
        linear = classification_metrics(test_y.numpy(), predictions.numpy(), nc)
        knn = classification_metrics(test_y.numpy(), knn_predictions(train_f, train_y, test_f, nc, device).numpy(), nc)
    return dict(linear_accuracy=linear['accuracy'], linear_macro_f1=linear['macro_f1'],
                knn_accuracy=knn['accuracy'], knn_macro_f1=knn['macro_f1'],
                validation_linear_macro_f1=val_linear['macro_f1'], validation_knn_macro_f1=val_knn['macro_f1'],
                selected_probe_epoch=best_epoch)
