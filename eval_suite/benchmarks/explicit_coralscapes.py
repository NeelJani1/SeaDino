"""Validation-selected segmentation head followed by a separate final test pass."""
from pathlib import Path
import gc

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_suite.datasets.coralscapes import CoralscapesDataset
from eval_suite.models import LinearSegmenter
from eval_suite.utils import seed_worker, set_seed


def segmentation_metrics(matrix):
    intersection = np.diag(matrix)
    union = matrix.sum(axis=0) + matrix.sum(axis=1) - intersection
    valid = union > 0
    miou = float(np.mean(intersection[valid] / union[valid]) * 100) if valid.any() else 0.
    accuracy = float(intersection.sum() / matrix.sum() * 100) if matrix.sum() else 0.
    return dict(miou=miou, pixel_accuracy=accuracy)


@torch.no_grad()
def score_segmentation(model, loader, num_classes, device, dtype, out_size, description):
    model.eval()
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, labels in tqdm(loader, desc=description, leave=False):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == 'cuda' and dtype == torch.bfloat16):
            logits = model(images.to(device=device, dtype=dtype), out_size=out_size)
        predictions = logits.argmax(dim=1).cpu().numpy()
        labels = labels.numpy()
        valid = (labels >= 0) & (labels < num_classes)
        np.add.at(matrix, (labels[valid], predictions[valid]), 1)
    return segmentation_metrics(matrix)


def evaluate_explicit_coralscapes(backbone, splits, num_classes, device, dtype,
                                 eval_split, epochs, batch_size, num_workers,
                                 seed, mean, std, probe_path, manifest_hash,
                                 width=896, height=448):
    set_seed(seed)
    def loader(part, shuffle=False):
        dataset = CoralscapesDataset(splits[part], width=width, height=height, mean=mean, std=std)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          pin_memory=device.type == 'cuda', worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(seed))
    train, validation = loader('train', True), loader('validation')
    model = LinearSegmenter(backbone, feat_dim=backbone.config.hidden_size, num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=.003, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    best, best_state, best_epoch = None, None, 0
    out_size = (height, width)
    for epoch in range(1, epochs + 1):
        model.head.train()
        for images, labels in tqdm(train, desc=f'      Coralscapes probe {epoch}/{epochs}', leave=False):
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == 'cuda' and dtype == torch.bfloat16):
                logits = model(images.to(device=device, dtype=dtype), out_size=out_size)
                loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            del images, labels, logits, loss
        metrics = score_segmentation(model, validation, num_classes, device, dtype, out_size, '      VALIDATION')
        print(f"      Probe epoch {epoch}/{epochs}: Validation mIoU={metrics['miou']:.2f}%")
        if best is None or metrics['miou'] > best['miou']:
            best, best_epoch = metrics, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.head.state_dict().items()}
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if best_state is None:
        raise ValueError('Coralscapes requires at least one probe epoch')
    model.head.load_state_dict(best_state)
    Path(probe_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(head_state_dict=best_state, best_epoch=best_epoch, selection_split='validation',
                    selection_metric='mIoU', manifest_hash=manifest_hash), probe_path)
    final = best if eval_split == 'validation' else score_segmentation(
        model, loader('test'), num_classes, device, dtype, out_size, '      FINAL TEST (selected head)')
    return dict(miou=final['miou'], pixel_accuracy=final['pixel_accuracy'],
                validation_miou=best['miou'], selected_probe_epoch=best_epoch)
