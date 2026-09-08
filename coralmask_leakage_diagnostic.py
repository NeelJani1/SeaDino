#!/usr/bin/env python3
"""
CoralMask Per-Image Leakage Diagnostic Tool
-------------------------------------------
Computes:
  1. Mean mIoU & Coral-IoU on the 7 leaked images
  2. Mean mIoU & Coral-IoU on the 823 clean images
  3. Difference-in-Differences analysis isolating SSL memorization vs baseline difficulty
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from eval_suite.config import DEFAULT_PATHS, IMAGENET_MEAN, IMAGENET_STD, BENTHIC_MEAN, BENTHIC_STD
from eval_suite.datasets.coralmask import CoralMaskDataset
from eval_suite.models import LinearSegmenter, load_dinov3_backbone
from eval_suite.utils import seed_worker, set_seed


def main():
    p = argparse.ArgumentParser(description="CoralMask Per-Image Leakage Diagnostic")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .ckpt file. Omit for off-the-shelf DINOv3 baseline.")
    p.add_argument("--coralmask_dir", type=str, default=DEFAULT_PATHS["coralmask_dir"])
    p.add_argument("--leaked_ids_file", type=str, default="/home/njan320/SeaDino/coralmask_test_leakage_ids.txt")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=3000,
                   help="Subsample training images for fast diagnostic (default 3000; set 0 for full dataset).")
    p.add_argument("--use_bf16", action="store_true", default=False)
    args = p.parse_args()

    max_samples = None if args.max_train_samples == 0 else args.max_train_samples

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if (args.use_bf16 and torch.cuda.is_available()) else torch.float32

    label = Path(args.checkpoint).parent.name if args.checkpoint else "off-the-shelf (baseline)"
    print("=" * 80)
    print(f"CORALMASK PER-IMAGE LEAKAGE DIAGNOSTIC: {label}")
    print("=" * 80)

    # 1. Load Model & Normalization
    model, epoch, loss, benthic_norm = load_dinov3_backbone(args.checkpoint, DEFAULT_PATHS["model_id"], device, dtype)
    active_mean = BENTHIC_MEAN if benthic_norm else IMAGENET_MEAN
    active_std = BENTHIC_STD if benthic_norm else IMAGENET_STD
    print(f"Model: {label} | Epoch: {epoch} | benthic_norm: {benthic_norm}")

    # 2. Datasets & Loaders
    train_ds = CoralMaskDataset(args.coralmask_dir, split="train", width=512, height=512,
                                max_samples=max_samples, mean=active_mean, std=active_std)
    test_ds = CoralMaskDataset(args.coralmask_dir, split="test", width=512, height=512, mean=active_mean, std=active_std)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    # 3. Train Linear Segmentation Probe
    out_size = (512, 512)
    segmenter = LinearSegmenter(model, feat_dim=model.config.hidden_size, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(segmenter.head.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining Linear Segmentation Probe ({args.epochs} epochs)...")
    for ep in range(1, args.epochs + 1):
        segmenter.head.train()
        total_loss = 0.0
        for img, target in tqdm(train_loader, desc=f"Epoch {ep}/{args.epochs}", leave=False):
            img, target = img.to(device=device, dtype=dtype), target.to(device=device)
            optimizer.zero_grad()
            loss = criterion(segmenter(img, out_size=out_size), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {ep:2d}/{args.epochs} - Loss: {total_loss / len(train_loader):.4f}")

    # 4. Per-Image Test Evaluation
    print("\nEvaluating per-image metrics on 830 test images...")
    leaked_ids = set()
    if Path(args.leaked_ids_file).exists():
        leaked_ids = set(Path(args.leaked_ids_file).read_text().split())

    segmenter.eval()
    per_image_results = []

    with torch.no_grad():
        for idx, (img, target) in enumerate(tqdm(test_loader, desc="Testing")):
            img_path, _ = test_ds.samples[idx]
            stem = Path(img_path).stem

            img = img.to(device=device, dtype=dtype)
            logits = segmenter(img, out_size=out_size)
            preds = torch.argmax(logits, dim=1).cpu().numpy().squeeze(0)
            t_np = target.numpy().squeeze(0)

            # Confusion matrix for this image
            cm = np.zeros((2, 2), dtype=np.int64)
            valid = (t_np >= 0) & (t_np < 2)
            np.add.at(cm, (t_np[valid], preds[valid]), 1)

            inter = np.diag(cm)
            union = cm.sum(axis=1) + cm.sum(axis=0) - inter

            ious = np.zeros(2)
            for c in range(2):
                ious[c] = (inter[c] / union[c]) * 100.0 if union[c] > 0 else 100.0
            miou = np.mean(ious)
            coral_iou = ious[1]

            per_image_results.append({
                "stem": stem,
                "is_leaked": stem in leaked_ids,
                "mIoU": miou,
                "coral_IoU": coral_iou,
            })

    # 5. Diagnostic Aggregation
    leaked_res = [r for r in per_image_results if r["is_leaked"]]
    clean_res = [r for r in per_image_results if not r["is_leaked"]]

    mean_miou_leaked = np.mean([r["mIoU"] for r in leaked_res]) if leaked_res else 0.0
    mean_miou_clean = np.mean([r["mIoU"] for r in clean_res])
    mean_miou_all = np.mean([r["mIoU"] for r in per_image_results])
    delta_miou = mean_miou_all - mean_miou_clean

    mean_coral_leaked = np.mean([r["coral_IoU"] for r in leaked_res]) if leaked_res else 0.0
    mean_coral_clean = np.mean([r["coral_IoU"] for r in clean_res])
    mean_coral_all = np.mean([r["coral_IoU"] for r in per_image_results])
    delta_coral = mean_coral_all - mean_coral_clean

    baseline_gap = 2.92  # 75.36% - 72.44%
    baseline_scores = {
        "GWLCM3PLALP66AYNHK5R": 61.88,
        "MSDUJJ926PAYNA65XT3C": 76.30,
        "O3D5E68I2VGQQO0T591O": 88.68,
        "VXJU92AF544PI700TIBJ": 69.99,
        "W88K8UUDL7U2NZ5K5FOK": 79.73,
        "YK5HW6CLMARBKK3L3XWG": 72.98,
        "ZSKJY6EMVB3AW163AJXF": 77.99,
    }

    gap_model = mean_miou_leaked - mean_miou_clean
    excess_gap = gap_model - baseline_gap

    print("\n" + "=" * 80)
    print("PER-IMAGE LEAKAGE DIAGNOSTIC RESULTS")
    print("=" * 80)
    print(f"Model Evaluated:                   {label}")
    print(f"1. Leaked Images (N = {len(leaked_res)}):          mIoU = {mean_miou_leaked:.2f}% | Coral-IoU = {mean_coral_leaked:.2f}%")
    print(f"2. Clean Images  (N = {len(clean_res)}):        mIoU = {mean_miou_clean:.2f}% | Coral-IoU = {mean_coral_clean:.2f}%")
    print(f"3. All Images    (N = {len(per_image_results)}):        mIoU = {mean_miou_all:.2f}% | Coral-IoU = {mean_coral_all:.2f}%")
    print("-" * 80)
    print(f"Model Gap (Leaked - Clean):        {gap_model:+.2f} pt")
    if args.checkpoint:
        print(f"Baseline Gap Control (N=7 - N=823): +{baseline_gap:.2f} pt")
        print(f"Excess Gap (Isolating SSL Leakage): {excess_gap:+.2f} pt")
    print(f"Net Aggregate Pull (All vs Clean): {delta_miou:+.3f} pt (diluted by 7/830 ratio)")
    print("=" * 80)

    print("\nPer-Image Breakdown for Leaked Stems:")
    for r in leaked_res:
        base_val = baseline_scores.get(r["stem"], None)
        diff_str = f" (Δ vs Baseline: {r['mIoU'] - base_val:+.2f} pt)" if (args.checkpoint and base_val is not None) else ""
        print(f"  {r['stem']}: mIoU = {r['mIoU']:.2f}% | Coral-IoU = {r['coral_IoU']:.2f}%{diff_str}")


if __name__ == "__main__":
    main()
