"""Dataset loader for FathomNet benthic organism classification benchmark.

Supports loading from Hugging Face Hub (e.g., FathomNet/2024-vme-benchmark,
FathomNet out-of-sample challenges) as well as local CSVs and image directories.
Handles bounding-box organism crops, full-frame organism classification, and
hierarchical taxonomic concepts with deterministic label mappings.
"""

import ast
import io
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from eval_suite.config import IMAGENET_MEAN, IMAGENET_STD


def _parse_bbox_xyxy(raw_bbox: Union[List, Tuple], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Converts bounding box to valid integer pixel coordinates (xmin, ymin, xmax, ymax).

    Supports:
      - COCO format: [x, y, width, height]
      - VOC format: [xmin, ymin, xmax, ymax]
      - Normalized coordinates in [0, 1]
    """
    if len(raw_bbox) != 4:
        return 0, 0, img_w, img_h

    b0, b1, b2, b3 = [float(x) for x in raw_bbox]

    # Check if normalized coordinates in [0, 1]
    if max(b0, b1, b2, b3) <= 1.0 and img_w > 1 and img_h > 1:
        b0 *= img_w
        b1 *= img_h
        b2 *= img_w
        b3 *= img_h

    # Differentiate [x, y, w, h] vs [xmin, ymin, xmax, ymax]
    # In standard COCO, b2 and b3 are width and height.
    if (b0 + b2) <= (img_w + 5) and (b1 + b3) <= (img_h + 5) and b2 > 0 and b3 > 0:
        xmin = int(round(b0))
        ymin = int(round(b1))
        xmax = int(round(b0 + b2))
        ymax = int(round(b1 + b3))
    else:
        xmin = int(round(min(b0, b2)))
        ymin = int(round(min(b1, b3)))
        xmax = int(round(max(b0, b2)))
        ymax = int(round(max(b1, b3)))

    # Clamp to valid image boundaries
    xmin = max(0, min(img_w - 1, xmin))
    ymin = max(0, min(img_h - 1, ymin))
    xmax = max(xmin + 1, min(img_w, xmax))
    ymax = max(ymin + 1, min(img_h, ymax))

    return xmin, ymin, xmax, ymax


class FathomNetDataset(Dataset):
    """PyTorch Dataset for FathomNet organism recognition.

    Each item in `samples` is a dict or tuple containing:
      - 'image': PIL Image, image file path, or bytes
      - 'bbox': optional bounding box (xmin, ymin, xmax, ymax) or [x, y, w, h]
      - 'label': integer class ID
    """

    def __init__(self, samples: List[Dict[str, Any]], transform=None, crop_margin: float = 0.05):
        self.samples = samples
        self.transform = transform
        self.crop_margin = crop_margin

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, item: Dict[str, Any]) -> Image.Image:
        raw = item.get("image")
        if isinstance(raw, Image.Image):
            img = raw.convert("RGB")
        elif isinstance(raw, (str, Path)):
            with Image.open(raw) as im:
                img = im.convert("RGB")
        elif isinstance(raw, (bytes, bytearray)):
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        elif isinstance(raw, dict) and "bytes" in raw and raw["bytes"]:
            img = Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
        elif isinstance(raw, dict) and "path" in raw and raw["path"]:
            with Image.open(raw["path"]) as im:
                img = im.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(raw)}")

        # Crop bounding box if specified
        bbox = item.get("bbox")
        if bbox is not None:
            w, h = img.size
            xmin, ymin, xmax, ymax = _parse_bbox_xyxy(bbox, w, h)
            if self.crop_margin > 0:
                bw = xmax - xmin
                bh = ymax - ymin
                pad_x = int(bw * self.crop_margin)
                pad_y = int(bh * self.crop_margin)
                xmin = max(0, xmin - pad_x)
                ymin = max(0, ymin - pad_y)
                xmax = min(w, xmax + pad_x)
                ymax = min(h, ymax + pad_y)
            if xmax > xmin and ymax > ymin:
                img = img.crop((xmin, ymin, xmax, ymax))

        return img

    def __getitem__(self, idx: int, retries: int = 0) -> Tuple[torch.Tensor, int]:
        if retries >= 10:
            raise RuntimeError(f"Failed to fetch valid FathomNet sample after {retries} retries starting at {idx}.")

        item = self.samples[idx]
        try:
            img = self._load_image(item)
            label = int(item["label"])
        except Exception:
            # Fault-tolerant adjacent fallback
            return self.__getitem__((idx + 1) % len(self.samples), retries=retries + 1)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)

        return img, label


def _extract_label_from_row(row: Any) -> Optional[str]:
    """Heuristically extracts class label / concept from a dataset record."""
    candidates = ["concept", "category", "label", "taxa", "taxon", "name", "category_name", "class_name"]
    for c in candidates:
        if c in row and row[c] is not None:
            val = row[c]
            if isinstance(val, (list, tuple)) and len(val) > 0:
                return str(val[0])
            return str(val)
    return None


def prepare_fathomnet_data(
    repo_or_dir: str = "FathomNet/2024-vme-benchmark",
    split_train: str = "train",
    split_test: str = "validation",
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    min_samples_per_class: int = 5,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, List[str]]:
    """Loads and standardizes FathomNet data from Hugging Face Hub or a local path.

    Returns:
      train_samples, test_samples, num_classes, class_names
    """
    random.seed(seed)
    train_records = []
    test_records = []

    # 1. Check if local directory with CSV or images
    local_path = Path(repo_or_dir)
    if local_path.is_file() and local_path.suffix.lower() == ".csv":
        import pandas as pd
        df = pd.read_csv(local_path)
        img_col = next((c for c in ["image_path", "image", "file_name", "filepath"] if c in df.columns), None)
        lbl_col = next((c for c in ["concept", "label", "category", "class"] if c in df.columns), None)
        split_col = next((c for c in ["split", "partition"] if c in df.columns), None)
        bbox_col = next((c for c in ["bbox", "bounding_box"] if c in df.columns), None)

        if img_col is None or lbl_col is None:
            raise ValueError(f"Local CSV {local_path} must have image and label columns.")

        for _, r in df.iterrows():
            rec = {
                "image": str(r[img_col]),
                "raw_label": str(r[lbl_col]),
                "bbox": ast.literal_eval(str(r[bbox_col])) if bbox_col and pd.notna(r[bbox_col]) else None,
            }
            part = str(r[split_col]).lower() if split_col and pd.notna(r[split_col]) else "train"
            if part in ["train", "training"]:
                train_records.append(rec)
            else:
                test_records.append(rec)

    elif local_path.is_dir() and (local_path / "metadata.csv").exists():
        import pandas as pd
        df = pd.read_csv(local_path / "metadata.csv")
        img_col = next((c for c in ["file_name", "image", "image_path"] if c in df.columns), "file_name")
        lbl_col = next((c for c in ["label", "concept", "category"] if c in df.columns), "label")
        split_col = next((c for c in ["split", "partition"] if c in df.columns), None)
        for _, r in df.iterrows():
            img_file = local_path / str(r[img_col])
            rec = {"image": str(img_file), "raw_label": str(r[lbl_col]), "bbox": None}
            part = str(r[split_col]).lower() if split_col and pd.notna(r[split_col]) else "train"
            if part in ["train", "training"]:
                train_records.append(rec)
            else:
                test_records.append(rec)

    else:
        # Load from Hugging Face Hub
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Hugging Face datasets library is required. Install via `pip install datasets`.")

        print(f"  [FathomNet] Loading dataset '{repo_or_dir}' from Hugging Face Hub...")
        try:
            ds_dict = load_dataset(repo_or_dir)
        except Exception as e:
            # Fallback attempt with specific configuration or revision
            print(f"  [FathomNet] Direct load error: {e}. Attempting fallback...")
            ds_dict = load_dataset(repo_or_dir, trust_remote_code=True)

        available_splits = list(ds_dict.keys())
        print(f"  [FathomNet] Available splits: {available_splits}")

        # Map requested splits to available splits
        chosen_train = split_train if split_train in available_splits else available_splits[0]
        test_candidates = [split_test, "validation", "val", "test"]
        chosen_test = next((s for s in test_candidates if s in available_splits and s != chosen_train), None)

        def _parse_hf_split(hf_split, out_list):
            for item in hf_split:
                img = item.get("image")
                # Check for nested object annotations (COCO-style detection)
                if "objects" in item and isinstance(item["objects"], dict):
                    objs = item["objects"]
                    bboxes = objs.get("bbox", [])
                    cats = objs.get("category", objs.get("label", objs.get("concept", [])))
                    if len(bboxes) > 0 and len(cats) > 0:
                        for b, c in zip(bboxes, cats):
                            out_list.append({"image": img, "bbox": b, "raw_label": str(c)})
                        continue

                # Standard classification crop
                lbl = _extract_label_from_row(item)
                if lbl is not None:
                    out_list.append({"image": img, "bbox": item.get("bbox"), "raw_label": lbl})

        _parse_hf_split(ds_dict[chosen_train], train_records)

        if chosen_test is not None:
            _parse_hf_split(ds_dict[chosen_test], test_records)
        else:
            # Split chosen_train 80/20 deterministically
            print("  [FathomNet] Single split detected; creating deterministic 80/20 train/test split...")
            random.seed(seed)
            shuffled = list(train_records)
            random.shuffle(shuffled)
            split_idx = int(0.8 * len(shuffled))
            train_records = shuffled[:split_idx]
            test_records = shuffled[split_idx:]

    # 2. Build class taxonomy mapping
    from collections import Counter
    train_counts = Counter(r["raw_label"] for r in train_records)
    valid_classes = {cls for cls, cnt in train_counts.items() if cnt >= min_samples_per_class}

    if not valid_classes:
        valid_classes = set(train_counts.keys())

    class_names = sorted(list(valid_classes))
    label_to_id = {cls_name: i for i, cls_name in enumerate(class_names)}
    num_classes = len(class_names)

    # 3. Format final samples with integer labels
    def _finalize(records):
        final = []
        for r in records:
            lbl = r["raw_label"]
            if lbl in label_to_id:
                final.append({
                    "image": r["image"],
                    "bbox": r.get("bbox"),
                    "label": label_to_id[lbl],
                })
        return final

    final_train = _finalize(train_records)
    final_test = _finalize(test_records)

    # 4. Apply sample limits if requested
    if max_train_samples is not None and len(final_train) > max_train_samples:
        random.seed(seed)
        final_train = random.sample(final_train, max_train_samples)

    if max_test_samples is not None and len(final_test) > max_test_samples:
        random.seed(seed)
        final_test = random.sample(final_test, max_test_samples)

    print(f"  [FathomNet] Ready: {len(final_train):,} train samples, {len(final_test):,} test samples, {num_classes} classes.")
    return final_train, final_test, num_classes, class_names
