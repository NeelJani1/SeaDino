"""Dataset loader for Brackish underwater marine fauna classification benchmark.

Reference: Brackish Underwater Dataset (Pedersen et al., 2019).
Supports 6 marine categories in turbid European waters:
  crab, fish, jellyfish, shrimp, small_fish, starfish.
"""

import ast
from collections import Counter
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from eval_suite.config import BRACKISH_CLASS_NAMES


def _parse_yolo_bbox(
    raw_bbox: Union[List, Tuple, str],
    img_w: int,
    img_h: int,
    margin: float = 0.15,
) -> Tuple[int, int, int, int]:
    """Converts YOLO normalized format [xc, yc, w, h] to pixel (xmin, ymin, xmax, ymax).

    If raw_bbox is already in xyxy or pixel coordinates, gracefully detects and converts.
    Applies contextual margin padding around the organism while clamping to image boundaries.
    """
    if isinstance(raw_bbox, str):
        try:
            raw_bbox = ast.literal_eval(raw_bbox)
        except Exception:
            return 0, 0, img_w, img_h

    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) < 4:
        return 0, 0, img_w, img_h

    try:
        b0, b1, b2, b3 = [float(x) for x in raw_bbox[:4]]
    except (ValueError, TypeError):
        return 0, 0, img_w, img_h

    # Detect normalized YOLO [xc, yc, w, h] in [0, 1]
    if max(b0, b1, b2, b3) <= 1.05:
        bw = b2 * img_w
        bh = b3 * img_h
        xc = b0 * img_w
        yc = b1 * img_h
        xmin = xc - (bw / 2.0)
        ymin = yc - (bh / 2.0)
        xmax = xc + (bw / 2.0)
        ymax = yc + (bh / 2.0)
    else:
        # Check if already pixel coordinates (either xywh or xyxy)
        if (b0 + b2) <= (img_w + 10) and (b1 + b3) <= (img_h + 10) and b2 > 0 and b3 > 0:
            xmin, ymin, xmax, ymax = b0, b1, b0 + b2, b1 + b3
        else:
            xmin, ymin, xmax, ymax = min(b0, b2), min(b1, b3), max(b0, b2), max(b1, b3)

    if margin > 0.0:
        bw = xmax - xmin
        bh = ymax - ymin
        pad_x = bw * margin
        pad_y = bh * margin
        xmin -= pad_x
        ymin -= pad_y
        xmax += pad_x
        ymax += pad_y

    xmin = max(0, min(img_w - 1, int(round(xmin))))
    ymin = max(0, min(img_h - 1, int(round(ymin))))
    xmax = max(xmin + 1, min(img_w, int(round(xmax))))
    ymax = max(ymin + 1, min(img_h, int(round(ymax))))

    return xmin, ymin, xmax, ymax


class BrackishDataset(Dataset):
    """PyTorch Dataset for Brackish multi-class marine fauna classification."""

    def __init__(
        self,
        samples: Union[List[Tuple[str, int]], List[Dict[str, Any]]],
        transform=None,
        crop_boxes: bool = False,
        crop_margin: float = 0.15,
    ):
        self.samples = samples
        self.transform = transform
        self.crop_boxes = crop_boxes
        self.crop_margin = crop_margin

    def __len__(self) -> int:
        return len(self.samples)

    def _unpack_sample(self, item: Any) -> Tuple[str, int, Optional[Any]]:
        if isinstance(item, dict):
            img_path = str(item.get("image") or item.get("file_name") or item.get("path") or "")
            label = int(item.get("label", item.get("cid", 0)))
            bbox = item.get("bbox", None)
            return img_path, label, bbox
        elif isinstance(item, (list, tuple)):
            if len(item) >= 3:
                return str(item[0]), int(item[1]), item[2]
            return str(item[0]), int(item[1]), None
        else:
            raise TypeError(f"Unsupported sample format: {type(item)}")

    def __getitem__(self, idx: int, retries: int = 0) -> Tuple[torch.Tensor, int]:
        if retries >= 10:
            raise RuntimeError(f"Failed to fetch valid Brackish sample after {retries} retries starting at {idx}.")

        item = self.samples[idx]
        try:
            img_path, label, bbox = self._unpack_sample(item)
            with Image.open(img_path) as im:
                im.load()
                img = im.convert("RGB")

            if self.crop_boxes and bbox is not None:
                w, h = img.size
                xmin, ymin, xmax, ymax = _parse_yolo_bbox(bbox, w, h, margin=self.crop_margin)
                if xmax > xmin and ymax > ymin:
                    img = img.crop((xmin, ymin, xmax, ymax))
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples), retries=retries + 1)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)

        return img, int(label)


def prepare_brackish_data(
    data_dir: str,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, List[str]]:
    """Loads and standardizes Brackish classification dataset.

    Supports:
      1. Standard split folders: data_dir/train/<class>/*.jpg, data_dir/test/<class>/*.jpg
      2. Flat structure with metadata.csv or annotations.csv (with YOLO bounding box support)
      3. YOLO layouts: data_dir/{train,valid,test}/images & labels
      4. Unsplit class folders (split deterministically 80/20)

    Returns:
      train_samples, test_samples, num_classes=6, class_names=BRACKISH_CLASS_NAMES
    """
    root = Path(data_dir)
    if (root / "data").is_dir() and not (root / "train").is_dir() and not (root / "metadata.csv").is_file():
        root = root / "data"
    class_names = list(BRACKISH_CLASS_NAMES)
    label_to_id = {cls_name.lower(): idx for idx, cls_name in enumerate(class_names)}
    num_classes = len(class_names)

    train_records: List[Dict[str, Any]] = []
    test_records: List[Dict[str, Any]] = []

    # 1. Check for metadata.csv
    if (root / "metadata.csv").is_file():
        df = pd.read_csv(root / "metadata.csv")
        img_col = next((c for c in ["file_name", "image", "image_path", "filename"] if c in df.columns), df.columns[0])
        lbl_col = next((c for c in ["label", "class", "category", "cid"] if c in df.columns), df.columns[1])
        split_col = next((c for c in ["split", "partition"] if c in df.columns), None)
        bbox_col = next((c for c in ["bbox", "bounding_box", "box"] if c in df.columns), None)

        has_train_split = False
        if split_col:
            has_train_split = df[split_col].astype(str).str.lower().isin(["train", "training"]).any()

        for _, r in df.iterrows():
            cname = str(r[lbl_col]).strip().lower()
            if cname in label_to_id:
                cid = label_to_id[cname]
            else:
                try:
                    cid_cand = int(r[lbl_col])
                    if 0 <= cid_cand < num_classes:
                        cid = cid_cand
                    else:
                        continue
                except (ValueError, TypeError):
                    continue

            img_rel = str(r[img_col])
            p = root / img_rel
            if not p.exists():
                if (root / "data" / img_rel).exists():
                    p = root / "data" / img_rel
                elif Path(img_rel).exists():
                    p = Path(img_rel)

            bbox = None
            if bbox_col and pd.notna(r[bbox_col]) and str(r[bbox_col]).strip():
                try:
                    val = r[bbox_col]
                    if isinstance(val, str):
                        bbox = ast.literal_eval(val)
                    elif isinstance(val, (list, tuple)):
                        bbox = list(val)
                except Exception:
                    bbox = None

            split_val = str(r[split_col]).lower() if split_col and pd.notna(r[split_col]) else "train"
            rec = {
                "image": str(p.resolve()) if p.exists() else str(p),
                "label": cid,
                "bbox": bbox,
            }
            if has_train_split:
                if split_val in ["train", "training"]:
                    train_records.append(rec)
                else:
                    test_records.append(rec)
            else:
                # If no explicit train split, treat valid/val as train, test as test
                if split_val in ["valid", "val"]:
                    train_records.append(rec)
                else:
                    test_records.append(rec)

        if not train_records and test_records:
            random.seed(seed)
            random.shuffle(test_records)
            split_idx = max(1, int(0.8 * len(test_records)))
            train_records = test_records[:split_idx]
            test_records = test_records[split_idx:]
        elif train_records and not test_records:
            random.seed(seed)
            random.shuffle(train_records)
            split_idx = max(1, int(0.8 * len(train_records)))
            test_records = train_records[split_idx:]
            train_records = train_records[:split_idx]

    # 2. Check for YOLO layout: any of train, valid, test containing images + labels
    elif any((root / s / "images").is_dir() and (root / s / "labels").is_dir() for s in ["train", "valid", "val", "test"]):
        def _scan_yolo(split_dir: Path) -> List[Dict[str, Any]]:
            recs = []
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"
            if not img_dir.is_dir() or not lbl_dir.is_dir():
                return recs
            for lbl_f in sorted(lbl_dir.glob("*.txt")):
                img_cand = next((img_dir / f"{lbl_f.stem}{ext}" for ext in [".jpg", ".png", ".jpeg"] if (img_dir / f"{lbl_f.stem}{ext}").is_file()), None)
                if not img_cand:
                    continue
                try:
                    with open(lbl_f, "r") as f:
                        lines = [l.strip() for l in f if l.strip()]
                    for line in lines:
                        parts = line.split()
                        if not parts:
                            continue
                        cid = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]] if len(parts) >= 5 else None
                        if 0 <= cid < num_classes:
                            recs.append({
                                "image": str(img_cand.resolve()),
                                "label": cid,
                                "bbox": bbox,
                            })
                except Exception:
                    continue
            return recs

        train_dir = root / "train" if (root / "train" / "images").is_dir() else None
        val_dir = root / "valid" if (root / "valid" / "images").is_dir() else (root / "val" if (root / "val" / "images").is_dir() else None)
        test_dir = root / "test" if (root / "test" / "images").is_dir() else None

        if train_dir and test_dir:
            train_records = _scan_yolo(train_dir)
            test_records = _scan_yolo(test_dir)
        elif train_dir and val_dir:
            train_records = _scan_yolo(train_dir)
            test_records = _scan_yolo(val_dir)
        elif val_dir and test_dir:
            train_records = _scan_yolo(val_dir)
            test_records = _scan_yolo(test_dir)
        else:
            avail_dir = train_dir or val_dir or test_dir
            all_recs = _scan_yolo(avail_dir)
            random.seed(seed)
            random.shuffle(all_recs)
            split_idx = max(1, int(0.8 * len(all_recs)))
            train_records = all_recs[:split_idx]
            test_records = all_recs[split_idx:]

    # 3. Check for class-name subdirectories: train/<class>/*.jpg
    elif (root / "train").is_dir() and ((root / "val").is_dir() or (root / "test").is_dir()):
        def _scan_folder(folder: Path) -> List[Dict[str, Any]]:
            recs = []
            for cls_folder in folder.iterdir():
                if not cls_folder.is_dir():
                    continue
                cname = cls_folder.name.lower()
                if cname not in label_to_id:
                    continue
                cid = label_to_id[cname]
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img_f in cls_folder.glob(ext):
                        recs.append({
                            "image": str(img_f.resolve()),
                            "label": cid,
                            "bbox": None,
                        })
            return recs

        train_records = _scan_folder(root / "train")
        test_dir = root / "test" if (root / "test").is_dir() else root / "val"
        test_records = _scan_folder(test_dir)

    # 4. Check for unsplit class subdirectories
    else:
        all_recs = []
        if root.is_dir():
            for cls_folder in root.iterdir():
                if not cls_folder.is_dir():
                    continue
                cname = cls_folder.name.lower()
                if cname not in label_to_id:
                    continue
                cid = label_to_id[cname]
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img_f in cls_folder.glob(ext):
                        all_recs.append({
                            "image": str(img_f.resolve()),
                            "label": cid,
                            "bbox": None,
                        })

        if not all_recs:
            raise FileNotFoundError(
                f"Brackish dataset not found in '{root}'.\n"
                f"Expected class folders ({class_names}) or 'metadata.csv'.\n"
                "Run `python scripts/download_brackish.py` to download to G-Drive."
            )

        random.seed(seed)
        random.shuffle(all_recs)
        split_idx = int(0.8 * len(all_recs))
        train_records = all_recs[:split_idx]
        test_records = all_recs[split_idx:]

    random.seed(seed)
    if max_train_samples is not None and len(train_records) > max_train_samples:
        train_records = random.sample(train_records, max_train_samples)
    if max_test_samples is not None and len(test_records) > max_test_samples:
        test_records = random.sample(test_records, max_test_samples)

    print(f"  [Brackish] Ready: {len(train_records):,} Train | {len(test_records):,} Test ({num_classes} classes)")
    return train_records, test_records, num_classes, class_names
