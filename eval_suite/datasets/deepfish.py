"""Dataset loader for DeepFish fish-habitat classification & evaluation benchmark.

Supports loading from extracted DeepFish archive or local dataset directory.
Handles official train/val/test splits across 20 tropical marine habitats with
fish / no-fish binary labels.
Reference: Saleh et al., Nature Scientific Reports 2020 (DOI: 10.1038/s41598-020-71639-x).
"""

import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from eval_suite.config import IMAGENET_MEAN, IMAGENET_STD


class DeepFishDataset(Dataset):
    """PyTorch Dataset for DeepFish binary fish classification.

    Each item in `samples` is a tuple of (image_path, label_int).
    """

    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int, retries: int = 0) -> Tuple[torch.Tensor, int]:
        if retries >= 10:
            raise RuntimeError(f"Failed to fetch readable image after {retries} retries starting at {idx}.")

        img_path, label = self.samples[idx]
        try:
            with Image.open(img_path) as im:
                im.load()
                img = im.convert("RGB")
        except Exception:
            # Fallback to adjacent sample if file is missing/unreadable
            return self.__getitem__((idx + 1) % len(self.samples), retries=retries + 1)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = TF.to_tensor(img)

        return img, int(label)


def _locate_deepfish_root(data_dir_or_path: str) -> Path:
    """Finds the root directory containing DeepFish classification files."""
    candidates = [
        Path(data_dir_or_path),
        Path(data_dir_or_path) / "DeepFish",
        Path(data_dir_or_path) / "Classification",
    ]
    for c in candidates:
        if (c / "train.csv").exists() or (c / "Classification" / "train.csv").exists():
            return c.resolve()
    return Path(data_dir_or_path).resolve()


def prepare_deepfish_data(
    data_dir: str,
    split_test: str = "test",
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], int, List[str]]:
    """Loads and standardizes DeepFish classification data.

    Expected directory structure:
      DeepFish/
        Classification/
          train.csv
          val.csv
          test.csv
          images/
            <habitat>/<filename>.jpg

    Returns:
      train_samples, test_samples, num_classes=2, class_names=["No-Fish", "Fish"]
    """
    root_path = _locate_deepfish_root(data_dir)
    clf_dir = root_path / "Classification" if (root_path / "Classification").is_dir() else root_path

    train_csv = clf_dir / "train.csv"
    test_csv = clf_dir / f"{split_test}.csv"
    if not test_csv.exists() and split_test == "test":
        test_csv = clf_dir / "val.csv"

    if not train_csv.exists():
        raise FileNotFoundError(
            f"DeepFish classification splits not found in '{clf_dir}'.\n"
            f"Expected 'train.csv' and '{split_test}.csv' in {clf_dir}.\n"
            "To download the official 7.1GB DeepFish dataset from Hugging Face:\n"
            "  pip install 'huggingface-hub<2.0,>=1.3.0'\n"
            f"  hf download Alzayats/DeepFish DeepFish.tar --repo-type dataset --local-dir {data_dir}\n"
            f"  cd {data_dir} && tar -xf DeepFish.tar\n"
        )

    # Candidate directories for image files
    search_dirs = [
        clf_dir,
        clf_dir / "images",
        root_path / "Classification" / "images",
        root_path / "images",
        root_path,
    ]
    existing_dirs = [d for d in search_dirs if d.is_dir()]

    def _load_split(csv_path: Path) -> List[Tuple[str, int]]:
        df = pd.read_csv(csv_path)
        id_col = next((c for c in ["ID", "id", "image_id", "file_name", "image"] if c in df.columns), df.columns[0])
        lbl_col = next((c for c in ["labels", "label", "class", "target"] if c in df.columns), df.columns[1])

        records = []
        missing_count = 0
        for _, row in df.iterrows():
            stem = str(row[id_col]).strip()
            img_path = None
            has_ext = stem.lower().endswith((".jpg", ".png", ".jpeg"))

            for base in existing_dirs:
                if has_ext:
                    cand = base / stem
                    if cand.is_file():
                        img_path = cand
                        break
                else:
                    for ext in [".jpg", ".png", ".jpeg"]:
                        cand = base / f"{stem}{ext}"
                        if cand.is_file():
                            img_path = cand
                            break
                if img_path is not None:
                    break

            if img_path is None:
                missing_count += 1
                img_path = clf_dir / (stem if has_ext else f"{stem}.jpg")

            lbl_val = row[lbl_col]
            # Binary label: 0 = No-Fish, 1 = Fish
            lbl = 1 if int(lbl_val) > 0 else 0
            records.append((str(img_path), lbl))

        if missing_count > 0:
            print(f"  [DeepFish Warning] {missing_count}/{len(records)} images in {csv_path.name} not found on disk at resolution time.")
        return records

    print(f"  [DeepFish] Loading splits from '{clf_dir}'...")
    train_records = _load_split(train_csv)
    test_records = _load_split(test_csv)

    # Subsample if requested
    random.seed(seed)
    if max_train_samples is not None and len(train_records) > max_train_samples:
        train_records = random.sample(train_records, max_train_samples)

    if max_test_samples is not None and len(test_records) > max_test_samples:
        test_records = random.sample(test_records, max_test_samples)

    class_names = ["No-Fish", "Fish"]
    num_classes = 2

    print(f"  [DeepFish] Ready: {len(train_records):,} Train | {len(test_records):,} Test ({num_classes} classes)")
    return train_records, test_records, num_classes, class_names
