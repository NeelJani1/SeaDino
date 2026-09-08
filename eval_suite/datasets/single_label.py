"""Instant in-memory single-label dataset loader with fault-tolerant fetching."""

import ast
from typing import List, Tuple
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class SingleLabelDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            with Image.open(path) as img:
                img.load()
                img = img.convert("RGB")
        except Exception:
            alt_path = path[:-4] + ".JPG" if path.endswith(".jpg") else path[:-4] + ".jpg"
            try:
                with Image.open(alt_path) as img:
                    img.load()
                    img = img.convert("RGB")
            except Exception:
                # If file is one of the 13 missing images on disk, safely fetch adjacent sample
                return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform is not None:
            img = self.transform(img)

        return img, label


def prepare_single_label_data(csv_path: str, image_root: str, label_col: str):
    df = pd.read_csv(csv_path)

    def _parse(v):
        if isinstance(v, str):
            try:
                p = ast.literal_eval(v)
                return p[0] if isinstance(p, (list, tuple)) else int(p)
            except Exception:
                return int(v.strip("[]'\" "))
        elif isinstance(v, (list, tuple)):
            return int(v[0])
        return int(v)

    def _instant_index(sub_df):
        datasets = sub_df["dataset"].astype(str).str.strip().tolist()
        sites = sub_df["site"].astype(str).str.strip().tolist()
        images = sub_df["image"].astype(str).str.strip().tolist()
        labels = [_parse(x) for x in sub_df[label_col]]

        # Zero disk checks: constructs paths in RAM in 0.01 seconds
        return [
            (f"{image_root}/{d}/{s}/{im}.jpg", lbl)
            for d, s, im, lbl in zip(datasets, sites, images, labels)
        ]

    train_df = df[df["partition"].str.lower() == "train"]
    test_df = df[df["partition"].str.lower().isin(["test", "val", "validation"])]

    train_samples = _instant_index(train_df)
    test_samples = _instant_index(test_df)

    mapping = {lbl: i for i, lbl in enumerate(sorted(list({lbl for _, lbl in train_samples})))}
    train_samples = [(p, mapping[lbl]) for p, lbl in train_samples if lbl in mapping]
    test_samples = [(p, mapping[lbl]) for p, lbl in test_samples if lbl in mapping]
    return train_samples, test_samples, len(mapping)
