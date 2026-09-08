"""Instant in-memory Biota dataset loader with fault-tolerant fetching."""

import ast
from typing import List, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset


class BiotaDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, List[int]]], num_classes: int, transform=None):
        self.samples = samples
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, labels = self.samples[idx]
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
                return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform is not None:
            img = self.transform(img)

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[labels] = 1.0
        return img, target


def prepare_biota_data(csv_path: str, image_root: str):
    df = pd.read_csv(csv_path).dropna(subset=["catami_biota"]).reset_index(drop=True)

    def _parse(raw):
        if isinstance(raw, str):
            try:
                p = ast.literal_eval(raw)
                return list(p) if isinstance(p, (list, tuple, set)) else [int(p)]
            except Exception:
                return [int(x.strip()) for x in raw.strip("[]'\" ").split(",") if x.strip().isdigit()]
        elif isinstance(raw, (list, tuple)):
            return [int(x) for x in raw]
        return [int(raw)]

    all_taxa = set()
    for item in df["catami_biota"]:
        all_taxa.update(_parse(item))
    mapping = {lbl: i for i, lbl in enumerate(sorted(list(all_taxa)))}
    num_classes = len(mapping)

    def _instant_index(sub_df):
        datasets = sub_df["dataset"].astype(str).str.strip().tolist()
        sites = sub_df["site"].astype(str).str.strip().tolist()
        images = sub_df["image"].astype(str).str.strip().tolist()
        biota_raw = [_parse(x) for x in sub_df["catami_biota"]]

        out = []
        for d, s, im, raw_lbls in zip(datasets, sites, images, biota_raw):
            parsed = [mapping[x] for x in raw_lbls if x in mapping]
            if parsed:
                out.append((f"{image_root}/{d}/{s}/{im}.jpg", parsed))
        return out

    train_df = df[df["partition"].str.lower() == "train"]
    test_df = df[df["partition"].str.lower().isin(["test", "val", "validation"])]

    train_samples = _instant_index(train_df)
    test_samples = _instant_index(test_df)
    return train_samples, test_samples, num_classes
