"""Multi-label Biota dataset loader returning compact uint8 tensors."""

import ast
from typing import List, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from eval_suite.utils import fast_resolve_path


class BiotaDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, List[int]]], num_classes: int):
        self.samples = samples
        self.num_classes = num_classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, labels = self.samples[idx]
        with Image.open(path) as img:
            img.load()
            img = img.convert("RGB")
        tensor_img = pil_to_tensor(img)

        target = torch.zeros(self.num_classes, dtype=torch.float32)
        target[labels] = 1.0
        return tensor_img, target


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

    def _fast_index(sub_df):
        out = []
        datasets = [str(x).strip() for x in sub_df["dataset"]]
        sites = [str(x).strip() for x in sub_df["site"]]
        images = [str(x).strip() for x in sub_df["image"]]
        biota_raw = [_parse(x) for x in sub_df["catami_biota"]]

        for d, s, im, raw_lbls in zip(datasets, sites, images, biota_raw):
            p = fast_resolve_path(image_root, d, s, im)
            if p is not None:
                parsed = [mapping[x] for x in raw_lbls if x in mapping]
                if parsed:
                    out.append((p, parsed))
        return out

    train_df = df[df["partition"].str.lower() == "train"]
    test_df = df[df["partition"].str.lower().isin(["test", "val", "validation"])]

    train_samples = _fast_index(train_df)
    test_samples = _fast_index(test_df)
    return train_samples, test_samples, num_classes
