"""Single-label dataset loaders returning compact uint8 tensors."""

import ast
from typing import List, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor
from eval_suite.utils import fast_resolve_path


class SingleLabelDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img.load()
            img = img.convert("RGB")
        # Returns raw (3, H, W) uint8 tensor (instant on CPU)
        tensor_img = pil_to_tensor(img)
        return tensor_img, label


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

    def _fast_index(sub_df):
        out = []
        datasets = [str(x).strip() for x in sub_df["dataset"]]
        sites = [str(x).strip() for x in sub_df["site"]]
        images = [str(x).strip() for x in sub_df["image"]]
        labels = [_parse(x) for x in sub_df[label_col]]

        for d, s, im, lbl in zip(datasets, sites, images, labels):
            p = fast_resolve_path(image_root, d, s, im)
            if p is not None:
                out.append((p, lbl))
        return out

    train_df = df[df["partition"].str.lower() == "train"]
    test_df = df[df["partition"].str.lower().isin(["test", "val", "validation"])]

    train_samples = _fast_index(train_df)
    test_samples = _fast_index(test_df)

    mapping = {lbl: i for i, lbl in enumerate(sorted(list({lbl for _, lbl in train_samples})))}
    train_samples = [(p, mapping[lbl]) for p, lbl in train_samples if lbl in mapping]
    test_samples = [(p, mapping[lbl]) for p, lbl in test_samples if lbl in mapping]
    return train_samples, test_samples, len(mapping)
