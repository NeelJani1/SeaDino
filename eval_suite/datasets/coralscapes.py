"""Dataset loader for EPFL Coralscapes segmentation benchmark."""

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from eval_suite.config import IMAGENET_MEAN, IMAGENET_STD


class CoralscapesDataset(Dataset):
    def __init__(self, hf_split, width: int = 896, height: int = 448, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.split = hf_split
        self.width, self.height = width, height
        self.transform_img = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        item = self.split[idx]
        img = item["image"].convert("RGB").resize((self.width, self.height), Image.BILINEAR)
        mask = item["label"].resize((self.width, self.height), Image.NEAREST)
        return self.transform_img(img), torch.from_numpy(np.array(mask, dtype=np.int64))
