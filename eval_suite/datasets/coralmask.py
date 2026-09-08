"""Dataset loader for CoralMask COCO RLE binary coral segmentation."""

import json
import os
import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from eval_suite.config import IMAGENET_MEAN, IMAGENET_STD


class CoralMaskDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", width: int = 512, height: int = 512,
                 max_samples=None, mean=IMAGENET_MEAN, std=IMAGENET_STD, test_manifest=None):
        self.img_dir = os.path.join(data_dir, split, "images")
        self.json_dir = os.path.join(data_dir, split, "jsons")
        self.width, self.height = width, height

        self.samples = []
        if split == "test" and test_manifest and os.path.isfile(test_manifest):
            from pathlib import Path
            with open(test_manifest, "r") as f:
                manifest_paths = [Path(line.strip()) for line in f if line.strip()]
            for p in manifest_paths:
                json_path = os.path.join(self.json_dir, f"{p.stem}.json")
                if os.path.isfile(json_path):
                    self.samples.append((str(p), json_path))
        else:
            img_files = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            for img_name in img_files:
                json_path = os.path.join(self.json_dir, f"{os.path.splitext(img_name)[0]}.json")
                if os.path.isfile(json_path):
                    self.samples.append((os.path.join(self.img_dir, img_name), json_path))

        if max_samples and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]

        self.transform_img = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]
        with Image.open(img_path) as raw_img:
            orig_w, orig_h = raw_img.size
            img = raw_img.convert("RGB").resize((self.width, self.height), Image.BILINEAR)

        binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            for ann in data.get("annotations", []):
                seg = ann.get("segmentation", None)
                if seg and isinstance(seg, dict) and "counts" in seg and "size" in seg:
                    rle = {"size": seg["size"], "counts": seg["counts"].encode("utf-8") if isinstance(seg["counts"], str) else seg["counts"]}
                    decoded = mask_utils.decode(rle)
                    if decoded.shape == (orig_h, orig_w):
                        binary_mask = np.maximum(binary_mask, decoded)
        except Exception:
            pass

        mask_pil = Image.fromarray(binary_mask).resize((self.width, self.height), Image.NEAREST)
        del binary_mask
        mask_tensor = torch.from_numpy(np.array(mask_pil, dtype=np.int64))
        return self.transform_img(img), mask_tensor
