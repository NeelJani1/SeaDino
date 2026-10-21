"""Dataset loader for CoralMask COCO RLE binary coral segmentation."""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
from PIL import Image
import pycocotools.mask as mask_utils
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from eval_suite.config import IMAGENET_MEAN, IMAGENET_STD


def prepare_coralmask_splits(
    data_dir: str,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    test_manifest: Optional[str] = None,
    max_train_samples: Optional[int] = None,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]], Dict[str, Any]]:
    """Prepares and strictly validates CoralMask train, val, and test partitions.

    Guarantees:
    - If a test manifest is requested, it must exist on disk and have unique entries.
    - Test, train, and validation partitions are non-empty.
    - Zero sample overlap between active partitions (by canonical path and by image content hash).
    - Identical-content images and available group metadata stay together within partitions.
    - Computes and returns an immutable split provenance fingerprint.
    """
    if val_ratio < 0.0 or val_ratio >= 1.0 or not np.isfinite(val_ratio):
        raise ValueError(f"Invalid CoralMask val_ratio {val_ratio}: must be strictly 0 <= val_ratio < 1.")

    train_img_dir = os.path.join(data_dir, "train", "images")
    train_json_dir = os.path.join(data_dir, "train", "jsons")
    test_img_dir = os.path.join(data_dir, "test", "images")
    test_json_dir = os.path.join(data_dir, "test", "jsons")
    test_samples: List[Tuple[str, str]] = []
    seen_test_canonical = set()

    if test_manifest:
        if not os.path.isfile(test_manifest):
            raise FileNotFoundError(f"CoralMask test manifest not found: '{test_manifest}'")
        with open(test_manifest, "r", encoding="utf-8") as f:
            manifest_lines = [line.strip() for line in f if line.strip()]
        if not manifest_lines:
            raise ValueError(f"CoralMask test manifest '{test_manifest}' is empty.")

        seen_manifest_entries = set()
        train_img_dir_real = os.path.realpath(train_img_dir)
        train_json_dir_real = os.path.realpath(train_json_dir)

        for line in manifest_lines:
            if line in seen_manifest_entries:
                raise ValueError(f"Duplicate entry found in CoralMask test manifest: '{line}'")
            seen_manifest_entries.add(line)

            p = Path(line)
            img_path = str(p)
            if not os.path.isfile(img_path):
                rel_to_data = os.path.join(data_dir, line)
                if os.path.isfile(rel_to_data):
                    img_path = rel_to_data
                else:
                    local_img = os.path.join(test_img_dir, p.name)
                    if os.path.isfile(local_img):
                        img_path = local_img
                    else:
                        raise FileNotFoundError(f"Test image from manifest not found: '{p}'")

            can_img = os.path.realpath(img_path)
            if can_img == train_img_dir_real or can_img.startswith(train_img_dir_real + os.sep):
                raise ValueError(f"Cross-partition leakage detected: test manifest references training image '{img_path}'.")

            json_path = os.path.join(test_json_dir, f"{p.stem}.json")
            if not os.path.isfile(json_path):
                img_p_obj = Path(img_path)
                for cand in [
                    os.path.join(img_p_obj.parent.parent, "jsons", f"{p.stem}.json"),
                    os.path.join(img_p_obj.parent, f"{p.stem}.json"),
                    os.path.join(img_p_obj.parent, "jsons", f"{p.stem}.json"),
                ]:
                    cand_real = os.path.realpath(cand)
                    if cand_real == train_json_dir_real or cand_real.startswith(train_json_dir_real + os.sep):
                        continue
                    if os.path.isfile(cand):
                        json_path = cand
                        break
                else:
                    raise FileNotFoundError(f"Annotation json for test image not found: '{json_path}'")

            can_json = os.path.realpath(json_path)
            if can_img in seen_test_canonical:
                raise ValueError(f"Duplicate test sample resolved in manifest: '{can_img}'")
            seen_test_canonical.add(can_img)
            test_samples.append((can_img, can_json))
    elif os.path.isdir(test_img_dir):
        img_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for img_name in img_files:
            img_path = os.path.realpath(os.path.join(test_img_dir, img_name))
            json_path = os.path.realpath(os.path.join(test_json_dir, f"{os.path.splitext(img_name)[0]}.json"))
            if os.path.isfile(json_path):
                if img_path in seen_test_canonical:
                    raise ValueError(f"Duplicate test sample found: '{img_path}'")
                seen_test_canonical.add(img_path)
                test_samples.append((img_path, json_path))

    if not test_samples:
        raise ValueError(f"CoralMask test partition is empty at '{data_dir}'.")

    # Verify unique image content within test partition
    test_content_hashes: Dict[str, str] = {}
    for img_p, _ in test_samples:
        with open(img_p, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        if h in test_content_hashes:
            raise ValueError(f"Duplicate image content in test partition: '{img_p}' matches '{test_content_hashes[h]}'")
        test_content_hashes[h] = img_p

    # Load candidate train samples
    train_img_dir = os.path.join(data_dir, "train", "images")
    train_json_dir = os.path.join(data_dir, "train", "jsons")
    candidate_samples: List[Tuple[str, str]] = []
    seen_train_canonical = set()

    if os.path.isdir(train_img_dir):
        img_files = sorted([f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for img_name in img_files:
            img_path = os.path.realpath(os.path.join(train_img_dir, img_name))
            json_path = os.path.realpath(os.path.join(train_json_dir, f"{os.path.splitext(img_name)[0]}.json"))
            if os.path.isfile(json_path):
                if img_path in seen_train_canonical:
                    raise ValueError(f"Duplicate train sample found: '{img_path}'")
                seen_train_canonical.add(img_path)
                candidate_samples.append((img_path, json_path))

    if not candidate_samples:
        raise ValueError(f"CoralMask train partition is empty at '{data_dir}'.")

    # Cross-partition leakage check between candidate train and test
    test_paths_set = set(s[0] for s in test_samples)
    for img_p, _ in candidate_samples:
        if img_p in test_paths_set:
            raise ValueError(f"Cross-partition leakage detected: path '{img_p}' is present in both train and test partitions.")
        with open(img_p, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        if h in test_content_hashes:
            raise ValueError(
                f"Cross-partition leakage detected: image content in '{img_p}' matches test image '{test_content_hashes[h]}'."
            )

    # Train / Val splitting with duplicate & group clustering
    if val_ratio > 0.0:
        # Build connected components over both content equivalence and metadata groups
        n_samples = len(candidate_samples)
        parent = list(range(n_samples))

        def find(i: int) -> int:
            path = []
            while parent[i] != i:
                path.append(i)
                i = parent[i]
            for p in path:
                parent[p] = i
            return i

        def union(i: int, j: int):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        content_map: Dict[str, int] = {}
        meta_map: Dict[str, int] = {}

        for idx, (img_p, json_p) in enumerate(candidate_samples):
            # 1. Content hash constraint
            with open(img_p, "rb") as f:
                chash = hashlib.sha256(f.read()).hexdigest()
            if chash in content_map:
                union(idx, content_map[chash])
            else:
                content_map[chash] = idx

            # 2. Metadata grouping constraint
            meta_key = None
            try:
                with open(json_p, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)
                for k in ("group", "group_id", "site", "location", "cluster", "image_group"):
                    if k in ann_data and ann_data[k] is not None:
                        meta_key = f"{k}_{ann_data[k]}"
                        break
            except Exception:
                pass

            if meta_key is not None:
                if meta_key in meta_map:
                    union(idx, meta_map[meta_key])
                else:
                    meta_map[meta_key] = idx

        # Group sample indices into connected components
        components_by_root: Dict[int, List[Tuple[str, str]]] = {}
        for idx, sample in enumerate(candidate_samples):
            root = find(idx)
            components_by_root.setdefault(root, []).append(sample)

        # Sort samples within each component, then sort components deterministically
        for comp in components_by_root.values():
            comp.sort(key=lambda s: s[0])

        unique_components = sorted(components_by_root.values(), key=lambda c: c[0][0])

        if len(candidate_samples) < 2 or len(unique_components) < 2:
            raise ValueError(
                f"Insufficient samples or groups ({len(candidate_samples)} samples, {len(unique_components)} groups) "
                f"to create non-empty train and validation splits with val_ratio={val_ratio}."
            )

        rng = np.random.RandomState(split_seed)
        perm = rng.permutation(len(unique_components))
        val_group_count = max(1, int(np.floor(len(unique_components) * val_ratio)))
        if val_group_count >= len(unique_components):
            val_group_count = len(unique_components) - 1

        val_components = [unique_components[i] for i in perm[:val_group_count]]
        train_components = [unique_components[i] for i in perm[val_group_count:]]

        train_samples = [s for c in train_components for s in c]
        val_samples = [s for c in val_components for s in c]
        train_samples.sort(key=lambda s: s[0])
        val_samples.sort(key=lambda s: s[0])
    else:
        train_samples = list(candidate_samples)
        val_samples = []

    if max_train_samples is not None:
        if max_train_samples <= 0:
            raise ValueError(f"CoralMask max_train_samples must be positive, got {max_train_samples}")
        train_samples = train_samples[:max_train_samples]

    if not train_samples:
        raise ValueError("CoralMask train partition is empty after filtering and split creation.")
    if val_ratio > 0.0 and not val_samples:
        raise ValueError("CoralMask validation partition is empty after filtering and split creation.")

    # Disjointness assert between train and val (both paths and content hashes)
    train_paths = set(s[0] for s in train_samples)
    val_paths = set(s[0] for s in val_samples)
    if train_paths & val_paths:
        raise ValueError("Train and validation partitions overlap by path!")

    if val_samples:
        def _get_content_sha(p_str: str) -> str:
            with open(p_str, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        train_content_hashes = {_get_content_sha(s[0]) for s in train_samples}
        val_content_hashes = {_get_content_sha(s[0]) for s in val_samples}
        overlap = train_content_hashes & val_content_hashes
        if overlap:
            raise ValueError(f"Cross-partition leakage detected: train and validation share {len(overlap)} identical content hashes!")

    # Calculate partition hashes and immutable split fingerprint
    train_hash = hashlib.sha256("".join(sorted(train_paths)).encode()).hexdigest()
    val_hash = hashlib.sha256("".join(sorted(val_paths)).encode()).hexdigest() if val_paths else "none"
    test_hash = hashlib.sha256("".join(sorted(test_paths_set)).encode()).hexdigest()
    if test_manifest and os.path.isfile(test_manifest):
        with open(test_manifest, "rb") as f:
            manifest_hash = hashlib.sha256(f.read()).hexdigest()
    else:
        manifest_hash = "none"

    fingerprint_str = f"train={train_hash}:val={val_hash}:test={test_hash}:seed={split_seed}:ratio={val_ratio}:manifest={manifest_hash}:rule=best_val_miou"
    split_fingerprint = hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]

    provenance = {
        "coralmask_split_fingerprint": split_fingerprint,
        "coralmask_val_ratio": val_ratio,
        "coralmask_split_seed": split_seed,
        "coralmask_test_manifest_sha256": manifest_hash,
        "coralmask_selection_rule": "best_val_miou",
        "coralmask_train_count": len(train_samples),
        "coralmask_val_count": len(val_samples),
        "coralmask_test_count": len(test_samples),
    }

    return train_samples, val_samples, test_samples, provenance


class CoralMaskDataset(Dataset):
    def __init__(
        self,
        data_dir: Optional[str] = None,
        split: str = "train",
        width: int = 512,
        height: int = 512,
        max_samples: Optional[int] = None,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        test_manifest: Optional[str] = None,
        val_ratio: float = 0.0,
        split_seed: int = 42,
        samples: Optional[List[Tuple[str, str]]] = None,
    ):
        self.width, self.height = width, height

        if samples is not None:
            self.samples = list(samples)
            if max_samples and max_samples < len(self.samples):
                self.samples = self.samples[:max_samples]
        elif data_dir is not None:
            if split not in ("train", "val", "test"):
                raise ValueError(f"Unknown split '{split}', expected 'train', 'val', or 'test'.")

            if split in ("train", "val"):
                if val_ratio < 0.0 or val_ratio >= 1.0 or not np.isfinite(val_ratio):
                    raise ValueError(f"Invalid val_ratio {val_ratio}: must be 0 <= val_ratio < 1.")
                if split == "val" and val_ratio == 0.0:
                    raise ValueError("Validation split requested with val_ratio=0.0: validation partition is empty.")
                effective_ratio = val_ratio
                train_s, val_s, _, _ = prepare_coralmask_splits(
                    data_dir,
                    val_ratio=effective_ratio,
                    split_seed=split_seed,
                    test_manifest=test_manifest,
                    max_train_samples=max_samples if split == "train" else None,
                )
                self.samples = train_s if split == "train" else val_s
            elif split == "test":
                _, _, test_s, _ = prepare_coralmask_splits(
                    data_dir,
                    val_ratio=0.0,
                    split_seed=split_seed,
                    test_manifest=test_manifest,
                    max_train_samples=None,
                )
                self.samples = test_s
                if max_samples and max_samples < len(self.samples):
                    self.samples = self.samples[:max_samples]
        else:
            self.samples = []

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
            with open(json_path, "r", encoding="utf-8") as f:
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
