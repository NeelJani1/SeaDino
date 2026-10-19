"""A single ordered sample selection shared by every compared backbone."""
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .common import digest, file_sha

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASSIFICATION = ("substrate", "german_bank", "fathomnet", "brackish", "deepfish")
SEGMENTATION = ("coralmask", "coralscapes")
_HF_SPLITS = None


def hf_item(location):
    """Resolve exactly the cached Coralscapes row recorded by the benchmark."""
    global _HF_SPLITS
    if _HF_SPLITS is None:
        from datasets import load_dataset
        _HF_SPLITS = load_dataset("EPFL-ECEO/coralscapes")
    owner, name, fingerprint, split, index = location.removeprefix("hf://").split("/")
    if (owner, name) != ("EPFL-ECEO", "coralscapes"):
        raise ValueError(f"Unsupported HF image source: {owner}/{name}")
    dataset = _HF_SPLITS[split]
    if dataset._fingerprint != fingerprint:
        raise ValueError("Coralscapes cache fingerprint differs from the saved manifest")
    return dataset[int(index)]


def open_image(record):
    """Strict image access: a broken image is never replaced by another sample."""
    location = record["image"]
    try:
        if location.startswith("hf://"):
            return hf_item(location)["image"].convert("RGB")
        with Image.open(location) as image:
            image.load()
            # Benchmark transforms operate on decoded RGB without EXIF rotation.
            return image.convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Unreadable sample {record['sample_id']}: {location}") from exc


def _input_path(value):
    supplied = Path(value).expanduser()
    resolved = supplied.resolve()
    if not resolved.exists():
        hint = ""
        if not supplied.is_absolute() and supplied.parts[:1] == ("mnt",):
            hint = f" WSL drive paths need a leading '/': /{supplied.as_posix()}"
        raise FileNotFoundError(f"Image input does not exist: {resolved}.{hint}")
    return resolved


def image_records(image):
    path = _input_path(image)
    if not path.is_file():
        raise ValueError(f"Expected a single image file: {path}; use --input_dir for a folder")
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix or '(none)'} ({path})")
    return [dict(image=str(path), sample_id=digest(str(path)), label_name=None,
                 partition="single-image")], dict(source=str(path), task="folder", split="single-image")


def folder_records(directory, label_mode="auto"):
    root = _input_path(directory)
    if root.is_file():
        return image_records(root)
    if not root.is_dir():
        raise ValueError(f"Expected an image folder or image file: {root}")
    records = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS):
        relative = path.relative_to(root)
        label = None
        if label_mode == "parent":
            label = path.parent.name
        elif label_mode == "auto" and len(relative.parts) > 1:
            label = relative.parts[0]
        records.append(dict(image=str(path), sample_id=digest(str(path)), label_name=label,
                            partition="folder", relative_path=str(relative)))
    return records, {"source": str(root), "task": "folder", "labels": "folder names; not verified annotations"}


def benchmark_records(name, split, manifest_dir=None, data_root=None):
    from eval_suite.config import DEFAULT_PATHS, REPO_ROOT
    root = Path(manifest_dir) if manifest_dir else REPO_ROOT / "data/benchmarks/splits/explicit-v1/seed-42"
    if name in CLASSIFICATION[:4] or name == "coralscapes":
        table, metadata = root / f"{name}.csv", root / f"{name}.json"
        if not table.is_file() or not metadata.is_file():
            raise FileNotFoundError(f"Existing explicit-v1 manifest required: {table}")
        meta = json.loads(metadata.read_text())
        actual = file_sha(table)
        if meta.get("csv_sha256") and actual != meta["csv_sha256"]:
            raise ValueError(f"Manifest checksum mismatch: {table}")
        rows = list(csv.DictReader(table.open()))
        records = []
        for row in rows:
            if row["partition"] != split:
                continue
            row["label_name"] = None if name == "coralscapes" else meta["class_names"][int(row["label"])]
            records.append(row)
        return records, {"source": str(table), "manifest_sha256": actual, "task": "segmentation" if name == "coralscapes" else "classification", "split": split, "protocol": meta.get("protocol"), "class_names": meta["class_names"]}
    if name == "deepfish":
        from eval_suite.datasets.deepfish import prepare_deepfish_data
        directory = data_root or DEFAULT_PATHS["deepfish_dir"]
        # Require the requested CSV, avoiding the legacy loader's test->val fallback.
        candidates = [Path(directory), Path(directory) / "Classification", Path(directory) / "DeepFish/Classification"]
        actual_split = "val" if split == "validation" else split
        source = next((p / f"{actual_split}.csv" for p in candidates if (p / f"{actual_split}.csv").is_file()), None)
        if source is None:
            raise FileNotFoundError(f"DeepFish {actual_split}.csv not found under {directory}")
        train, selected, _, names = prepare_deepfish_data(directory, split_test=actual_split)
        samples = train if split == "train" else selected
        records = [dict(image=str(p), sample_id=digest(str(p)), label_name=names[int(y)], partition=split) for p, y in samples]
        return records, {"source": str(source), "manifest_sha256": file_sha(source), "task": "classification", "split": split, "protocol": "DeepFish supplied partitions"}
    if name == "coralmask":
        if split not in ("train", "test"):
            raise ValueError("CoralMask has train/test only; validation is not synthesized")
        directory = data_root or DEFAULT_PATHS["coralmask_dir"]
        if not Path(directory).exists() and Path("/mnt/d/Neel/CoralMaskv1/CoralMask").exists():
            directory = "/mnt/d/Neel/CoralMaskv1/CoralMask"
        clean = DEFAULT_PATHS["coralmask_test_manifest"]
        if split == "test" and not Path(clean).is_file():
            raise FileNotFoundError(f"CoralMask clean test manifest required: {clean}")
        image_dir, json_dir = Path(directory) / split / "images", Path(directory) / split / "jsons"
        # Match the benchmark's image/annotation eligibility with two directory listings,
        # rather than tens of thousands of individual filesystem checks.
        annotations = {p.stem for p in json_dir.iterdir() if p.suffix.lower() == ".json"}
        if split == "test":
            paths = [image_dir / Path(line.strip()).name for line in Path(clean).read_text().splitlines() if line.strip()]
        else:
            paths = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
        paths = [p for p in paths if p.stem in annotations]
        rows = [dict(image=str(p), sample_id=digest(str(p)), label_name=None, partition=split) for p in paths]
        return rows, {"source": directory, "task": "segmentation", "split": split, "clean_test_manifest": clean if split == "test" else None}
    raise ValueError(f"Unsupported benchmark: {name}")


def select_records(records, maximum, seed, balanced=True):
    if maximum < 1:
        raise ValueError("max_images must be positive")
    unique = {}
    for row in records:
        key = row["image"]
        if key in unique and unique[key].get("label_name") != row.get("label_name"):
            raise ValueError(f"Conflicting labels for image {key}")
        unique.setdefault(key, row)
    rows = sorted(unique.values(), key=lambda r: (r["sample_id"], r["image"]))
    if not rows:
        raise ValueError("No images found in the requested input/split")
    rng = np.random.default_rng(seed)
    if balanced and all(r.get("label_name") is not None for r in rows):
        groups = defaultdict(list)
        for row in rows:
            groups[row["label_name"]].append(row)
        for group in groups.values():
            rng.shuffle(group)
        chosen = []
        while len(chosen) < min(maximum, len(rows)):
            for label in sorted(groups):
                if groups[label] and len(chosen) < maximum:
                    chosen.append(groups[label].pop())
        return chosen
    order = rng.permutation(len(rows))[:maximum]
    return [rows[int(i)] for i in order]


def stage_images(records, output):
    """Decode once, preserve identities, and make later GPU passes independent of G: latency."""
    directory = Path(output) / "images"
    directory.mkdir(parents=True, exist_ok=True)
    result = []
    for index, source in enumerate(records):
        row = dict(source)
        suffix = ".png" if row["image"].startswith("hf://") else Path(row["image"]).suffix.lower()
        path = directory / f"{index:05d}_{row['sample_id'][:16]}{suffix}"
        if row["image"].startswith("hf://"):
            image = open_image(row)
            image.save(path)
        else:
            shutil.copyfile(row["image"], path)
            image = open_image(dict(row, image=str(path)))
        row.update(staged_image=str(path), image_sha256=file_sha(path), width=image.width, height=image.height)
        thumb = image.copy()
        thumb.thumbnail((160, 120), Image.Resampling.LANCZOS)
        thumb_path = directory / f"{index:05d}_{row['sample_id'][:16]}_thumb.jpg"
        thumb.save(thumb_path, quality=80)
        row["thumbnail"] = str(thumb_path)
        result.append(row)
        if index % 100 == 0:
            print(f"Staged {index + 1}/{len(records)} images", flush=True)
    return result


def sample_summary(records):
    return dict(total=len(records), labels=dict(sorted(Counter(r.get("label_name") or "Unlabeled" for r in records).items())))
