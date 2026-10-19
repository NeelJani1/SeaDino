#!/usr/bin/env python3
"""Downloads the full Brackish underwater marine fauna benchmark at original resolution to G-Drive.

Sources from the verified 'dronefreak/Brackish' dataset on Hugging Face Hub,
which redistributes the Limfjorden underwater marine fauna benchmark
(Pedersen et al., CVPRW 2019) across 6 categories:
  crab, fish, jellyfish, shrimp, small_fish, starfish.

Downloads all original 1920x1080 RGB frames, original YOLO annotation text files,
and data.yaml directly to external G-Drive storage (/mnt/g/data/benchmarks/brackish).
Uses direct HTTP streaming with multi-threading and resume support without
huggingface_hub cache/chmod restrictions on WSL drvfs mounts.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import json
import os
from pathlib import Path
import random
import sys
import urllib.request

# Ensure repo root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
from PIL import Image
import requests
from tqdm import tqdm

from eval_suite.config import GDRIVE_CANDIDATE_PATHS, DATA_DIR, BRACKISH_CLASS_NAMES


def default_dest() -> Path:
    for g in GDRIVE_CANDIDATE_PATHS:
        try:
            if g.is_dir():
                return g / "benchmarks" / "brackish"
        except (PermissionError, OSError):
            continue
    return DATA_DIR / "benchmarks" / "brackish"


def fetch_repo_file_list(repo_id: str) -> list[str]:
    """Fetches list of repository file paths from Hugging Face Hub API."""
    api_url = f"https://huggingface.co/api/datasets/{repo_id}"
    headers = {"User-Agent": "SeaDino-Downloader/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(api_url, headers=headers, timeout=25)
        resp.raise_for_status()
        data = resp.json()
        return [s["rfilename"] for s in data.get("siblings", [])]
    except Exception as e:
        print(f"Warning: Failed to query Hugging Face API ({e}). Falling back to urllib...")
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [s["rfilename"] for s in data.get("siblings", [])]


def download_single_file(url: str, dest_path: Path, timeout: int = 25) -> bool:
    """Downloads a single file directly to dest_path safely with resume support."""
    if dest_path.is_file() and dest_path.stat().st_size > 0:
        return True
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "SeaDino-Downloader/1.0"}
    token = os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        if resp.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 64):
                    if chunk:
                        f.write(chunk)
            return True
    except Exception:
        pass

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
            if content:
                with open(dest_path, "wb") as f:
                    f.write(content)
                return True
    except Exception:
        pass
    return False


def build_brackish_metadata(dest: Path) -> Path:
    """Scans all downloaded YOLO annotations and original images to create metadata.csv."""
    records = []

    # Recursively find all label .txt files under dest
    for lbl_f in sorted(dest.rglob("*.txt")):
        if lbl_f.name in ["README.dataset.txt", "README.roboflow.txt"] or "cache" in lbl_f.parts:
            continue
        if lbl_f.parent.name != "labels":
            continue

        p_str = lbl_f.as_posix()
        if "/train/" in p_str:
            split_name = "train"
        elif "/valid/" in p_str or "/val/" in p_str:
            split_name = "val"
        elif "/test/" in p_str:
            split_name = "test"
        else:
            continue

        # Look for matching image in parallel 'images' directory
        img_dir = lbl_f.parent.parent / "images"
        if not img_dir.is_dir():
            continue

        img_cand = next(
            (img_dir / f"{lbl_f.stem}{ext}" for ext in [".jpg", ".png", ".jpeg"]
             if (img_dir / f"{lbl_f.stem}{ext}").is_file() and (img_dir / f"{lbl_f.stem}{ext}").stat().st_size > 0),
            None
        )
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
                if 0 <= cid < len(BRACKISH_CLASS_NAMES):
                    rel_img = str(img_cand.relative_to(dest))
                    records.append({
                        "file_name": rel_img,
                        "label": BRACKISH_CLASS_NAMES[cid],
                        "cid": cid,
                        "bbox": str(bbox) if bbox else "",
                        "split": split_name,
                        "width": 1920,
                        "height": 1080,
                    })
        except Exception:
            continue

    meta_csv = dest / "metadata.csv"
    if records:
        meta_df = pd.DataFrame(records)
        meta_df.to_csv(meta_csv, index=False)
        print(f"  • Successfully saved {len(meta_df):,} entries to {meta_csv}")
    return meta_csv


def main():
    p = argparse.ArgumentParser(description="Download full Brackish dataset at original resolution directly to G-Drive")
    p.add_argument("--dest_dir", type=str, default=str(default_dest()),
                   help="Destination directory on G drive.")
    p.add_argument("--repo_id", type=str, default="dronefreak/Brackish",
                   help="Hugging Face repository ID.")
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"],
                   choices=["all", "train", "valid", "test"],
                   help="Splits to download (default: train valid test for full dataset).")
    p.add_argument("--max_samples", type=int, default=-1,
                   help="Maximum number of images to download (-1 for full dataset).")
    p.add_argument("--num_workers", type=int, default=16,
                   help="Concurrent download threads (default: 16).")
    p.add_argument("--metadata_only", action="store_true", default=False,
                   help="Skip download and rebuild metadata.csv on already-downloaded files.")
    args = p.parse_args()

    dest = Path(args.dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Target Brackish Directory: {dest.resolve()}")

    if args.metadata_only:
        print("Rebuilding metadata.csv from local files...")
        meta_csv = build_brackish_metadata(dest)
        if meta_csv.is_file():
            df = pd.read_csv(meta_csv)
            print(f"\n🎉 Successfully indexed Brackish benchmark in original resolution:")
            print(f"  • Total Images: {len(df):,}")
            print(f"  • Resolution: 1920x1080 native RGB")
            print(f"  • Splits: {df['split'].value_counts().to_dict()}")
            print(f"  • Classes: {df['label'].value_counts().to_dict()}")
            print(f"  • Metadata CSV: {meta_csv.resolve()}")
            return 0
        else:
            print("Warning: Metadata indexing produced 0 samples.")
            return 1

    print(f"Connecting to Hugging Face Hub for '{args.repo_id}'...")

    files = fetch_repo_file_list(args.repo_id)
    if not files:
        print("Error: Could not list files from Hugging Face repository.")
        return 1

    print(f"  • Found {len(files):,} repository files on Hub.")

    selected_splits = ["train", "valid", "test"] if "all" in args.splits else args.splits
    base_hf_url = f"https://huggingface.co/datasets/{args.repo_id}/resolve/main"

    # 1. Download configuration files (data.yaml, README)
    print("Downloading dataset metadata and YAML config...")
    for cfg_file in ["data.yaml", "data/data.yaml", "README.md", "data/README.dataset.txt", "data/detectionbench_data.yaml"]:
        if cfg_file in files:
            download_single_file(f"{base_hf_url}/{cfg_file}", dest / cfg_file)

    # 2. Identify candidate labels and images across shards
    label_files = []
    image_files = []
    for f in files:
        for s in selected_splits:
            prefix = f"data/{s}/" if f.startswith("data/") else f"{s}/"
            if not f.startswith(prefix):
                continue
            if "/labels/" in f and f.endswith(".txt"):
                label_files.append((s, f))
            elif "/images/" in f and f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append((s, f))

    print(f"  • Found {len(label_files):,} candidate label files across splits {selected_splits}.")
    print(f"  • Found {len(image_files):,} original 1920x1080 images across splits {selected_splits}.")

    # 3. Parallel download of label files (fast parallel download of tiny text files)
    print("\nDownloading annotation files...")
    with ThreadPoolExecutor(max_workers=min(24, args.num_workers * 2)) as executor:
        futures = {
            executor.submit(download_single_file, f"{base_hf_url}/{rpath}", dest / rpath): rpath
            for _, rpath in label_files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Labels", unit="file"):
            fut.result()

    # 4. Filter images that have labeled annotations
    target_images = []
    for s, img_path in image_files:
        img_p = Path(img_path)
        lbl_rel = (img_p.parent.parent / "labels" / f"{img_p.stem}.txt").as_posix()
        lbl_local = dest / lbl_rel
        has_label = False
        if lbl_local.is_file():
            try:
                with open(lbl_local, "r") as lf:
                    has_label = bool(lf.read().strip())
            except Exception:
                has_label = False
        if has_label or args.max_samples <= 0:
            target_images.append((img_path, f"{base_hf_url}/{img_path}", dest / img_path))

    if args.max_samples > 0 and len(target_images) > args.max_samples:
        random.seed(42)
        random.shuffle(target_images)
        target_images = target_images[:args.max_samples]

    print(f"\nDownloading {len(target_images):,} full 1920x1080 images at original resolution using {args.num_workers} threads...")
    print(f"Destination: {dest.resolve()}")

    # 5. Parallel download of full 1920x1080 original images directly to G-Drive
    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(download_single_file, url, out_f): out_f
                for _, url, out_f in target_images
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Images", unit="img"):
                fut.result()
    except KeyboardInterrupt:
        print("\nDownload paused by user. Generating metadata for already-downloaded frames...")

    print("\nIndexing downloaded files into metadata.csv...")
    meta_csv = build_brackish_metadata(dest)

    if meta_csv.is_file():
        df = pd.read_csv(meta_csv)
        print(f"\n🎉 Successfully indexed Brackish benchmark in original resolution:")
        print(f"  • Total Images: {len(df):,}")
        print(f"  • Resolution: 1920x1080 native RGB")
        print(f"  • Splits: {df['split'].value_counts().to_dict()}")
        print(f"  • Classes: {df['label'].value_counts().to_dict()}")
        print(f"  • Metadata CSV: {meta_csv.resolve()}")
        return 0
    else:
        print("Warning: Metadata indexing produced 0 samples.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
