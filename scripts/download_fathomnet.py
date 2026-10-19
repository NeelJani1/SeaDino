#!/usr/bin/env python3
"""FathomNet Benthic Organism Downloader - Full Dataset at Original Resolution.

Downloads the complete FathomNet benchmark dataset at original uncropped resolution (1920x1080)
with original COCO annotations and metadata directly to G-Drive (/mnt/g/data/benchmarks/fathomnet).
Supports downloading all images from MBARI source URLs with resume support, multi-threading,
and deterministic train/validation indexing.
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

from eval_suite.config import DATA_DIR, GDRIVE_CANDIDATE_PATHS


def default_fathomnet_dest() -> Path:
    for g in GDRIVE_CANDIDATE_PATHS:
        try:
            if g.is_dir():
                return g / "benchmarks" / "fathomnet"
        except (PermissionError, OSError):
            continue
    return DATA_DIR / "benchmarks" / "fathomnet"


def download_original_image(url: str, out_path: Path, timeout: int = 25) -> bool:
    """Fetches full uncropped image at original resolution with resume capability."""
    if out_path.is_file() and out_path.stat().st_size > 0:
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "SeaDino-FathomNet-Downloader/1.0"}

    try:
        resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
        if resp.status_code == 200:
            with open(out_path, "wb") as f:
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
                with open(out_path, "wb") as f:
                    f.write(content)
                return True
    except Exception:
        pass
    return False


def main():
    p = argparse.ArgumentParser(description="Download full FathomNet dataset at original resolution to G-Drive")
    p.add_argument("--dest_dir", type=str, default=str(default_fathomnet_dest()),
                   help="Destination directory on G drive (defaults to /mnt/g/data/benchmarks/fathomnet).")
    p.add_argument("--repo_id", type=str, default="FathomNet/2024-vme-benchmark",
                   help="Hugging Face repository ID.")
    p.add_argument("--max_samples", type=int, default=-1,
                   help="Maximum samples to download (-1 for full dataset).")
    p.add_argument("--num_workers", type=int, default=12,
                   help="Concurrent image download workers (default: 12).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    dest = Path(args.dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    img_dir = dest / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target FathomNet Directory: {dest.resolve()}")
    print(f"Loading COCO annotations from Hugging Face Hub '{args.repo_id}'...")

    coco_data = None
    coco_path_dest = dest / "coco_train.json"

    try:
        from huggingface_hub import hf_hub_download
        coco_candidates = ["coco_train.json", "train.json", "annotations.json"]
        for cname in coco_candidates:
            try:
                hub_path = hf_hub_download(repo_id=args.repo_id, filename=cname, repo_type="dataset")
                if hub_path and os.path.exists(hub_path):
                    with open(hub_path, "r", encoding="utf-8") as f:
                        coco_data = json.load(f)
                    # Also save a copy directly in G-Drive destination
                    with open(coco_path_dest, "w", encoding="utf-8") as f_out:
                        json.dump(coco_data, f_out)
                    print(f"  Found and saved COCO annotations: {coco_path_dest.resolve()}")
                    break
            except Exception:
                continue
    except Exception as e:
        print(f"  [COCO Hub note: {e}]")

    if not coco_data and coco_path_dest.is_file():
        with open(coco_path_dest, "r", encoding="utf-8") as f:
            coco_data = json.load(f)
        print(f"  Loaded existing local annotations: {coco_path_dest.resolve()}")

    if not coco_data:
        print("Error: Could not locate or download COCO annotation JSON.")
        return 1

    id2cat = {c["id"]: c.get("name", str(c["id"])) for c in coco_data.get("categories", [])}
    id2img = {img["id"]: img for img in coco_data.get("images", [])}

    print(f"  • Categories: {id2cat}")
    print(f"  • Total Images: {len(id2img):,}")
    print(f"  • Total Annotations: {len(coco_data.get('annotations', [])):,}")

    # Build image download candidates
    img2anns = {}
    for ann in coco_data.get("annotations", []):
        img2anns.setdefault(ann.get("image_id"), []).append(ann)

    items_to_download = []
    for img_id, img_info in id2img.items():
        url = img_info.get("source_url") or img_info.get("coco_url") or img_info.get("url") or img_info.get("image_url")
        if not url:
            continue
        fname = img_info.get("file_name", f"{img_id}.jpg")
        anns = img2anns.get(img_id, [])
        primary_label = "unknown"
        primary_bbox = None
        if anns:
            first_ann = anns[0]
            cat_id = first_ann.get("category_id")
            primary_label = id2cat.get(cat_id, str(cat_id))
            primary_bbox = first_ann.get("bbox")

        split_name = img_info.get("split", "train")
        items_to_download.append({
            "image_id": img_id,
            "file_name": f"images/{fname}",
            "dest_path": img_dir / fname,
            "url": url,
            "label": str(primary_label).strip(),
            "bbox": primary_bbox,
            "split": split_name,
            "width": img_info.get("width", 1920),
            "height": img_info.get("height", 1080),
        })

    if args.max_samples > 0 and len(items_to_download) > args.max_samples:
        random.seed(args.seed)
        random.shuffle(items_to_download)
        items_to_download = items_to_download[:args.max_samples]

    print(f"\nDownloading {len(items_to_download):,} original uncropped 1920x1080 images with {args.num_workers} threads...")
    print(f"Destination: {dest.resolve()}")

    successful_records = []
    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {
                executor.submit(download_original_image, item["url"], item["dest_path"]): item
                for item in items_to_download
            }
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Images", unit="img"):
                item = futures[fut]
                if fut.result() and item["dest_path"].is_file():
                    successful_records.append({
                        "file_name": item["file_name"],
                        "label": item["label"],
                        "bbox": str(item["bbox"]) if item["bbox"] else "",
                        "split": item["split"],
                        "width": item["width"],
                        "height": item["height"],
                        "url": item["url"],
                    })
    except KeyboardInterrupt:
        print("\nDownload paused by user. Saving metadata for already-downloaded frames...")

    meta_csv = dest / "metadata.csv"
    if successful_records:
        meta_df = pd.DataFrame(successful_records)
        meta_df.to_csv(meta_csv, index=False)
        print(f"\n🎉 Successfully indexed FathomNet benchmark in original resolution:")
        print(f"  • Total Images: {len(meta_df):,}")
        print(f"  • Resolution: 1920x1080 native MBARI frames")
        print(f"  • Splits: {meta_df['split'].value_counts().to_dict()}")
        print(f"  • Classes: {meta_df['label'].value_counts().to_dict()}")
        print(f"  • Metadata CSV: {meta_csv.resolve()}")
        return 0
    else:
        print("Error: No images were successfully downloaded.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
