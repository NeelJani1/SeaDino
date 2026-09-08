#!/usr/bin/env python3
"""
High-Speed Parallel Shard Hasher
Distributes WebDataset / tar shards across CPU workers.
"""

import os
from pathlib import Path
import tarfile
import io
import glob
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import imagehash
import numpy as np
from tqdm import tqdm

SHARD_DIR = "/home/njan320/ssl_pretrain_shards"  # Verify this path
OUT_HASHES = Path("benthicnet_shard_hashes.npz")
HASH_SIZE = 8
NUM_WORKERS = max(1, os.cpu_count() - 2)


def process_single_shard(tar_path):
    """Worker function: Extracts and hashes all images inside one tar shard."""
    records = []
    shard_name = Path(tar_path).name
    try:
        with tarfile.open(tar_path, "r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue
                if not member.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                try:
                    f = tar.extractfile(member)
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                    h = imagehash.phash(img, hash_size=HASH_SIZE)
                    records.append((f"{shard_name}:{member.name}", int(str(h), 16)))
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading shard {tar_path}: {e}")
    return records


def main():
    shards = sorted(glob.glob(f"{SHARD_DIR}/shard-*.tar"))
    if not shards:
        shards = sorted(glob.glob(f"{SHARD_DIR}/*.tar"))
    print(f"Found {len(shards)} shards under {SHARD_DIR}")
    if not shards:
        print("No shards found! Check directory path.")
        return

    all_keys = []
    all_hashes = []

    print(f"Hashing shards in parallel using {NUM_WORKERS} workers...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for result in tqdm(executor.map(process_single_shard, shards), total=len(shards), desc="Processing Shards"):
            for key, h in result:
                all_keys.append(key)
                all_hashes.append(h)

    print(f"\nSuccessfully hashed {len(all_hashes):,} shard images.")
    # Cache to disk so you never have to recompute this
    np.savez_compressed(OUT_HASHES, keys=np.array(all_keys), hashes=np.array(all_hashes, dtype=np.uint64))
    print(f"Saved hash registry to: {OUT_HASHES.resolve()}")


if __name__ == "__main__":
    main()