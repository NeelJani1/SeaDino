#!/usr/bin/env python3
"""
Final Decisive Shard Cross-Match
Audits CoralMask Test & Train against the 189,101 BenthicNet SSL Shards.
"""

import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

# --- CONFIGURATION ---
SHARD_NPZ = Path("/home/njan320/SeaDino/benthicnet_shard_hashes.npz")
CM_TEST_DIR = Path("/home/njan320/Neel/CoralMaskv1/CoralMask/test/images")
CM_TRAIN_DIR = Path("/home/njan320/Neel/CoralMaskv1/CoralMask/train/images")

HASH_SIZE = 8
STRICT_THRESHOLD = 4   # Definite identical image
SUSPECT_THRESHOLD = 6  # Re-compressed / consecutive transect duplicate
NUM_WORKERS = max(1, os.cpu_count() - 2)


def _hash_single_image(path_str):
    try:
        with Image.open(path_str) as img:
            h = imagehash.phash(img.convert("RGB"), hash_size=HASH_SIZE)
            return path_str, int(str(h), 16)
    except Exception:
        return None


def parallel_hash(paths, desc="Hashing"):
    valid_paths, hashes = [], []
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for res in tqdm(executor.map(_hash_single_image, paths, chunksize=128), total=len(paths), desc=desc):
            if res is not None:
                p, h = res
                valid_paths.append(p)
                hashes.append(h)
    return valid_paths, np.array(hashes, dtype=np.uint64)


def popcount64(arr):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(arr).astype(np.uint8)
    v = arr
    v = v - ((v >> np.uint64(1)) & np.uint64(0x5555555555555555))
    v = (v & np.uint64(0x3333333333333333)) + ((v >> np.uint64(2)) & np.uint64(0x3333333333333333))
    v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((v * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


def match_arrays(query_paths, query_arr, target_keys, target_arr, label="Query"):
    print(f"\nComparing {len(query_arr):,} {label} images against {len(target_arr):,} SSL Shard images...")
    matches = []
    BLOCK_SIZE = 256

    for i in tqdm(range(0, len(query_arr), BLOCK_SIZE), desc=f"Matching {label}"):
        q_block = query_arr[i : i + BLOCK_SIZE]
        xor_mat = np.bitwise_xor(q_block[:, None], target_arr[None, :])
        dist_mat = popcount64(xor_mat)

        hit_q, hit_t = np.where(dist_mat <= SUSPECT_THRESHOLD)
        for qi, ti in zip(hit_q, hit_t):
            d = int(dist_mat[qi, ti])
            matches.append((query_paths[i + qi], target_keys[ti], d))

    return matches


def main():
    print("=" * 80)
    print("DECISIVE AUDIT: CORALMASK vs. BENTHICNET SSL SHARDS (189,101 IMAGES)")
    print("=" * 80)

    # 1. Load Precomputed Shard Hashes
    print(f"Loading shard hashes from {SHARD_NPZ}...")
    data = np.load(SHARD_NPZ)
    shard_keys, shard_arr = data["keys"], data["hashes"]
    print(f"Loaded {len(shard_arr):,} SSL pretraining shard hashes.")

    valid_exts = {".jpg", ".jpeg", ".png"}

    # 2. Audit CoralMask TEST Set
    test_files = [str(p) for p in CM_TEST_DIR.rglob("*") if p.suffix.lower() in valid_exts]
    cm_test_paths, cm_test_arr = parallel_hash(test_files, desc="Hashing CoralMask Test")
    test_matches = match_arrays(cm_test_paths, cm_test_arr, shard_keys, shard_arr, label="CoralMask Test")

    # 3. Audit CoralMask TRAIN Set
    train_files = [str(p) for p in CM_TRAIN_DIR.rglob("*") if p.suffix.lower() in valid_exts]
    cm_train_paths, cm_train_arr = parallel_hash(train_files, desc="Hashing CoralMask Train")
    train_matches = match_arrays(cm_train_paths, cm_train_arr, shard_keys, shard_arr, label="CoralMask Train")

    # 4. Summary & Decontamination Export
    print("\n" + "=" * 80)
    print("AUDIT SUMMARY REPORT")
    print("=" * 80)

    test_definite = [m for m in test_matches if m[2] <= STRICT_THRESHOLD]
    test_suspect = [m for m in test_matches if m[2] > STRICT_THRESHOLD]
    print(f"\n1. CoralMask TEST Set (830 total images):")
    print(f"   - Definite Duplicate Frames (Dist <= {STRICT_THRESHOLD}): {len(test_definite)}")
    print(f"   - Near-Frame Duplicates (Dist 5-{SUSPECT_THRESHOLD}):     {len(test_suspect)}")
    print(f"   - Total Contaminated Test Images:          {len(set(m[0] for m in test_matches))}")

    train_definite = [m for m in train_matches if m[2] <= STRICT_THRESHOLD]
    train_suspect = [m for m in train_matches if m[2] > STRICT_THRESHOLD]
    print(f"\n2. CoralMask TRAIN Set ({len(train_files):,} total images):")
    print(f"   - Definite Duplicate Frames (Dist <= {STRICT_THRESHOLD}): {len(train_definite)}")
    print(f"   - Near-Frame Duplicates (Dist 5-{SUSPECT_THRESHOLD}):     {len(train_suspect)}")
    print(f"   - Total Contaminated Train Images:         {len(set(m[0] for m in train_matches))}")

    # Export test leakage IDs for CoralMask-Clean
    leaked_test_stems = sorted(list(set(Path(m[0]).stem for m in test_matches)))
    out_txt = Path("coralmask_test_leakage_ids.txt")
    out_txt.write_text("\n".join(leaked_test_stems))
    print(f"\nSaved {len(leaked_test_stems)} contaminated test stems to: {out_txt.resolve()}")
    print("Use this file to filter test set and create CoralMask-Clean.")
    print("=" * 80)


if __name__ == "__main__":
    main()