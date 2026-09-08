#!/usr/bin/env python3
"""Perceptual Hash Threshold Sensitivity Sweep (d in [0, 10]) for CoralMask vs BenthicNet Shards."""

from pathlib import Path
import numpy as np
from PIL import Image
import imagehash
from tqdm import tqdm

SHARD_NPZ = Path("/home/njan320/SeaDino/benthicnet_shard_hashes.npz")
CM_TEST_DIR = Path("/home/njan320/Neel/CoralMaskv1/CoralMask/test/images")

print(f"Loading shard hashes from {SHARD_NPZ}...")
shard_data = np.load(SHARD_NPZ)
shard_arr = shard_data["hashes"]
print(f"Loaded {len(shard_arr):,} SSL shard hashes.")

test_files = sorted([p for p in CM_TEST_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
print(f"Found {len(test_files):,} CoralMask test images.")

cm_hashes = []
for p in tqdm(test_files, desc="Hashing Test Images"):
    with Image.open(p) as img:
        cm_hashes.append(int(str(imagehash.phash(img.convert("RGB"), hash_size=8)), 16))
cm_arr = np.array(cm_hashes, dtype=np.uint64)


def popcount64(arr):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(arr).astype(np.uint8)
    v = arr
    v = v - ((v >> np.uint64(1)) & np.uint64(0x5555555555555555))
    v = (v & np.uint64(0x3333333333333333)) + ((v >> np.uint64(2)) & np.uint64(0x3333333333333333))
    v = (v + (v >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return ((v * np.uint64(0x0101010101010101)) >> np.uint64(56)).astype(np.uint8)


min_dists = []
BLOCK_SIZE = 256
for i in range(0, len(cm_arr), BLOCK_SIZE):
    q_block = cm_arr[i : i + BLOCK_SIZE]
    xor_mat = np.bitwise_xor(q_block[:, None], shard_arr[None, :])
    dist_mat = popcount64(xor_mat)
    min_dists.extend(dist_mat.min(axis=1).tolist())

min_dists = np.array(min_dists)

print("\n" + "=" * 80)
print("CUMULATIVE TEST CONTAMINATION BY HAMMING DISTANCE:")
print("=" * 80)
for d in range(0, 11):
    count = int((min_dists <= d).sum())
    pct = (count / len(cm_arr)) * 100
    print(f"  Threshold <= {d:2d}:  {count:3d} images  ({pct:5.2f}%)")
print("=" * 80)

# Save test min distances
out_npz = Path("coralmask_test_min_dists.npz")
file_stems = [p.stem for p in test_files]
np.savez(out_npz, stems=file_stems, min_dists=min_dists)
print(f"\nSaved test distance distribution to: {out_npz}")

# Materialize Clean Test Split (N = 823)
leaked_file = Path("coralmask_test_leakage_ids.txt")
if leaked_file.exists():
    leaked_ids = set(leaked_file.read_text().split())
    clean_test_imgs = [p for p in test_files if p.stem not in leaked_ids]
    clean_manifest = Path("coralmask_test_clean.txt")
    clean_manifest.write_text("\n".join(str(p) for p in clean_test_imgs))
    print(f"CoralMask-Clean test manifest written to: {clean_manifest}")
    print(f"  Total Clean Test Images: {len(clean_test_imgs)} / {len(test_files)} (Excluded {len(leaked_ids)} leaked images)")
