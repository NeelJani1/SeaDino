import os
import sys
from pathlib import Path
import ast
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_suite.config import DATA_DIR

# Paths
BASE_DIR = os.getenv("BENTHICNET_BASE_DIR", "/home/njan320")
MASTER_CSV = os.getenv("MASTER_CSV", f"{BASE_DIR}/Neel/BenthicNet/csvs/finalized_csvs/benthicnet_labelled.csv")
BIOTA_CSV = os.getenv("BIOTA_CSV", f"{BASE_DIR}/Neel/BenthicNet/csvs/finalized_csvs/trainable/benthicnet_nn.csv")
OUTPUT_BIOTA_CSV = str(DATA_DIR / "splits" / "biota_spatial_split.csv")

EARTH_RADIUS_M = 6_371_000.0

def _parse(raw):
    if isinstance(raw, str):
        try:
            p = ast.literal_eval(raw)
            return list(p) if isinstance(p, (list, tuple, set)) else [int(p)]
        except Exception:
            return [int(x.strip()) for x in raw.strip("[]'\" ").split(",") if x.strip().isdigit()]
    elif isinstance(raw, (list, tuple)):
        return [int(x) for x in raw]
    return [int(raw)]

def compute_nearest_haversine(train_df, test_df):
    train_rad = np.radians(train_df[["latitude", "longitude"]].to_numpy())
    test_rad = np.radians(test_df[["latitude", "longitude"]].to_numpy())
    tree = BallTree(train_rad, metric="haversine")
    dists_rad, idxs = tree.query(test_rad, k=1)
    dists_m = dists_rad.flatten() * EARTH_RADIUS_M
    return dists_m, idxs.flatten()

def task1_diagnose(df_merged, benchmark_name):
    print("\n" + "=" * 80)
    print(f"TASK 1 DIAGNOSIS: {benchmark_name}")
    print("=" * 80)

    train = df_merged[df_merged["partition"].str.lower() == "train"].reset_index(drop=True)
    test = df_merged[df_merged["partition"].str.lower().isin(["test", "val", "validation"])].reset_index(drop=True)

    print(f"Total merged samples: {len(df_merged):,}")
    print(f"Train: {len(train):,} | Test: {len(test):,}")

    dists_m, nearest_train_idx = compute_nearest_haversine(train, test)

    print("\nDistance percentiles (Test -> Nearest Train):")
    for p in [0, 1, 5, 10, 25, 50, 75, 90, 99, 100]:
        print(f"  {p:>3}th percentile: {np.percentile(dists_m, p):>10.2f} meters")

    print("\nTest points within distance thresholds of ANY train point:")
    thresholds = [0.001, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    for t in thresholds:
        cnt = (dists_m < t).sum()
        pct = (cnt / len(dists_m)) * 100.0
        label = f"< {t}m" if t >= 1.0 else "== 0.0m (exact)"
        print(f"  {label:<18}: {cnt:>6,} / {len(dists_m):,} ({pct:>5.2f}%)")

    return dists_m

def build_spatial_block_split(df, cell_size_m=500.0, buffer_m=100.0, test_frac=0.20, seed=42):
    df = df.copy().reset_index(drop=True)
    lat_rad = np.radians(df["latitude"].to_numpy())
    m_per_deg_lat = 111_139.0
    m_per_deg_lon = 111_139.0 * np.cos(lat_rad)

    df["grid_y"] = (df["latitude"] * m_per_deg_lat / cell_size_m).astype(int)
    df["grid_x"] = (df["longitude"] * m_per_deg_lon / cell_size_m).astype(int)
    df["spatial_cell"] = df["dataset"].astype(str) + "_" + df["grid_y"].astype(str) + "_" + df["grid_x"].astype(str)

    rng = np.random.default_rng(seed)
    unique_cells = np.array(sorted(df["spatial_cell"].unique()))
    rng.shuffle(unique_cells)

    total_images = len(df)
    target_test_images = int(total_images * test_frac)
    cell_counts = df["spatial_cell"].value_counts().to_dict()

    test_cells = set()
    accum_test = 0
    for cell in unique_cells:
        test_cells.add(cell)
        accum_test += cell_counts[cell]
        if accum_test >= target_test_images:
            break

    df["initial_partition"] = np.where(df["spatial_cell"].isin(test_cells), "test", "train")

    train_subset = df[df["initial_partition"] == "train"]
    test_subset = df[df["initial_partition"] == "test"]

    print(f"  Pre-buffer: {len(train_subset):,} train, {len(test_subset):,} test")

    test_rad = np.radians(test_subset[["latitude", "longitude"]].to_numpy())
    train_rad = np.radians(train_subset[["latitude", "longitude"]].to_numpy())

    tree = BallTree(test_rad, metric="haversine")
    dists_rad, _ = tree.query(train_rad, k=1)
    train_to_test_m = dists_rad.flatten() * EARTH_RADIUS_M

    train_keep_mask = train_to_test_m >= buffer_m
    dropped_train_count = (~train_keep_mask).sum()

    train_indices = train_subset.index.to_numpy()
    kept_train_indices = set(train_indices[train_keep_mask])

    final_partition = []
    for idx in df.index:
        if idx in test_subset.index:
            final_partition.append("test")
        elif idx in kept_train_indices:
            final_partition.append("train")
        else:
            final_partition.append("dropped")

    df["partition"] = final_partition
    print(f"  Buffer filtering ({buffer_m}m): dropped {dropped_train_count:,} train images")
    return df

def task3_validate(df_split, benchmark_name, buffer_m):
    print("\n" + "=" * 80)
    print(f"TASK 3 VALIDATION: {benchmark_name}")
    print("=" * 80)

    train = df_split[df_split["partition"] == "train"].reset_index(drop=True)
    test = df_split[df_split["partition"] == "test"].reset_index(drop=True)

    print(f"Partition counts:")
    print(f"  Train:   {len(train):,} ({len(train)/len(df_split)*100:.1f}%)")
    print(f"  Test:    {len(test):,} ({len(test)/len(df_split)*100:.1f}%)")

    train_counts = Counter()
    for row in train["catami_biota"]:
        train_counts.update(_parse(row))
    
    test_counts = Counter()
    for row in test["catami_biota"]:
        test_counts.update(_parse(row))

    all_classes = sorted(list(set(train_counts.keys()) | set(test_counts.keys())))
    print(f"\n[Check] Test set class counts (Total classes: {len(all_classes)}):")
    
    # We want to identify if any of the 272 classes drop to 0 or very low in the test set
    min_test_count = min(test_counts.get(c, 0) for c in all_classes)
    low_test_classes = [(c, test_counts.get(c, 0)) for c in all_classes if test_counts.get(c, 0) < 5]
    print(f"  Minimum test occurrences for any class: {min_test_count}")
    if low_test_classes:
        print(f"  WARNING: {len(low_test_classes)} classes have < 5 test instances!")
        for c, count in low_test_classes:
            print(f"    Class {c}: {count} instances")
    
    print("\n  -> VALIDATION COMPLETED.")
    return len(low_test_classes)

def main():
    print("Loading coordinates...")
    coords_df = pd.read_csv(
        MASTER_CSV,
        low_memory=False,
        usecols=["image", "longitude", "latitude"]
    ).dropna(subset=["longitude", "latitude"]).drop_duplicates(subset=["image"])
    
    print("Loading Biota...")
    biota_raw = pd.read_csv(BIOTA_CSV, low_memory=False).dropna(subset=["catami_biota"])
    
    df_merged = biota_raw.merge(coords_df, on="image", how="inner")
    print(f"Merged Biota rows: {len(df_merged):,}")

    task1_diagnose(df_merged, "Biota (Original Split)")

    # Try cell=250m, buffer=50m (similar to German Bank, as Biota has many fine-grained classes)
    cell_size = 250.0
    buffer_m = 50.0
    print(f"\nBuilding split: cell={cell_size}m, buffer={buffer_m}m")
    
    df_split = build_spatial_block_split(df_merged, cell_size_m=cell_size, buffer_m=buffer_m, test_frac=0.20, seed=42)
    task3_validate(df_split, f"Biota (New Split - Cell:{cell_size}m, Buffer:{buffer_m}m)", buffer_m)
    
    df_split.to_csv(OUTPUT_BIOTA_CSV, index=False)
    print(f"Saved to {OUTPUT_BIOTA_CSV}")

if __name__ == "__main__":
    main()
